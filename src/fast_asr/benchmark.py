"""Self-contained direct-ORT Whisper benchmark with portable JSONL results."""

from __future__ import annotations

import json
import resource
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import onnxruntime as ort
from datasets import load_dataset
from tokenizers import Tokenizer
from transformers import WhisperFeatureExtractor

from fast_asr.evaluation import load_records, write_evaluation_report
from fast_asr.provenance import write_provenance


def _session(path: Path, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_mem_pattern = False
    return ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])


def _prompt(tokenizer: Tokenizer) -> np.ndarray:
    ids = [tokenizer.token_to_id(token) for token in ["<|startoftranscript|>", "<|notimestamps|>"]]
    if any(token is None for token in ids):
        raise ValueError("Whisper tokenizer is missing required prompt tokens.")
    return np.asarray([ids], dtype=np.int32)


def _transcribe(
    encoder: ort.InferenceSession,
    decoder: ort.InferenceSession,
    feature_extractor: Any,
    tokenizer: Tokenizer,
    audio: np.ndarray,
    sample_rate: int,
    max_tokens: int,
) -> tuple[str, dict[str, float | int]]:
    start = time.perf_counter()
    features = feature_extractor(
        audio, sampling_rate=sample_rate, return_tensors="np"
    ).input_features.astype(np.float32)
    encoder_start = time.perf_counter()
    encoded = encoder.run(None, {"audio_features": features})
    encoder_ms = (time.perf_counter() - encoder_start) * 1_000
    encoder_outputs = dict(
        zip(
            [item.name for item in encoder.get_outputs()],
            cast(list[np.ndarray], encoded),
            strict=True,
        )
    )
    prompt_ids = _prompt(tokenizer)
    prompt_length = int(prompt_ids.shape[1])
    feeds: dict[str, np.ndarray] = {"input_ids": prompt_ids}
    for item in decoder.get_inputs():
        if item.name.startswith(("past_key_cross_", "past_value_cross_")):
            feeds[item.name] = encoder_outputs[item.name.replace("past_", "present_", 1)]
        elif item.name.startswith(("past_key_self_", "past_value_self_")):
            feeds[item.name] = np.empty((1, 8, 0, 64), dtype=np.float32)
    output_names = [item.name for item in decoder.get_outputs()]
    token_ids = list(feeds["input_ids"][0])
    decode_start = time.perf_counter()
    decoder_first_token_ms: float | None = None
    ttft_ms: float | None = None
    eos = tokenizer.token_to_id("<|endoftext|>")
    reached_eos = False
    for _ in range(max_tokens - prompt_length):
        outputs = dict(
            zip(output_names, cast(list[np.ndarray], decoder.run(None, feeds)), strict=True)
        )
        if outputs["logits"].shape[1] == 0:
            message = "Decoder returned zero logits; refusing to score a partial transcript."
            raise RuntimeError(message)
        token = int(np.argmax(outputs["logits"][0, -1]))
        if decoder_first_token_ms is None:
            decoder_first_token_ms = (time.perf_counter() - decode_start) * 1_000
            ttft_ms = (time.perf_counter() - start) * 1_000
        token_ids.append(token)
        if token == eos:
            reached_eos = True
            break
        feeds = {"input_ids": np.asarray([[token]], dtype=np.int32)}
        for name, value in outputs.items():
            if name.startswith(("present_key_self_", "present_value_self_")):
                feeds[name.replace("present_", "past_")] = value
        for name, value in encoder_outputs.items():
            if name.startswith(("present_key_cross_", "present_value_cross_")):
                feeds[name.replace("present_", "past_")] = value
    decode_ms = (time.perf_counter() - decode_start) * 1_000
    e2e_ms = (time.perf_counter() - start) * 1_000
    generated_tokens = len(token_ids) - prompt_length
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip(), {
        "e2e_latency_ms": e2e_ms,
        "encoder_latency_ms": encoder_ms,
        "first_token_latency_ms": decoder_first_token_ms or decode_ms,
        "ttft_ms": ttft_ms or e2e_ms,
        "decode_latency_ms": decode_ms,
        "generated_tokens": generated_tokens,
        "tpot_ms": decode_ms / max(generated_tokens, 1),
        "truncated": not reached_eos,
    }


def _transcribe_audio(
    encoder: ort.InferenceSession,
    decoder: ort.InferenceSession,
    feature_extractor: Any,
    tokenizer: Tokenizer,
    audio: np.ndarray,
    sample_rate: int,
    max_tokens: int,
) -> tuple[str, dict[str, float | int | bool]]:
    """Transcribe arbitrary-length audio as sequential 30-second Whisper chunks."""
    chunk_samples = 30 * sample_rate
    outputs = [
        _transcribe(
            encoder,
            decoder,
            feature_extractor,
            tokenizer,
            audio[start : start + chunk_samples],
            sample_rate,
            max_tokens,
        )
        for start in range(0, len(audio), chunk_samples)
    ]
    texts = [text for text, _ in outputs]
    timings = [timing for _, timing in outputs]
    generated_tokens = sum(int(item["generated_tokens"]) for item in timings)
    decode_ms = sum(float(item["decode_latency_ms"]) for item in timings)
    return " ".join(texts), {
        "e2e_latency_ms": sum(float(item["e2e_latency_ms"]) for item in timings),
        "encoder_latency_ms": sum(float(item["encoder_latency_ms"]) for item in timings),
        "first_token_latency_ms": float(timings[0]["first_token_latency_ms"]),
        "decode_latency_ms": decode_ms,
        "generated_tokens": generated_tokens,
        "tpot_ms": decode_ms / max(generated_tokens, 1),
        "truncated": any(bool(item["truncated"]) for item in timings),
    }


def benchmark_librispeech(
    model_directory: Path,
    output_directory: Path,
    *,
    split: str = "test.clean",
    max_samples: int = 64,
    threads: int = 4,
    max_tokens: int = 448,
) -> dict[str, Any]:
    """Benchmark one artifact and write records plus a scored summary."""
    if max_samples < 0:
        raise ValueError("max_samples must be non-negative; zero means the full split.")
    output_directory.mkdir(parents=True, exist_ok=True)
    provenance_path = output_directory / "provenance.json"
    requested_provenance = write_provenance(
        output_directory / "provenance.pending.json",
        model_directory,
        split=split,
        max_samples=max_samples,
        threads=threads,
        max_tokens=max_tokens,
    )
    if provenance_path.is_file():
        existing = json.loads(provenance_path.read_text(encoding="utf-8"))
        if existing.get("settings") != requested_provenance.get("settings"):
            raise ValueError("Cannot resume: provenance settings changed.")
        existing_artifact = dict(existing.get("artifact", {}))
        requested_artifact = dict(requested_provenance.get("artifact", {}))
        existing_artifact.pop("location", None)
        requested_artifact.pop("location", None)
        if existing_artifact != requested_artifact:
            raise ValueError("Cannot resume: artifact identity changed.")
        (output_directory / "provenance.pending.json").unlink()
    else:
        (output_directory / "provenance.pending.json").replace(provenance_path)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_directory)
    tokenizer = Tokenizer.from_file(str(model_directory / "tokenizer.json"))
    encoder = _session(model_directory / "encoder.onnx", threads)
    decoder = _session(model_directory / "decoder.onnx", threads)
    if split in {"validation.clean", "validation.other"}:
        subset = split.rsplit(".", maxsplit=1)[1]
        loaded_dataset = load_dataset(
            "openslr/librispeech_asr",
            subset,
            split="validation",
            streaming=bool(max_samples),
        )
    else:
        loaded_dataset = load_dataset(
            "hf-audio/esb-datasets-test-only-sorted", "librispeech", split=split
        )
    dataset: Any = cast(Any, loaded_dataset)
    if max_samples:
        dataset = (
            dataset.take(max_samples)
            if hasattr(dataset, "take")
            else dataset.select(range(min(max_samples, len(dataset))))
        )
    records_path = output_directory / "records.jsonl"
    previous_records = load_records(records_path) if records_path.is_file() else []
    completed = {str(record["utterance_id"]) for record in previous_records}
    if len(completed) != len(previous_records):
        raise ValueError("Cannot resume records containing duplicate utterance IDs.")
    manifest_path = output_directory / "sample_ids.json"
    previous_sample_ids = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else []
    )
    sample_ids: list[str] = []
    with records_path.open("a", encoding="utf-8") as records:
        for index, sample in enumerate(dataset):
            utterance_id = str(sample.get("id", index))
            if index < len(previous_sample_ids) and previous_sample_ids[index] != utterance_id:
                raise ValueError("Cannot resume: dataset sample IDs changed.")
            sample_ids.append(utterance_id)
            if utterance_id in completed:
                continue
            audio = sample["audio"]
            prediction, timings = _transcribe_audio(
                encoder,
                decoder,
                feature_extractor,
                tokenizer,
                audio["array"],
                audio["sampling_rate"],
                max_tokens,
            )
            record = {
                "utterance_id": utterance_id,
                "reference": sample["text"],
                "prediction": prediction,
                "audio_duration_s": len(audio["array"]) / audio["sampling_rate"],
                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                **timings,
            }
            records.write(json.dumps(record) + "\n")
            records.flush()
            temporary_manifest = manifest_path.with_suffix(".json.tmp")
            temporary_manifest.write_text(
                json.dumps(sample_ids, indent=2) + "\n", encoding="utf-8"
            )
            temporary_manifest.replace(manifest_path)
    if len(previous_sample_ids) > len(sample_ids):
        raise ValueError("Cannot resume: dataset contains fewer sample IDs.")
    return write_evaluation_report(records_path, output_directory / "summary.json", model_directory)
