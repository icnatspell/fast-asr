"""LoRA and teacher-distillation recovery training for compressed Whisper."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import jiwer
import numpy as np
import torch
import torch.nn.functional as functional
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import (
    EvalPrediction,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers.trainer_utils import get_last_checkpoint


@dataclass(frozen=True)
class RecoveryConfig:
    """Serializable architecture and recovery-training configuration."""

    encoder_stride_factor: int = 2
    hidden_pool_factor: int = 1
    lora_rank: int = 8
    learning_rate: float = 1e-4
    max_steps: int = 1_000
    kl_weight: float = 0.5
    hidden_weight: float = 0.25
    max_train_samples: int = 0
    preprocessing_workers: int = 4
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    seed: int = 42
    eval_steps: int = 50
    eval_samples_per_split: int = 64
    generation_max_length: int = 128

    def __post_init__(self) -> None:
        positive_integers = {
            "encoder_stride_factor": self.encoder_stride_factor,
            "hidden_pool_factor": self.hidden_pool_factor,
            "lora_rank": self.lora_rank,
            "max_steps": self.max_steps,
            "preprocessing_workers": self.preprocessing_workers,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "eval_steps": self.eval_steps,
            "eval_samples_per_split": self.eval_samples_per_split,
            "generation_max_length": self.generation_max_length,
        }
        invalid = [name for name, value in positive_integers.items() if value < 1]
        if invalid:
            raise ValueError(f"Positive values required for: {', '.join(invalid)}.")
        if self.max_train_samples < 0:
            raise ValueError("max_train_samples cannot be negative.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.kl_weight < 0 or self.hidden_weight < 0:
            raise ValueError("Distillation weights cannot be negative.")


class WhisperDataCollator:
    """Pad Whisper audio features and decoder labels independently."""

    def __init__(
        self, feature_extractor: WhisperFeatureExtractor, tokenizer: WhisperTokenizer
    ) -> None:
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        inputs = self.feature_extractor.pad(
            [
                {
                    "input_features": item["input_features"],
                    "attention_mask": item["attention_mask"],
                }
                for item in features
            ],
            return_tensors="pt",
        )
        labels = self.tokenizer.pad(
            [{"input_ids": item["labels"]} for item in features], return_tensors="pt"
        )
        label_ids = cast(torch.Tensor, labels["input_ids"])
        label_mask = cast(torch.Tensor, labels["attention_mask"])
        label_ids = label_ids.masked_fill(label_mask.ne(1), -100)
        if (label_ids[:, 0] == self.tokenizer.bos_token_id).all().item():
            label_ids = label_ids[:, 1:]
        return {
            "input_features": inputs["input_features"],
            "attention_mask": inputs["attention_mask"],
            "labels": label_ids,
        }


def _compress_encoder(model: WhisperForConditionalGeneration, config: RecoveryConfig) -> None:
    encoder = model.model.encoder
    factor = config.encoder_stride_factor
    if factor > 1:
        encoder.conv2.stride = (2 * factor,)
        original = encoder.embed_positions.weight.detach()
        positions = nn.Embedding(original.shape[0] // factor, original.shape[1])
        positions.weight.data.copy_(original[::factor])
        encoder.embed_positions = positions
        model.config.max_source_positions = original.shape[0] // factor

    if config.hidden_pool_factor > 1:
        pool_factor = config.hidden_pool_factor

        def pool_hidden_states(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            hidden = output.last_hidden_state
            output.last_hidden_state = functional.avg_pool1d(
                hidden.transpose(1, 2), kernel_size=pool_factor, stride=pool_factor
            ).transpose(1, 2)
            return output

        encoder.register_forward_hook(pool_hidden_states)


def _enable_encoder_input_gradients(model: WhisperForConditionalGeneration) -> None:
    """Keep encoder LoRA gradients alive through gradient checkpointing."""

    def require_grad(
        _module: nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor
    ) -> torch.Tensor:
        output.requires_grad_(True)
        return output

    model.model.encoder.conv1.register_forward_hook(require_grad)


class DistillationTrainer(Seq2SeqTrainer):
    """Combine transcript loss with teacher logits and encoder representations."""

    def __init__(
        self,
        *args: Any,
        teacher: WhisperForConditionalGeneration,
        kl_weight: float,
        hidden_weight: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.teacher = teacher.eval()
        self.teacher.requires_grad_(False)
        self.kl_weight = kl_weight
        self.hidden_weight = hidden_weight

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        del num_items_in_batch
        student = model(**inputs, output_hidden_states=True)
        input_device = inputs["input_features"].device
        if next(self.teacher.parameters()).device != input_device:
            self.teacher.to(input_device)
        with torch.no_grad():
            teacher = self.teacher(**inputs, output_hidden_states=True)
        logits_loss = functional.kl_div(
            functional.log_softmax(student.logits.float(), dim=-1),
            functional.softmax(teacher.logits.float(), dim=-1),
            reduction="batchmean",
        ) / student.logits.shape[1]
        teacher_hidden = teacher.encoder_last_hidden_state
        student_hidden = student.encoder_last_hidden_state
        teacher_hidden = functional.adaptive_avg_pool1d(
            teacher_hidden.transpose(1, 2), student_hidden.shape[1]
        ).transpose(1, 2)
        hidden_loss = functional.mse_loss(student_hidden.float(), teacher_hidden.float())
        loss = student.loss + self.kl_weight * logits_loss + self.hidden_weight * hidden_loss
        return (loss, student) if return_outputs else loss


def train_recovery(output_directory: Path, config: RecoveryConfig) -> None:
    """Train, merge, and save one compressed Whisper recovery model."""
    processor = WhisperProcessor.from_pretrained("openai/whisper-base.en")
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base.en")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base.en")
    student = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base.en")
    teacher = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base.en")
    _compress_encoder(student, config)
    _enable_encoder_input_gradients(student)
    student.config.use_cache = False
    student.config.forced_decoder_ids = None
    student.config.suppress_tokens = []
    teacher.config.forced_decoder_ids = None
    teacher.config.suppress_tokens = []
    lora = LoraConfig(
        r=config.lora_rank,
        lora_alpha=2 * config.lora_rank,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    )
    student = get_peft_model(student, lora)

    def prepare(sample: dict[str, Any]) -> dict[str, Any]:
        audio = sample["audio"]
        features = feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_attention_mask=True,
        )
        return {
            "input_features": features.input_features[0],
            "attention_mask": features.attention_mask[0],
            "labels": tokenizer(sample["text"]).input_ids,
        }

    train_stream = cast(
        Any,
        load_dataset("openslr/librispeech_asr", "clean", split="train.100", streaming=True),
    ).skip(config.eval_samples_per_split)
    if config.max_train_samples > 0:
        train_stream = train_stream.take(config.max_train_samples)
    train_stream = train_stream.shuffle(seed=config.seed, buffer_size=1_000)
    train_data = train_stream.map(prepare, remove_columns=train_stream.column_names)

    def evaluation_slice(subset: str, split: str, *, skip: int = 0) -> Dataset:
        stream = cast(
            Any,
            load_dataset("openslr/librispeech_asr", subset, split=split, streaming=True),
        )
        if skip:
            stream = stream.skip(skip)
        prepared = [prepare(sample) for sample in stream.take(config.eval_samples_per_split)]
        return Dataset.from_list(prepared)

    evaluation_data = {
        "train_holdout": evaluation_slice("clean", "train.100"),
        "validation_clean": evaluation_slice("clean", "validation"),
        "validation_other": evaluation_slice("other", "validation"),
    }
    normalizer = BasicTextNormalizer()

    def compute_metrics(prediction: EvalPrediction) -> dict[str, float]:
        predicted_ids = prediction.predictions
        if isinstance(predicted_ids, tuple):
            predicted_ids = predicted_ids[0]
        label_ids = np.asarray(prediction.label_ids).copy()
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        predictions = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        references = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        normalized_predictions = [normalizer(text).strip() for text in predictions]
        normalized_references = [normalizer(text).strip() for text in references]
        scored = [
            (reference, hypothesis)
            for reference, hypothesis in zip(
                normalized_references, normalized_predictions, strict=True
            )
            if reference
        ]
        return {
            "wer": float(
                jiwer.wer([item[0] for item in scored], [item[1] for item in scored])
            )
        }

    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_directory = output_directory / "checkpoints"
    arguments = Seq2SeqTrainingArguments(
        output_dir=str(checkpoint_directory),
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        max_steps=config.max_steps,
        warmup_steps=max(1, round(config.max_steps * 0.05)),
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        predict_with_generate=True,
        generation_max_length=config.generation_max_length,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        save_strategy="steps",
        save_steps=config.eval_steps,
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        seed=config.seed,
        data_seed=config.seed,
    )
    trainer = DistillationTrainer(
        model=student,
        teacher=teacher,
        args=arguments,
        train_dataset=train_data,
        eval_dataset=evaluation_data,
        data_collator=WhisperDataCollator(feature_extractor, tokenizer),
        processing_class=processor,
        compute_metrics=compute_metrics,
        kl_weight=config.kl_weight,
        hidden_weight=config.hidden_weight,
    )
    checkpoint = get_last_checkpoint(str(checkpoint_directory))
    trainer.train(resume_from_checkpoint=checkpoint)
    (output_directory / "training_metrics.json").write_text(
        json.dumps(trainer.state.log_history, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    merged = cast(Any, student).merge_and_unload()
    merged.save_pretrained(output_directory / "merged-model")
    processor.save_pretrained(output_directory / "merged-model")
    (output_directory / "recovery_config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
