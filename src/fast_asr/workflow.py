"""Generate an Olive workflow derived from its maintained Whisper base.en CPU recipe."""

from __future__ import annotations

import json
from pathlib import Path


def _librispeech_test_data_config(
    name: str, split: str, max_samples: int
) -> dict[str, object]:
    return {
        "name": name,
        "type": "HuggingfaceContainer",
        "load_dataset_config": {
            "type": "huggingface_dataset",
            "params": {
                "data_name": "hf-audio/esb-datasets-test-only-sorted",
                "subset": "librispeech",
                "split": split,
            },
        },
        "pre_process_data_config": {
            "type": "speech_transcription_pre_process",
            "params": {
                "audio_col": "audio",
                "text_col": "text",
                "sample_rate": 16000,
                "max_samples": max_samples,
            },
        },
    }


def write_full_librispeech_evaluation_workflow(
    workflow_path: Path, model_directory: Path, *, max_samples: int = 0
) -> None:
    """Write the shared full LibriSpeech evaluator for one GenAI model artifact."""
    def metric(name: str, data_config: str, priority: int) -> dict[str, object]:
        return {
            "name": name,
            "type": "accuracy",
            "data_config": data_config,
            "sub_types": [
                {"name": "wer", "higher_is_better": False, "priority": priority},
                {"name": "rtfx", "higher_is_better": True, "priority": priority + 1},
            ],
        }

    workflow = {
        "input_model": {
            "type": "OnnxModel",
            "model_path": str(model_directory),
            "onnx_file_name": "decoder.onnx",
        },
        "data_configs": [
            _librispeech_test_data_config("test_clean_data", "test.clean", max_samples),
            _librispeech_test_data_config("test_other_data", "test.other", max_samples),
        ],
        "evaluators": {
            "common_evaluator": {
                "metrics": [
                    metric("test_clean", "test_clean_data", priority=1),
                    metric("test_other", "test_other_data", priority=3),
                ]
            }
        },
        "systems": {
            "local_system": {
                "type": "LocalSystem",
                "accelerators": [
                    {"device": "cpu", "execution_providers": ["CPUExecutionProvider"]}
                ],
            }
        },
        "engine": {
            "host": "local_system",
            "target": "local_system",
            "evaluator": "common_evaluator",
        },
    }
    workflow_path.parent.mkdir(parents=True, exist_ok=True)
    workflow_path.write_text(json.dumps(workflow, indent=2) + "\n", encoding="utf-8")


def write_whisper_base_en_workflow(
    workflow_path: Path,
    calibration_directory: Path,
    output_directory: Path,
    *,
    smoothquant_alpha: float = 0.5,
    calibration_samples: int = 128,
) -> None:
    """Write a runnable Olive workflow for ``openai/whisper-base.en``.

    The input model, CPU target, and ModelBuilder export pass are intentionally
    retained from Olive's official base.en CPU recipe. The K-quant pass is
    replaced with static activation quantization through INC SmoothQuant.
    """
    calibration_script = Path(__file__).with_name("calibration.py")
    workflow = {
        "input_model": {
            "type": "HfModel",
            "model_path": "openai/whisper-base.en",
            "task": "automatic-speech-recognition",
        },
        "systems": {
            "local_system": {
                "type": "LocalSystem",
                "accelerators": [
                    {"device": "cpu", "execution_providers": ["CPUExecutionProvider"]}
                ],
            }
        },
        "engine": {"target": "local_system"},
        "data_configs": [
            {
                "name": "whisper_onnx_npz_calibration",
                "user_script": str(calibration_script),
                "load_dataset_config": {
                    "type": "npz_calibration_dataset",
                    "params": {
                        "calibration_directory": str(calibration_directory),
                        "max_samples": calibration_samples,
                    },
                },
                "dataloader_config": {"type": "npz_calibration_dataloader"},
            }
        ],
        "passes": {
            "builder": {"type": "ModelBuilder", "precision": "fp32"},
            "static_smoothquant_int8": {
                "type": "IncStaticQuantization",
                "approach": "static",
                "device": "cpu",
                "backend": "default",
                "domain": "nlp",
                "quant_format": "QOperator",
                "calibration_sampling_size": [calibration_samples],
                "data_config": "whisper_onnx_npz_calibration",
                "recipes": {
                    "smooth_quant": True,
                    "smooth_quant_args": {"alpha": smoothquant_alpha},
                },
                "tuning_criterion": {
                    "strategy": "basic",
                    "max_trials": 1,
                    "objective": "performance",
                },
                "save_as_external_data": True,
                "all_tensors_to_one_file": True,
            },
        },
        "output_dir": str(output_directory),
    }
    workflow_path.parent.mkdir(parents=True, exist_ok=True)
    workflow_path.write_text(json.dumps(workflow, indent=2) + "\n", encoding="utf-8")
