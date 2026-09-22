"""Command-line interface for static quantization and controlled CPU profiling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fast_asr.benchmark import benchmark_librispeech
from fast_asr.evaluation import write_evaluation_report
from fast_asr.output_stride import (
    create_antialiased_encoder_stride_candidate,
    create_encoder_stride_candidate,
    create_hidden_state_pool_candidate,
    create_intermediate_pool_candidate,
    create_output_stride_candidate,
)
from fast_asr.profiling import profile_cpu_model
from fast_asr.result_table import refresh_result_tables, write_result_tables
from fast_asr.screening import run_screening_plan, write_screening_plan
from fast_asr.workflow import (
    write_full_librispeech_evaluation_workflow,
    write_whisper_base_en_workflow,
)


def build_parser() -> argparse.ArgumentParser:
    """Construct the package CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    quantize = commands.add_parser(
        "quantize", help="Run Olive INC static SmoothQuant on one ONNX component."
    )
    quantize.add_argument("--model", type=Path, required=True)
    quantize.add_argument("--calibration-directory", type=Path, required=True)
    quantize.add_argument("--output-directory", type=Path, required=True)
    quantize.add_argument("--alpha", type=float, default=0.5)
    quantize.add_argument("--calibration-samples", type=int, default=128)

    profile = commands.add_parser("profile", help="Profile one ONNX component on CPU.")
    profile.add_argument("--model", type=Path, required=True)
    profile.add_argument("--input", type=Path, required=True)
    profile.add_argument("--output-directory", type=Path, required=True)
    profile.add_argument("--threads", type=int, default=4)
    profile.add_argument("--warmup-runs", type=int, default=10)
    profile.add_argument("--measured-runs", type=int, default=50)

    score = commands.add_parser(
        "score", help="Aggregate a runtime-neutral ASR evaluation JSONL file."
    )
    score.add_argument("--records", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--artifact", type=Path)

    workflow = commands.add_parser(
        "write-base-en-workflow",
        help="Write an Olive workflow derived from Olive's Whisper base.en CPU recipe.",
    )
    workflow.add_argument("--workflow", type=Path, required=True)
    workflow.add_argument("--calibration-directory", type=Path, required=True)
    workflow.add_argument("--output-directory", type=Path, required=True)
    workflow.add_argument("--alpha", type=float, default=0.5)
    workflow.add_argument("--calibration-samples", type=int, default=128)

    full_evaluation = commands.add_parser(
        "write-full-evaluation-workflow",
        help="Write the reusable full LibriSpeech test-clean/test-other Olive evaluator.",
    )
    full_evaluation.add_argument("--workflow", type=Path, required=True)
    full_evaluation.add_argument("--model-directory", type=Path, required=True)
    full_evaluation.add_argument(
        "--max-samples", type=int, default=0, help="Per-split limit; zero evaluates the full split."
    )

    output_stride = commands.add_parser(
        "create-output-stride-candidate",
        help="Create a Whisper artifact with strided encoder cross-attention caches.",
    )
    output_stride.add_argument("--source-model-directory", type=Path, required=True)
    output_stride.add_argument("--output-model-directory", type=Path, required=True)
    output_stride.add_argument("--stride", type=int, default=2)
    encoder_stride = commands.add_parser(
        "create-encoder-stride-candidate",
        help="Create a Whisper artifact with a larger encoder convolution stride.",
    )
    encoder_stride.add_argument("--source-model-directory", type=Path, required=True)
    encoder_stride.add_argument("--output-model-directory", type=Path, required=True)
    encoder_stride.add_argument("--factor", type=int, default=2)
    hidden_pool = commands.add_parser(
        "create-hidden-state-pool-candidate",
        help="Mean-pool encoder states before cross-attention K/V projection.",
    )
    hidden_pool.add_argument("--source-model-directory", type=Path, required=True)
    hidden_pool.add_argument("--output-model-directory", type=Path, required=True)
    hidden_pool.add_argument("--factor", type=int, default=2)
    antialiased_stride = commands.add_parser(
        "create-antialiased-encoder-stride-candidate",
        help="Create stride-2 encoder using fixed low-pass filtering before subsampling.",
    )
    antialiased_stride.add_argument("--source-model-directory", type=Path, required=True)
    antialiased_stride.add_argument("--output-model-directory", type=Path, required=True)
    antialiased_stride.add_argument(
        "--kernel", choices=["average", "binomial3", "binomial5"], default="binomial3"
    )
    intermediate_pool = commands.add_parser(
        "create-intermediate-pool-candidate",
        help="Mean-pool tokens after a selected Whisper encoder layer.",
    )
    intermediate_pool.add_argument("--source-model-directory", type=Path, required=True)
    intermediate_pool.add_argument("--output-model-directory", type=Path, required=True)
    intermediate_pool.add_argument("--after-layer", type=int, choices=range(1, 6), required=True)
    intermediate_pool.add_argument("--factor", type=int, default=2)
    benchmark = commands.add_parser(
        "benchmark", help="Run self-contained direct-ORT ASR evaluation."
    )
    benchmark.add_argument("--model-directory", type=Path, required=True)
    benchmark.add_argument("--output-directory", type=Path, required=True)
    benchmark.add_argument("--split", default="test.clean")
    benchmark.add_argument(
        "--max-samples",
        type=int,
        required=True,
        help="Per-split limit; zero evaluates the full split.",
    )
    benchmark.add_argument("--threads", type=int, default=4)
    benchmark.add_argument("--max-tokens", type=int, default=448)
    recovery = commands.add_parser(
        "train-recovery", help="Fine-tune and merge LoRA recovery for a compressed encoder."
    )
    recovery.add_argument("--output-directory", type=Path, required=True)
    recovery.add_argument("--encoder-stride-factor", type=int, default=2)
    recovery.add_argument("--hidden-pool-factor", type=int, default=1)
    recovery.add_argument("--lora-rank", type=int, default=8)
    recovery.add_argument("--learning-rate", type=float, default=1e-4)
    recovery.add_argument("--max-steps", type=int, default=1000)
    recovery.add_argument("--kl-weight", type=float, default=0.5)
    recovery.add_argument("--hidden-weight", type=float, default=0.25)
    recovery.add_argument("--max-train-samples", type=int, default=0)
    recovery.add_argument("--preprocessing-workers", type=int, default=4)
    recovery.add_argument("--batch-size", type=int, default=1)
    recovery.add_argument("--gradient-accumulation-steps", type=int, default=8)
    recovery.add_argument("--seed", type=int, default=42)
    recovery.add_argument("--eval-steps", type=int, default=50)
    recovery.add_argument("--eval-samples-per-split", type=int, default=64)
    recovery.add_argument("--generation-max-length", type=int, default=128)
    recovery_export = commands.add_parser(
        "write-recovery-export-workflow",
        help="Validate a merged recovery checkpoint and write FP32 plus INT8 export passes.",
    )
    recovery_export.add_argument("--checkpoint-directory", type=Path, required=True)
    recovery_export.add_argument("--workflow", type=Path, required=True)
    recovery_export.add_argument("--output-directory", type=Path, required=True)
    finalize_export = commands.add_parser(
        "finalize-recovery-export",
        help="Restore non-serializable architecture changes in an exported recovery model.",
    )
    finalize_export.add_argument("--checkpoint-directory", type=Path, required=True)
    finalize_export.add_argument("--model-directory", type=Path, required=True)
    result_table = commands.add_parser(
        "write-result-table", help="Aggregate evaluation summaries into CSV and Markdown."
    )
    result_table.add_argument("--summaries", type=Path, nargs="+", required=True)
    result_table.add_argument("--csv", type=Path, required=True)
    result_table.add_argument("--markdown", type=Path, required=True)
    refresh_table = commands.add_parser(
        "refresh-result-table", help="Discover summaries and regenerate CSV and Markdown ledgers."
    )
    refresh_table.add_argument("--results-root", type=Path, required=True)
    refresh_table.add_argument("--csv", type=Path, required=True)
    refresh_table.add_argument("--markdown", type=Path, required=True)
    comparison = commands.add_parser(
        "compare", help="Run paired bootstrap analysis and candidate promotion gates."
    )
    comparison.add_argument("--baseline-records", type=Path, required=True)
    comparison.add_argument("--candidate-records", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True)
    comparison.add_argument("--max-wer-regression", type=float, default=0.01)
    comparison.add_argument("--min-speedup", type=float, default=1.1)
    comparison.add_argument("--max-truncation-regression", type=float, default=0.005)
    comparison.add_argument("--bootstrap-samples", type=int, default=1000)
    comparison.add_argument("--seed", type=int, default=42)
    screening_plan = commands.add_parser(
        "write-screening-plan", help="Write a resumable candidate-by-split benchmark matrix."
    )
    screening_plan.add_argument("--model-directories", type=Path, nargs="+", required=True)
    screening_plan.add_argument("--output-root", type=Path, required=True)
    screening_plan.add_argument("--plan", type=Path, required=True)
    screening_plan.add_argument(
        "--splits", nargs="+", default=["validation.clean", "validation.other"]
    )
    screening_plan.add_argument("--max-samples", type=int, default=256)
    screening_plan.add_argument("--threads", type=int, default=4)
    run_screening = commands.add_parser(
        "run-screening-plan", help="Run or resume a previously written screening matrix."
    )
    run_screening.add_argument("--plan", type=Path, required=True)
    return parser


def main() -> None:
    """Execute the selected subcommand."""
    arguments = build_parser().parse_args()
    if arguments.command == "quantize":
        from fast_asr.quantize import quantize_with_smoothquant

        output_model = quantize_with_smoothquant(
            arguments.model,
            arguments.calibration_directory,
            arguments.output_directory,
            smoothquant_alpha=arguments.alpha,
            calibration_samples=arguments.calibration_samples,
        )
        print(output_model)
        return

    if arguments.command == "write-base-en-workflow":
        write_whisper_base_en_workflow(
            arguments.workflow,
            arguments.calibration_directory,
            arguments.output_directory,
            smoothquant_alpha=arguments.alpha,
            calibration_samples=arguments.calibration_samples,
        )
        print(arguments.workflow)
        return

    if arguments.command == "score":
        report = write_evaluation_report(arguments.records, arguments.output, arguments.artifact)
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    if arguments.command == "write-full-evaluation-workflow":
        write_full_librispeech_evaluation_workflow(
            arguments.workflow, arguments.model_directory, max_samples=arguments.max_samples
        )
        print(arguments.workflow)
        return

    if arguments.command == "create-output-stride-candidate":
        create_output_stride_candidate(
            arguments.source_model_directory,
            arguments.output_model_directory,
            stride=arguments.stride,
        )
        print(arguments.output_model_directory)
        return

    if arguments.command == "create-encoder-stride-candidate":
        create_encoder_stride_candidate(
            arguments.source_model_directory,
            arguments.output_model_directory,
            factor=arguments.factor,
        )
        print(arguments.output_model_directory)
        return

    if arguments.command == "create-hidden-state-pool-candidate":
        create_hidden_state_pool_candidate(
            arguments.source_model_directory,
            arguments.output_model_directory,
            factor=arguments.factor,
        )
        print(arguments.output_model_directory)
        return

    if arguments.command == "create-antialiased-encoder-stride-candidate":
        create_antialiased_encoder_stride_candidate(
            arguments.source_model_directory,
            arguments.output_model_directory,
            kernel=arguments.kernel,
        )
        print(arguments.output_model_directory)
        return

    if arguments.command == "create-intermediate-pool-candidate":
        create_intermediate_pool_candidate(
            arguments.source_model_directory,
            arguments.output_model_directory,
            after_layer=arguments.after_layer,
            factor=arguments.factor,
        )
        print(arguments.output_model_directory)
        return

    if arguments.command == "benchmark":
        report = benchmark_librispeech(
            arguments.model_directory,
            arguments.output_directory,
            split=arguments.split,
            max_samples=arguments.max_samples,
            threads=arguments.threads,
            max_tokens=arguments.max_tokens,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    if arguments.command == "train-recovery":
        from fast_asr.recovery_training import RecoveryConfig, train_recovery

        config = RecoveryConfig(
            encoder_stride_factor=arguments.encoder_stride_factor,
            hidden_pool_factor=arguments.hidden_pool_factor,
            lora_rank=arguments.lora_rank,
            learning_rate=arguments.learning_rate,
            max_steps=arguments.max_steps,
            kl_weight=arguments.kl_weight,
            hidden_weight=arguments.hidden_weight,
            max_train_samples=arguments.max_train_samples,
            preprocessing_workers=arguments.preprocessing_workers,
            batch_size=arguments.batch_size,
            gradient_accumulation_steps=arguments.gradient_accumulation_steps,
            seed=arguments.seed,
            eval_steps=arguments.eval_steps,
            eval_samples_per_split=arguments.eval_samples_per_split,
            generation_max_length=arguments.generation_max_length,
        )
        train_recovery(arguments.output_directory, config)
        print(arguments.output_directory)
        return

    if arguments.command == "write-recovery-export-workflow":
        from fast_asr.recovery_export import write_recovery_export_workflow

        write_recovery_export_workflow(
            arguments.checkpoint_directory, arguments.workflow, arguments.output_directory
        )
        print(arguments.workflow)
        return

    if arguments.command == "finalize-recovery-export":
        from fast_asr.recovery_export import finalize_recovery_export

        finalize_recovery_export(arguments.checkpoint_directory, arguments.model_directory)
        print(arguments.model_directory)
        return

    if arguments.command == "write-result-table":
        write_result_tables(arguments.summaries, arguments.csv, arguments.markdown)
        print(arguments.markdown)
        return

    if arguments.command == "refresh-result-table":
        refresh_result_tables(arguments.results_root, arguments.csv, arguments.markdown)
        print(arguments.markdown)
        return

    if arguments.command == "compare":
        from fast_asr.comparison import PromotionGate, write_comparison

        report = write_comparison(
            arguments.baseline_records,
            arguments.candidate_records,
            arguments.output,
            gate=PromotionGate(
                max_wer_regression=arguments.max_wer_regression,
                min_speedup=arguments.min_speedup,
                max_truncation_regression=arguments.max_truncation_regression,
            ),
            bootstrap_samples=arguments.bootstrap_samples,
            seed=arguments.seed,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    if arguments.command == "write-screening-plan":
        write_screening_plan(
            arguments.model_directories,
            arguments.output_root,
            arguments.plan,
            splits=arguments.splits,
            max_samples=arguments.max_samples,
            threads=arguments.threads,
        )
        print(arguments.plan)
        return

    if arguments.command == "run-screening-plan":
        reports = run_screening_plan(arguments.plan)
        print(json.dumps(reports, indent=2, sort_keys=True))
        return

    result = profile_cpu_model(
        arguments.model,
        arguments.input,
        arguments.output_directory,
        threads=arguments.threads,
        warmup_runs=arguments.warmup_runs,
        measured_runs=arguments.measured_runs,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
