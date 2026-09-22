# Fast ASR

Fast ASR produces and profiles static-activation INT8 Whisper
ONNX components on CPU. It uses Olive's Intel Neural Compressor (INC) static
quantization pass with SmoothQuant, emitting QOperator graphs.

The generated QOperator configuration is **U8 activations and S8 weights**.
This is the correct CPU QOperator combination: signed activation QOperator is
not a useful x86-64 configuration. It is still normally described as INT8
deployment because the integer arithmetic and weights are eight-bit.

## Installation

The project uses a single uv-managed environment for its runtime and development
dependencies.

```bash
uv sync --all-groups
```

Olive is pinned to a specific source commit: that revision contains
`IncStaticQuantization`, which was absent from the latest tested PyPI wheel.

## Starting from Olive's base.en recipe

The workflow generator preserves the official recipe's source model, CPU target,
and `ModelBuilder` FP32 export pass. It replaces that recipe's weight-only
K-quant pass with INC static SmoothQuant.

```bash
uv run \
  fast-asr write-base-en-workflow \
  --workflow artifacts/whisper-base-en-smoothquant.json \
  --calibration-directory artifacts/calibration/encoder \
  --output-directory artifacts/whisper-base-en-smoothquant

uv run \
  olive run --config artifacts/whisper-base-en-smoothquant.json
```

Use this generated workflow only when its calibration NPZ tensors match the
model produced by `ModelBuilder`. For debugging and for component-wise control,
the direct `quantize` command below is preferred.

## Calibration fixture contract

Quantize the encoder, decoder-initial, and decoder-with-past exports separately.
Create one `*.npz` file per representative invocation for each component. Each
NPZ key is the exact ONNX input name; its tensor has the exact exported dtype
and shape, including decoder cache tensors. Use held-out training/development
audio only—never LibriSpeech test-clean or test-other—to create these inputs.

For example, an encoder fixture normally contains one named log-mel input;
decoder fixtures additionally contain token IDs and cache tensors. Verify the
names with `session.get_inputs()` before producing fixtures.

## Component quantization and profiling

```bash
uv run \
  fast-asr quantize \
  --model artifacts/fp32/encoder.onnx \
  --calibration-directory artifacts/calibration/encoder \
  --output-directory artifacts/int8/encoder \
  --alpha 0.5 --calibration-samples 128

uv run \
  fast-asr profile \
  --model artifacts/int8/encoder/encoder.onnx \
  --input artifacts/calibration/encoder/00000.npz \
  --output-directory artifacts/profiles/int8-encoder \
  --threads 4 --warmup-runs 10 --measured-runs 50
```

`profile` defaults to four intra-op threads, one inter-op thread, batch one, and
sequential execution. It writes a p50/p95-compatible JSON summary and the ORT
trace. Set `--threads` explicitly in published runs and use the same fixture,
warmup, and run count for FP32 and INT8.

## Quality gates

An operator profile is not an ASR result. Before accepting a candidate, run the
complete decoder pipeline with matched generation settings and report WER on
both LibriSpeech test-clean and test-other, alongside end-to-end latency. Keep
SmoothQuant alpha, calibration IDs, exported-model hashes, ORT version, and
thread count with the result.

Use the reusable scorer for every runtime. Each runner writes one JSON object per
utterance with required `utterance_id`, `reference`, `prediction`, and
`audio_duration_s` fields. Add `e2e_latency_ms`, `encoder_latency_ms`,
`first_token_latency_ms`, `decode_latency_ms`, `tpot_ms`, `generated_tokens`,
`peak_rss_bytes`, `peak_vram_bytes`, `token_agreement`, `kld`,
`timestamp_mae_ms`, `is_silence`, and `truncated` whenever available.

```bash
uv run \
  fast-asr score \
  --records artifacts/evaluations/candidate-records.jsonl \
  --artifact artifacts/candidate \
  --output artifacts/evaluations/candidate-summary.json
```

The report includes WER/CER and insertion/deletion/substitution rates, exact
utterance-match rate, optional KLD/token-agreement diagnostics, p50/p95 stage
latencies, RTF/RTFx, decoder tokens per second, memory/artifact size, timestamp
coverage and MAE, silence hallucinations, and truncation rate.

The direct benchmark is resumable by default. It validates the encoder/decoder
cache contract, records deployment-file hashes and environment metadata in
`provenance.json`, freezes ordered dataset IDs in `sample_ids.json`, and flushes
each utterance record. Resume is rejected if settings, artifact identity, or
sample IDs changed. New records also capture peak process RSS.

Result ledgers derive a stable 12-character experiment ID from model hashes,
settings, host, package versions, and Git revision. Absolute local paths are
excluded. `--max-samples` is mandatory for direct benchmarks: use a positive
value for screening or explicit zero for a complete split. Experiment plans
also accept zero and execute jobs sequentially to avoid CPU contention.

Candidate promotion uses paired utterances and predeclared gates. It reports WER
delta and speedup with paired bootstrap 95% confidence intervals, truncation
regression, and a machine-readable decision:

```bash
uv run \
  fast-asr compare \
  --baseline-records artifacts/baseline/test-clean/records.jsonl \
  --candidate-records artifacts/candidate/test-clean/records.jsonl \
  --output artifacts/comparisons/candidate-test-clean.json \
  --max-wer-regression 0.01 --min-speedup 1.10 \
  --max-truncation-regression 0.005
```

## Optimization roadmap

Every optimization below is a separate experiment. **Do not promote a latency
result without running the same end-to-end evaluation for that artifact**:
matched decoding, LibriSpeech test-clean and test-other WER, RTFx, and the
four-thread component profile. Evaluate combinations again; individual results
do not prove that their gains compose.

Use two evaluation stages. During implementation, reject clearly bad variants on
a fixed, documented subset of **LibriSpeech dev-clean and dev-other**. Do not
tune against the test sets. Every retained variant and every reported
combination must then run the complete test-clean and test-other protocol below.
The development subset is a preflight check, never a paper-comparable result.

| Priority | Candidate | Expected benefit | Evaluation requirement |
| --- | --- | --- | --- |
| P0 | FP32 export | Quality reference | Full test-clean and test-other WER/CER reference. |
| P0 | Official full INT8 K-quant | Quantized deployment baseline | Full test-clean and test-other; retain per-utterance predictions. |
| P1 (deferred) | Static SmoothQuant / selective precision | Alternative quantization schemes | Do not run in the current study. |
| P2 | Output-side stride-2 audio-token reduction | Reduce decoder cross-attention work | Check WER, deletions, and timestamp behaviour separately. |
| P2 | Adjacent encoder-token merging | Adaptive token reduction | Compare 10/20/40% reduction at matched measured latency. |
| P3 | LiteASR low-rank encoder projections | Reduce encoder dense arithmetic | Recalibrate per model and benchmark ONNX kernels, not FLOPs alone. |
| P3 | Speculative decoding | Reduce serial decoder steps | Verify exact greedy-token agreement with the INT8 verifier. |

Record every result with model revision, provider, package lock, calibration
sample IDs, decoding parameters, WER delta, RTFx, p50/p95 latency, and peak
memory. The current Olive recipe evaluator is a 64-item test-clean smoke test;
use full test-clean and test-other before drawing a quality conclusion.

## Experiment ledger

The smoke protocol is 64 utterances per test split and is diagnostic only. Full
test-clean/test-other is required before reporting a result externally. `—`
means the current Olive evaluator does not emit that metric; the reusable JSONL
scorer will fill it once the runtime runner records per-utterance timings.

All paired efficiency/diagnostic values below are `test-clean / test-other`.
Latency values are milliseconds; size is decimal MB. WER and CER use Whisper's
basic text normalization.

| Config | Encoder frames | WER clean | WER other | CER clean / other | RTFx | TTFT p50 | TTFT p95 | TPS | TPOT p50 | TPOT p95 | Truncated | Size MB |
| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| FP32 | 1500 | 6.62% | 8.78% | 3.32% / 4.74% | 21.59× / 20.29× | 436.95 / 434.58 | 944.16 / 851.09 | 111.3 / 114.5 | 8.85 / 8.57 | 9.98 / 9.55 | 0% / 0% | 400.8 |
| INT8 K-quant | 1500 | 6.47% | 8.73% | 3.28% / 4.76% | 32.47× / 29.78× | 398.07 / 398.50 | 783.76 / 789.95 | 226.3 / 225.7 | 4.30 / 4.36 | 4.94 / 4.82 | 0% / 0% | 213.5 |
| INT8 output stride 2 | 750 | 5.56% | 8.47% | 1.91% / 4.23% | 32.92× / 28.76× | 424.34 / 426.20 | 783.35 / 845.85 | 251.6 / 232.9 | 3.77 / 4.14 | 5.37 / 5.30 | 0% / 0% | 213.5 |
| INT8 output stride 3 | 500 | 5.53% | 26.46% | 2.02% / 18.45% | 32.94× / 27.92× | 416.51 / 420.91 | 850.88 / 858.90 | 257.5 / 258.5 | 3.81 / 3.82 | 4.64 / 4.50 | 0% / 4.69% | 213.5 |
| INT8 output stride 4 | 375 | 26.50% | 16.42% | 20.81% / 11.86% | 31.82× / 31.20× | 404.25 / 404.49 | 839.58 / 794.08 | 278.0 / 282.2 | 3.41 / 3.46 | 4.09 / 3.99 | 4.69% / 1.56% | 213.5 |
| INT8 encoder stride 2 | 750 | 6.08% | 10.04% | 2.22% / 5.19% | 50.85× / 45.20× | 175.73 / 182.02 | 341.27 / 332.22 | 253.1 / 239.8 | 3.86 / 3.99 | 4.77 / 5.24 | 0% / 0% | 211.9 |
| INT8 encoder stride 2 + output stride 2 | 375 | 23.98% | 33.92% | 16.48% / 26.04% | 51.78× / 47.67× | 170.70 / 165.05 | 355.81 / 336.31 | 292.2 / 308.0 | 3.31 / 3.05 | 3.88 / 3.52 | 3.12% / 4.69% | 211.9 |

## Next experiment program

Each track is developed and evaluated independently before combinations are
attempted. `dev-clean`/`dev-other` are the selection sets; test splits are used
only for retained candidates.

| Track | Candidate series | Gate | Status |
| --- | --- | --- | --- |
| E1 | Encoder stride 2 | Full test-clean and test-other | Full matched evaluation complete; clean passes, other fails |
| E2 | LoRA + teacher-distilled encoder stride 2 | Distill at 750 frames; compare with E1 | Full matched evaluation complete; clean gate passes, other gate fails |
| E3 | Adaptive token merging | 10%, 20%, 30%, 40%, 50% reduction | ONNX candidates built; matched validation screening below |
| E4 | Encoder depth reduction | 6→5→4 layers with distillation | Implementation queued |
| E5 | Variable-length encoder input | Match baseline tokens on unpadded clips | Export investigation queued |
| E6 | Operator profiling | Encoder/decoder p50, p95, kernel trace at four threads | Four-thread baseline, pooling, and stride profiles complete |

### Training-free recovery screening

Matched 256-utterance validation splits, four CPU threads. Paired speedups use
the INT8 baseline. All intermediate-pooling candidates pass the one-point WER
and 1.1× speed gates.

| Candidate | WER clean / other | RTFx clean / other | Speedup clean / other | WER delta clean / other |
| --- | --- | --- | --- | --- |
| INT8 baseline | 4.12% / 8.22% | 12.30× / 12.51× | — | — |
| Anti-alias average | 4.94% / 10.28% | 24.17× / 24.34× | 1.97× / 1.95× | +0.82 / +2.07 points |
| Anti-alias binomial-3 | 5.04% / 10.22% | 24.20× / 23.84× | 1.97× / 1.91× | +0.93 / +2.00 points |
| Anti-alias binomial-5 | 5.14% / 10.40% | 24.84× / 24.44× | 2.02× / 1.95× | +1.03 / +2.18 points |
| Pool after layer 2 | 4.36% / 8.33% | 18.54× / 17.20× | 1.51× / 1.37× | +0.25 / +0.11 points |
| Pool after layer 3 | 4.28% / 8.17% | 16.76× / 16.03× | 1.36× / 1.28× | +0.16 / -0.05 points |
| Pool after layer 4 | 4.20% / 8.31% | 14.71× / 14.57× | 1.20× / 1.17× | +0.08 / +0.09 points |

Anti-aliasing retains raw stride-2 speed but does not recover its hard-speech
quality. Pooling after layer 2 is the best training-free speed/quality tradeoff:
its WER deltas are statistically compatible with zero on both validation sets
while preserving 1.37–1.51× end-to-end speedup. It advances to full evaluation.

### Pooling-method comparison

Matched 256-utterance validation splits; each candidate pools after encoder
layer 2 to 750 tokens. Mean pooling remains the deployment candidate.

| Method | WER clean / other | Speed vs mean clean / other |
| --- | --- | --- |
| Mean | 4.36% / 8.33% | 1.00× / 1.00× |
| Max | 4.69% / 8.67% | 0.84× / 0.88× |
| Binomial-3 | 4.51% / 9.26% | 0.84× / 0.98× |
| Binomial-5 | 4.98% / 10.17% | 0.91× / 0.93× |
| Left-weighted | 4.51% / 8.60% | 0.86× / 0.88× |
| Right-weighted | 4.36% / 8.22% | 0.89× / 0.94× |

Right-weighted pooling ties mean on clean and improves other by 0.11 WER
points, but paired bootstrap does not establish a quality difference (95% CI
-0.67 to +0.43 points) and measured end-to-end speed is lower.

### Operator profile

Four CPU threads, synthetic 3,000-frame mel input, three warmups and ten
measured runs. This isolates graph cost; end-to-end LibriSpeech benchmarks
remain the deployment speed reference.

| Model | Encoder median | Decoder step median | Encoder Attention/run | Encoder MatMulNBits/run |
| --- | ---: | ---: | ---: | ---: |
| INT8 | 428.0 ms | 5.8 ms | 229.6 ms | 171.5 ms |
| Pool after layer 2 | 260.3 ms | 5.6 ms | 112.1 ms | 97.8 ms |
| Encoder stride 2 | 166.6 ms | 5.2 ms | 63.7 ms | 78.9 ms |

Attention and quantized matrix multiplication dominate encoder compute.
Layer-2 pooling cuts both substantially; further gains should target these
operators and token count. Decoder step time changes less.

### Content-aware merge candidate

The ONNX merge transform scores change between adjacent layer-2 states, keeps
the largest changes as segment boundaries, and averages states inside each
segment. Output length stays fixed so decoder cache shapes remain compatible.
Candidates were screened on matched 256-utterance validation-clean and
validation-other samples with four CPU threads.

| Layer-2 merge reduction | WER clean / other | RTFx clean / other |
| --- | ---: | ---: |
| 10% | 4.09% / 8.15% | 12.76× / 11.56× |
| 20% | 4.05% / 8.06% | 14.10× / 13.57× |
| 30% | 3.97% / 7.92% | 13.44× / 12.15× |
| 40% | 3.99% / 8.20% | 15.45× / 15.57× |
| 50% | 4.03% / 8.22% | 17.94× / 17.48× |

The merge operator overhead limits end-to-end speed at lower reductions,
despite competitive WER. At 50% reduction, speed approaches layer-2 mean
pooling (18.54× / 17.20×) with slightly lower point-estimate WER. These are
screening results, not full-corpus test results. Against mean pooling, paired
bootstrap gives a clean WER difference of -0.33 points (95% CI -0.66 to
-0.04), but the other-split difference of -0.11 points is inconclusive (95% CI
-0.61 to +0.38). The 50% merge is 3.3% slower on clean and 1.6% faster on
other. Retain it as a quality-oriented candidate for full testing, not a
clear speed upgrade over mean pooling.

```bash
uv run fast-asr create-content-aware-merge-candidate \
  --source-model-directory artifacts/olive-recipe/whisper-base-en_cpu_int8 \
  --output-model-directory artifacts/optimizations/merge-layer2-30 \
  --after-layer 2 --reduction-ratio 0.30
```

Dynamic segment choice uses ONNX `TopK`, `CumSum`, and `ScatterND`; operator
support and end-to-end speed must be measured on each target runtime.

### Dynamic-length feasibility

Current encoder input and positional embeddings are fixed to 3,000 mel frames,
producing 1,500 encoder frames. Dynamic inference requires coordinated symbolic
input dimensions, runtime positional-embedding slicing, dynamic cross-attention
reshapes, symbolic decoder cache dimensions, unpadded feature extraction, and
duration buckets. Quality should remain unchanged, but ONNX Runtime fused
Attention and ORT GenAI cache compatibility require runtime validation first.

### Compression-location smoke results

64 utterances per split, four threads. Values are `test-clean / test-other`.

| Candidate | Final frames | WER | RTFx | Truncated | Gate |
| --- | ---: | --- | --- | --- | --- |
| Hidden pool 2 | 750 | 5.43% / 8.45% | 34.94× / 32.41× | 0% / 0% | Pass |
| Hidden pool 3 | 500 | 5.51% / 8.93% | 36.55× / 33.67× | 0% / 0% | Pass |
| Hidden pool 4 | 375 | 16.77% / 20.19% | 36.00× / 33.38× | 1.56% / 1.56% | Reject |
| Encoder stride 3 | 500 | 14.15% / 53.96% | 60.74× / 44.19× | 1.56% / 9.38% | Recovery target |
| Encoder stride 4 | 375 | 29.39% / 74.93% | 64.28× / 48.78× | 3.12% / 9.38% | Reject |
| Encoder stride 2 + hidden pool 2 | 375 | 8.05% / 39.69% | 58.55× / 46.97× | 0% / 4.69% | Recovery target |
| Encoder stride 2 + hidden pool 3 | 250 | 63.03% / 70.40% | 46.29× / 42.82× | 9.38% / 9.38% | Reject |
| Encoder stride 3 + hidden pool 2 | 250 | 26.99% / 65.85% | 69.53× / 51.38× | 1.56% / 7.81% | Reject |
| Encoder stride 3 + hidden pool 4 | 125 | 317.26% / 182.49% | 27.00× / 42.31× | 48.44% / 21.88% | Reject |
| Encoder stride 4 + hidden pool 3 | 125 | 209.73% / 232.28% | 37.47× / 33.85× | 29.69% / 31.25% | Reject |

Full encoder-stride-2 result: test-clean WER 5.30%, CER 2.17%, RTFx
28.31×; test-other WER 12.94%, CER 6.50%, RTFx 26.44×. A full INT8
baseline run is required for matched full-split speed and WER deltas.

Matched full INT8 evaluation is complete. Test-clean: WER 4.74%, CER 1.98%,
RTFx 14.46×, TTFT p50 397.76 ms, and decoder throughput 221.15 tokens/s.
Test-other: WER 10.87%, CER 5.16%, RTFx 12.93×, TTFT p50 397.58 ms, and
decoder throughput 214.21 tokens/s. Encoder stride 2 is 1.96× / 2.04× faster
with +0.57 / +2.07 percentage-point WER changes on clean / other. It therefore
needs recovery before promotion under the current one-point WER gate.

Paired 1,000-sample bootstrap confirms the split-dependent result. Test-clean:
speedup 1.957× (95% CI 1.946–1.969×), WER delta +0.57 points (CI +0.32 to
+0.77), gate pass. Test-other: speedup 2.045× (CI 2.030–2.056×), WER delta
+2.07 points (CI +1.79 to +2.31), gate fail. The reports are stored under
`artifacts/comparisons/`.

### Full benchmark results

All runs below cover the complete LibriSpeech test split with identical decoding
settings and four CPU threads. Latencies are milliseconds; paired efficiency
values are `test-clean / test-other`.

| Config | WER clean / other | CER clean / other | RTFx clean / other | TTFT p50 clean / other | TPS clean / other | TPOT p50 clean / other | Truncated clean / other | Size MB |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| FP32 baseline | 4.72% / 10.86% | 1.97% / 5.16% | 11.13× / 10.51× | 439.80 / 427.44 | 110.7 / 114.4 | 8.85 / 8.37 | 0% / 0% | 400.8 |
| INT8 baseline | 4.74% / 10.87% | 1.98% / 5.16% | 14.46× / 12.93× | 397.76 / 397.58 | 221.1 / 214.2 | 4.39 / 4.46 | 0% / 0% | 213.5 |
| INT8 pool after layer 2 | 4.86% / 11.57% | 2.04% / 5.60% | 23.10× / 19.81× | 233.98 / 243.34 | 285.5 / 261.5 | 3.36 / 3.74 | 0% / 0.03% | 213.4 |
| INT8 encoder stride 2 | 5.30% / 12.94% | 2.17% / 6.50% | 28.31× / 26.44× | 168.42 / 166.64 | 268.3 / 277.2 | 3.60 / 3.65 | 0% / 0.03% | 211.9 |
| INT8 recovered encoder stride 2 | 5.13% / 12.52% | 2.09% / 6.76% | 27.43× / 27.12× | 174.61 / 161.64 | 261.4 / 285.9 | 3.91 / 3.38 | 0% / 0.03% | 211.8 |

INT8 is 1.30× / 1.23× faster than FP32 with statistically negligible WER
changes (+0.01 / +0.01 points). Pooling after encoder layer 2 is 1.60× / 1.53×
faster than INT8, with +0.12 points WER on test-clean (95% CI +0.04 to +0.22)
and +0.70 points on test-other (CI +0.29 to +1.37). It passes the predeclared
point-estimate gate on both splits, although the test-other CI crosses the
one-point threshold and should be treated as borderline.

Against the INT8 baseline, the recovered model is 1.896× faster on test-clean
(95% CI 1.888–1.905×) with a +0.39-point WER change (CI +0.15 to +0.59), so it
passes the one-point WER gate. On test-other it is 2.098× faster (CI
2.081–2.112×) with a +1.65-point WER change (CI +1.40 to +1.91), so it fails.

Recovery significantly improves the unadapted stride-2 model: WER falls by
0.18 points on test-clean (CI 0.07–0.28) and 0.41 points on test-other (CI
0.22–0.61). The gain is nevertheless insufficient on harder speech. The next
recovery run should therefore target validation-other explicitly rather than
increasing stride compression.

### Recovery training

`train-recovery` applies encoder compression, trains LoRA adapters with transcript,
teacher-logit, and encoder-hidden-state losses, merges the adapters, and saves a
self-contained Transformers checkpoint plus `recovery_config.json`. A one-step
end-to-end smoke run passes. The first 250-step stride-2 pilot uses 2,000 streamed
LibriSpeech train-clean-100 examples and starts after the active full INT8 benchmark.

Training audio is streamed and transformed per batch; it is never materialized
as decoded audio in memory. Every 50 optimizer steps, generated-transcript WER
is measured on three fixed, disjoint 64-utterance sets: a train-clean-100
holdout, validation-clean, and validation-other. Trainer checkpoints contain
intermediate metric history; the completed run also writes
`training_metrics.json`. Interrupted runs resume from the latest checkpoint.

```bash
uv run \
  fast-asr train-recovery \
  --output-directory artifacts/recovery-training/encoder-stride-2-pilot \
  --encoder-stride-factor 2 --max-steps 250 --max-train-samples 2000 \
  --gradient-accumulation-steps 8
```

The first retained recovery checkpoint has been exported, statically quantized,
and evaluated with the same full benchmark protocol. It passes the clean-speech
gate but remains a recovery candidate because it fails the test-other WER gate.

Validate and export a merged recovery checkpoint using the same FP32
ModelBuilder and full-INT8 K-quant pass as the deployment baseline:

```bash
uv run \
  fast-asr write-recovery-export-workflow \
  --checkpoint-directory artifacts/recovery-training/encoder-stride-2-pilot \
  --workflow artifacts/recovery-training/encoder-stride-2-pilot/export.json \
  --output-directory artifacts/recovery-training/encoder-stride-2-pilot/int8
```

Development screening is represented by a resumable model-by-split plan. The
defaults are 256 samples each from LibriSpeech validation-clean and
validation-other at four threads:

```bash
uv run \
  fast-asr write-screening-plan \
  --model-directories artifacts/candidate-a artifacts/candidate-b \
  --output-root artifacts/dev-screening --plan artifacts/dev-screening/plan.json

uv run \
  fast-asr run-screening-plan \
  --plan artifacts/dev-screening/plan.json
```

`write-result-table` converts any collection of versioned `summary.json` files
to stable CSV and Markdown. The current generated ledger lives under
`artifacts/results/benchmark-ledger.{csv,md}`.

`refresh-result-table --results-root artifacts` discovers every summary and
regenerates both formats without maintaining a manual path list.

Generate a fresh full evaluator for any artifact instead of copying a model-specific
JSON file:

```bash
uv run \
  fast-asr write-full-evaluation-workflow \
  --model-directory artifacts/candidate \
  --workflow artifacts/evaluations/candidate-full-eval.json
```

## Development

```bash
uv run ruff check .
uv run pyrefly check
uv run pytest
```
