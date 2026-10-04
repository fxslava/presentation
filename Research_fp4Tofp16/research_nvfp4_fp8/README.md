# NVFP4 → FP8 companion experiments

The primary deliverable is [the analyzer specification](../docs/spec_analyzer_backend_nvfp4_fp8.md). These scripts provide software numerical ablations and pre-materialized FP8 GEMM controls, not fused CUDA/CUTLASS kernels or backend implementations.

From the repository root, using Python with NumPy and CUDA-enabled PyTorch:

```powershell
python research_nvfp4_fp8/run_study.py --full
python research_nvfp4_fp8/verify_and_report.py
```

The first command executes N=K=4096 fidelity cases and 15 GPU controls. Omitting `--full` uses 512 for exploratory runs; do not use those results as full-size evidence. The second command checks E4M3/E5M2 mapping, device conversion, bank-padding and contention acceptance cases, then regenerates the measured tables in the specification. The verification expects the full study matrix.

Artifacts:

- `results/study.json`: 62 synthetic fidelity cases and 15 CUDA-event timing controls.
- `results/verification.json`: format and analytical acceptance checks.

NumPy synthetic inputs and stochastic rounding use seed 20261004. Torch-generated control activations in the recorded run were not seeded; timing and those control inputs vary across reruns. M=1 and M=8 are padded to 16. Timing excludes conversion, routing, and materialization; it cannot establish the best fused path. The software fidelity benchmark accumulates in FP32, with TF32 disabled, on FP16-rounded original inputs. See the specification for scale policies and interpretation limits.
