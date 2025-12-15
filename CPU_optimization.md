# Fun-CosyVoice CPU Performance Notes

Use this checklist when trying to squeeze more throughput/RTF on CPU-only deployments. Items are organized from “flip a switch” tweaks to invasive refactors.

## 1. Baseline setup
- Install the CPU-only requirements (`requirements.cpu.txt`) and ensure MKL/OpenMP env vars (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`) match the physical cores.
- Pin PyTorch threads inside your launcher (e.g. at the top of `app.py` or your service wrapper):
  ```python
  import torch, multiprocessing as mp
  torch.set_num_threads(mp.cpu_count())
  torch.set_num_interop_threads(1)
  ```
  This prevents the default oversubscription that often leaves cores idle.

## 2. Mixed precision & compiler optimizations
- In `cosyvoice/cli/model.py`, `CosyVoiceModel.llm_job()` and `CosyVoiceModel.token2wav()` only autocast on CUDA. Switch to:
  ```python
  with torch.amp.autocast(device_type=self.device.type,
                          dtype=torch.bfloat16,
                          enabled=self.fp16 or self.device.type == "cpu"):
      ...
  ```
  Modern CPUs (Sapphire Rapids, Zen4, Apple M-series) run BF16 natively.
- Wrap LLM/Flow forward paths with `torch.compile(fullgraph=True, backend="inductor")` when running PyTorch ≥2.3. Compile once during model load (`CosyVoiceModel.load`). On CPU this fuses matmuls/softmaxes and reduces Python overhead.

## 3. ONNX Runtime for the heavy blocks
- The repo ships `onnx`, `onnxconverter-common`, and `onnxruntime`. Export the Flow decoder via `cosyvoice/bin/export_onnx.py` and load it with ORT:
  ```python
  import onnxruntime as ort
  providers = [("CPUExecutionProvider",
                {"intra_op_num_threads": mp.cpu_count(),
                 "execution_mode": ort.ExecutionMode.ORT_PARALLEL})]
  session = ort.InferenceSession("flow.decoder.onnx",
                                 providers=providers)
  ```
- Replace `self.flow.decoder.estimator` with a thin wrapper around the ORT session. Heavy DiT inference benefits from ORT’s optimized matmul kernels (1.2×–1.8× on Intel/AMD CPUs).
- Same approach applies to HiFi-GAN (`cosyvoice/hifigan/generator.py`) when ultimate latency matters. Export using `torch.onnx.export` (static shapes) and invoke via ORT.

## 4. Quantization
- PyTorch’s AO quantization stack works on CPU without extra deps. For LLM and Flow encoders (implemented in `cosyvoice/llm/llm.py` and `cosyvoice/flow/flow.py`):
  1. Run a calibration script that feeds a few minutes of text/audio pairs through `prepare_fx`.
  2. Convert to per-channel INT8 (`qconfig = get_default_qconfig("fbgemm")`).
  3. Serialize the quantized weights and load them inside `CosyVoiceModel.load`.
- For HiFi-GAN, use dynamic quantization on linear layers or leverage SmoothQuant to reduce activation ranges before INT8 export.

## 5. Better streaming pipeline
- `CosyVoiceModel.tts()` currently polls tokens every 100 ms (`time.sleep(0.1)`) and shares state via Python lists guarded by a lock. On CPU, GIL contention slows the flow decoder.
- Replace the polling with `queue.Queue` + producer/consumer threads or use `asyncio`:
  - LLM thread pushes token chunks into a queue.
  - Flow worker consumes immediately and feeds HiFi-GAN.
  - Remove `time.sleep` and use `queue.get(timeout=...)` to unblock as soon as data is ready.
- If multiple requests run concurrently, dedicate separate worker pools (process-based via `multiprocessing.Process`) for Flow and HiFi-GAN to bypass GIL entirely.

## 6. Prompt preprocessing cache
- `CosyVoiceFrontEnd` (see `cosyvoice/cli/frontend.py`) recomputes tokenizer outputs, log-mel prompts, and speaker embeddings every call. Cache them for frequent speakers:
  ```python
  torch.save({"flow_prompt_speech_token": flow_prompt.clone(),
              "prompt_speech_feat": prompt_feat.clone(),
              "embedding": embedding.clone()},
             "prompt_cache/{spk_id}.pt")
  ```
  Loading these tensors avoids CPU-bound ONNX tokenizer + feature extractor runs, cutting ~50–100 ms per request.

## 7. Profiling & verification
- Use `torch.profiler` (CPU activities only) around the inference loop to confirm hotspots. Example:
  ```python
  with profiler.profile(activities=[profiler.ProfilerActivity.CPU],
                        record_shapes=True) as prof:
      next(model.tts(...))
  print(prof.key_averages().table(sort_by="self_cpu_time_total"))
  ```
- Validate each change with identical prompts and track RTF + peak RSS. Keep a spreadsheet of: baseline FP32 eager, +BF16, +compile, +ORT, +quantization to ensure regressions are easy to spot.

## 8. Deployment tips
- Pin the process to performance cores (Linux `taskset`, Windows Processor Groups) and disable CPU frequency scaling for tighter latency.
- When running inside containers, set `--cpuset-cpus` and `--cpu-shares` high enough; otherwise OpenMP throttles itself.
- Expose tunables (threads, precision, queue depths) via env vars so ops can tweak without rebuilding.

Following this roadmap typically yields 2×–4× better CPU throughput compared with the default FP32 eager path while keeping model quality unchanged.
