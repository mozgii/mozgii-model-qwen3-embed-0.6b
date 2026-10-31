# mozgii-model-qwen3-embed-0.6b

One of MOZGII's local models (ARCHITECTURE.md §35), run by the core's `llm` task with ONNX Runtime on the CPU.

- Source: `Qwen/Qwen3-Embedding-0.6B 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` (Apache-2.0, see LICENSE).
- Conversion: ONNX opset 18; MatMul weights 8-bit block-wise (MatMulNBits, block 32, symmetric); vocabulary table float16; parity vs PyTorch fp32: cosine min 0.99878, mean 0.99924 on 15 ru/en texts
- `model.json`: what MOZGII needs to run it (input, padding, prefixes or template), every file's SHA-256 and size,
  and probes: values the model gave when it was built, which every computer must reproduce before the model counts as ready.
- `dist/` (not in git): `model.onnx`, `tokenizer.json`. A release attaches `model.json`, `model.onnx` and
  `tokenizer.json` as assets of tag `v<version>`; MOZGII's catalog (`core/src/mozgmind_core/internal/llm/catalog.py`)
  pins each one's size and SHA-256, so publish exactly these files.
- `scripts/`: the export (`export.py`), quantization (`quant.py`, `table.py`) and parity check (`parity.py`) used to
  make it, run in a throwaway environment with torch, transformers, onnx and onnxruntime.
