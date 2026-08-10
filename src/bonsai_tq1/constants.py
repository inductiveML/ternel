from __future__ import annotations

from pathlib import Path

MODEL_REPO = "prism-ml/Ternary-Bonsai-27B-gguf"
MODEL_REVISION = "abbae723028d71be674e71e1a71201a6f43fab22"
MODEL_FILENAME = "Ternary-Bonsai-27B-Q2_0.gguf"
MODEL_SIZE = 7_165_121_600
MODEL_SHA256 = "868c11714cf8fe47f5ec9eeb2be0ab1a337112886f92ee0ede6b855c4fa31757"

PRISM_REPO = "https://github.com/PrismML-Eng/llama.cpp.git"
PRISM_COMMIT = "9ca265a57f85f2117942490f421f64a226dd9847"

TARGET_GPU_NAME = "NVIDIA RTX 6000 Ada Generation"
TARGET_COMPUTE_CAPABILITY = "8.9"
PRIMARY_TENSOR = "blk.0.ffn_down.weight"
PRIMARY_M = 5_120
PRIMARY_K = 17_408

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
MODEL_DIR = ARTIFACTS / "models"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME
PACKED_DIR = ARTIFACTS / "packed"
PACKED_PATH = PACKED_DIR / f"{MODEL_FILENAME}.tq1g128"
RESULTS_DIR = ARTIFACTS / "results"
REPORTS_DIR = ROOT / "reports"
PRISM_DIR = ARTIFACTS / "src" / "llama.cpp"
BUILD_DIR = ROOT / "build"

