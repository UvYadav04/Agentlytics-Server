"""Runtime safety net for shared/onnx_intent/weights/model.onnx.

That file is gitignored (too large to commit) - see download_model.py for the maintainer-facing
script that regenerates the whole weights/ directory from BAAI/bge-small-en-v1.5 via
optimum+transformers+torch (build-time only, not in shared/requirements.txt, meant to be run by
hand). This module is the lightweight counterpart that actually runs automatically at process
startup: if model.onnx is missing - the normal state right after a fresh git clone/deploy, e.g. a
new EC2 box - fetch the already-converted file from Xenova/bge-small-en-v1.5 on the Hugging Face
Hub (same architecture/weights as BAAI/bge-small-en-v1.5, pre-exported to ONNX by Xenova for
transformers.js) instead of doing the conversion ourselves. That means this only needs
`huggingface_hub` (small, no torch) rather than the full optimum/transformers/torch stack. The
tokenizer/config files next to model.onnx are NOT gitignored and stay committed as-is, so only
the model file itself needs fetching here.

Called from model.py's _load(), so it runs once per process the first time embed() is used, and
is a no-op on every call after that (file already exists). Safe under multiple worker processes
racing to fetch this on the same first boot (e.g. several uvicorn/gunicorn workers) - each writes
its own temp file, then os.replace() atomically swaps it into place, so no process can ever see a
half-written model.onnx.
"""
import logging
import os
import shutil
import tempfile

logger = logging.getLogger("shared.onnx_intent")

_HF_REPO_ID = "Xenova/bge-small-en-v1.5"
_HF_FILENAME = "onnx/model.onnx"


def ensure_model_downloaded(target_path: str) -> None:
    """Never raises for a missing/failed download on its own - lets the caller's existing
    try/except (model.py's _load(), called from intent_router.init()) treat that the same as any
    other reason the ONNX tier couldn't come up: log it and fall back to the LLM classifier
    rather than fail startup."""
    if os.path.isfile(target_path):
        return

    logger.warning(
        "onnx_intent: %s not found (gitignored - expected right after a fresh clone/deploy) - "
        "downloading the pre-converted weights from %s (first boot only, ~130MB)",
        target_path, _HF_REPO_ID,
    )
    from huggingface_hub import hf_hub_download  # imported lazily - only ever needed on this path

    cached_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=_HF_FILENAME)

    weights_dir = os.path.dirname(target_path)
    os.makedirs(weights_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=weights_dir, suffix=".onnx.tmp")
    os.close(fd)
    try:
        shutil.copy2(cached_path, tmp_path)
        os.replace(tmp_path, target_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    logger.info("onnx_intent: model downloaded and saved to %s", target_path)
