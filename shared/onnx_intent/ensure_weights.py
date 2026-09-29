import logging
import os
import shutil
import tempfile

logger = logging.getLogger("shared.onnx_intent")

_HF_REPO_ID = "Xenova/bge-small-en-v1.5"
_HF_FILENAME = "onnx/model.onnx"


def ensure_model_downloaded(target_path: str) -> None:
    if os.path.isfile(target_path):
        return

    logger.warning(
        "onnx_intent: %s not found (gitignored - expected right after a fresh clone/deploy) - "
        "downloading the pre-converted weights from %s (first boot only, ~130MB)",
        target_path, _HF_REPO_ID,
    )
    from huggingface_hub import hf_hub_download

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
