import os

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from shared.config import get_settings

_DEFAULT_WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
_MAX_LENGTH = 64

_session: ort.InferenceSession | None = None
_tokenizer: Tokenizer | None = None


def _weights_dir() -> str:
    return get_settings().get("ONNX_INTENT_WEIGHTS_DIR", "") or _DEFAULT_WEIGHTS_DIR


def _model_path() -> str:
    configured = get_settings().get("ONNX_INTENT_MODEL_PATH", "") or ""
    return configured or os.path.join(_weights_dir(), "model.onnx")


def _tokenizer_path() -> str:
    configured = get_settings().get("ONNX_INTENT_TOKENIZER_PATH", "") or ""
    return configured or os.path.join(_weights_dir(), "tokenizer.json")


def _load() -> None:
    global _session, _tokenizer
    if _session is not None and _tokenizer is not None:
        return

    tokenizer = Tokenizer.from_file(_tokenizer_path())
    pad_id = tokenizer.token_to_id("[PAD]")
    tokenizer.enable_padding(pad_id=pad_id if pad_id is not None else 0, pad_token="[PAD]")
    tokenizer.enable_truncation(max_length=_MAX_LENGTH)

    session = ort.InferenceSession(_model_path(), providers=["CPUExecutionProvider"])

    _tokenizer = tokenizer
    _session = session


def embed(texts: list[str]) -> np.ndarray:
    _load()

    encodings = _tokenizer.encode_batch(texts)
    input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
    token_type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64)

    available = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }
    feed = {i.name: available[i.name] for i in _session.get_inputs() if i.name in available}
    outputs = _session.run(None, feed)

    embeddings = outputs[0][:, 0]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (embeddings / norms).astype(np.float32)
