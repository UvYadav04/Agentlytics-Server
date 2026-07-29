import os

from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

MODEL_ID = "BAAI/bge-small-en-v1.5"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    model = ORTModelForFeatureExtraction.from_pretrained(MODEL_ID, export=True)
    model.save_pretrained(OUT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.save_pretrained(OUT_DIR)


if __name__ == "__main__":
    main()
