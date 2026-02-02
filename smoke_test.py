import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

NLLB_BASE = "facebook/nllb-200-distilled-1.3B"
ARAT5_BASE = "UBC-NLP/AraT5v2-base-1024"

NLLB_ADAPTER_PATH = "models/nllb_adapter"
ARAT5_ADAPTER_PATH = "models/arat5v2_adapter"

def check_adapter(path):
    need = ["adapter_config.json", "adapter_model.safetensors"]
    missing = [f for f in need if not os.path.exists(os.path.join(path, f))]
    if missing:
        raise FileNotFoundError(f"Missing in {path}: {missing}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE:", device)

    # 1) Check files exist
    check_adapter(NLLB_ADAPTER_PATH)
    check_adapter(ARAT5_ADAPTER_PATH)
    print("✅ Adapter files exist")

    # 2) Load NLLB + adapter
    tok_n = AutoTokenizer.from_pretrained(NLLB_BASE)
    base_n = AutoModelForSeq2SeqLM.from_pretrained(NLLB_BASE)
    model_n = PeftModel.from_pretrained(base_n, NLLB_ADAPTER_PATH)
    model_n = model_n.to(device).eval()
    print("✅ NLLB adapter loads")

    # 3) Load AraT5v2 + adapter
    tok_a = AutoTokenizer.from_pretrained(ARAT5_BASE)
    base_a = AutoModelForSeq2SeqLM.from_pretrained(ARAT5_BASE)
    model_a = PeftModel.from_pretrained(base_a, ARAT5_ADAPTER_PATH)
    model_a = model_a.to(device).eval()
    print("✅ AraT5 adapter loads")

    print("\nALL GOOD ✅")

if __name__ == "__main__":
    main()
