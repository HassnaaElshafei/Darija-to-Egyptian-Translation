import streamlit as st
import torch
import types

# Streamlit Cloud sometimes ships a torch build without torch.distributed.tensor (DTensor).
# PEFT expects it, so we add a harmless stub to prevent AttributeError.
if hasattr(torch, "distributed") and not hasattr(torch.distributed, "tensor"):
    class _FakeDTensor: 
        pass
    torch.distributed.tensor = types.SimpleNamespace(DTensor=_FakeDTensor)
    
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# ----------------------------
# CONFIG
# ----------------------------
NLLB_BASE = "facebook/nllb-200-distilled-1.3B"
NLLB_SRC_LANG = "ary_Arab"  # Moroccan Darija (Arabic script)
NLLB_TGT_LANG = "arz_Arab"  # Egyptian Arabic (Arabic script)

DARJA2MSA_MODEL = "Saidtaoussi/AraT5_Darija_to_MSA"
ARAT5_BASE = "UBC-NLP/AraT5v2-base-1024"

# Local adapter paths
NLLB_ADAPTER_PATH = "HassnaaElshafei/nllb_adapter"
ARAT5_ADAPTER_PATH = "HassnaaElshafei/arat5v2_adapter"

# ----------------------------
# DEVICE
# ----------------------------
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = get_device()


# ----------------------------
# LOADERS (cached)
# ----------------------------
@st.cache_resource
def load_nllb_pipeline():
    """
    Loads: NLLB base + your LoRA adapter
    """
    tokenizer = AutoTokenizer.from_pretrained(NLLB_BASE)

    # If you want lighter GPU memory, you can try:
    # model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_BASE, device_map="auto", torch_dtype=torch.float16)
    # But simplest/most compatible:
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_BASE)

    model = PeftModel.from_pretrained(model, NLLB_ADAPTER_PATH)

    if DEVICE == "cuda":
        model = model.to("cuda")
    model.eval()

    # optional: merge for faster inference
    try:
        model = model.merge_and_unload()
        model.eval()
    except Exception:
        pass

    return tokenizer, model


@st.cache_resource
def load_arat5_pipeline():
    """
    Loads: Darija->MSA public model + AraT5v2 base + your LoRA adapter
    """
    # Darija -> MSA
    tok_d = AutoTokenizer.from_pretrained(DARJA2MSA_MODEL)
    model_d = AutoModelForSeq2SeqLM.from_pretrained(DARJA2MSA_MODEL)

    # MSA -> Egyptian (your adapter on AraT5v2 base)
    tok_e = AutoTokenizer.from_pretrained(ARAT5_BASE)
    base_e = AutoModelForSeq2SeqLM.from_pretrained(ARAT5_BASE)
    model_e = PeftModel.from_pretrained(base_e, ARAT5_ADAPTER_PATH)

    if DEVICE == "cuda":
        model_d = model_d.to("cuda")
        model_e = model_e.to("cuda")

    model_d.eval()
    model_e.eval()

    # optional merge for stability/speed
    try:
        model_e = model_e.merge_and_unload()
        model_e.eval()
    except Exception:
        pass

    return tok_d, model_d, tok_e, model_e


# ----------------------------
# INFERENCE FUNCTIONS
# ----------------------------
def translate_with_nllb(text: str, tokenizer, model,
                        max_length=128, num_beams=5):
    tokenizer.src_lang = NLLB_SRC_LANG

    inputs = tokenizer(text, return_tensors="pt")
    if DEVICE == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Force Egyptian output
    forced_bos = tokenizer.convert_tokens_to_ids(NLLB_TGT_LANG)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
        )

    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def translate_with_arat5_two_step(text: str, tok_d, model_d, tok_e, model_e,
                                  max_length=128, num_beams=5):
    # Step 1: Darija -> MSA
    in1 = tok_d(text, return_tensors="pt", padding=True)
    if DEVICE == "cuda":
        in1 = {k: v.to("cuda") for k, v in in1.items()}

    with torch.no_grad():
        out1 = model_d.generate(**in1, max_length=max_length)

    msa = tok_d.decode(out1[0], skip_special_tokens=True).strip()

    # Step 2: MSA -> Egyptian
    prefix = "حول للمصري: "
    in2_text = prefix + msa

    in2 = tok_e(in2_text, return_tensors="pt")
    if DEVICE == "cuda":
        in2 = {k: v.to("cuda") for k, v in in2.items()}

    with torch.no_grad():
        out2 = model_e.generate(
            **in2,
            max_length=max_length,
            num_beams=num_beams,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    egy = tok_e.decode(out2[0], skip_special_tokens=True).strip()
    return egy, msa


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Darija → Egyptian Translator", layout="centered")
st.title("🇲🇦 Darija → 🇪🇬 Egyptian Translator")

st.caption(f"Running on: **{DEVICE}**")

model_choice = st.radio(
    "Choose translation method:",
    ["NLLB (direct)", "AraT5 (Darija→MSA→Egyptian)"],
    horizontal=True
)

text = st.text_area("Enter Moroccan sentence (Darija):", height=120, placeholder="مثال: كيداير؟ كلشي مزيان؟")

with st.expander("Decoding settings"):
    max_length = st.slider("max_length", 32, 256, 128, 8)
    num_beams = st.slider("num_beams", 1, 10, 5, 1)

translate_btn = st.button("Translate")

if translate_btn:
    if not text.strip():
        st.warning("Please enter a sentence.")
        st.stop()

    with st.spinner("Translating..."):
        if model_choice == "NLLB (direct)":
            tok, model = load_nllb_pipeline()
            out = translate_with_nllb(text, tok, model, max_length=max_length, num_beams=num_beams)

            st.subheader("🇪🇬 Output (Egyptian)")
            st.write(out)

        else:
            tok_d, model_d, tok_e, model_e = load_arat5_pipeline()
            out, msa_mid = translate_with_arat5_two_step(
                text, tok_d, model_d, tok_e, model_e,
                max_length=max_length, num_beams=num_beams
            )

            st.subheader("🇪🇬 Output (Egyptian)")
            st.write(out)

            st.subheader("🧩 Intermediate (MSA)")
            st.write(msa_mid)

st.markdown("---")
st.caption("Tip: If NLLB is slow on CPU, run on GPU or reduce num_beams.")

