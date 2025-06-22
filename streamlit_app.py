import os
import sys
import json
import subprocess
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import torch
import soundfile as sf
from transformers import MusicgenProcessor, MusicgenForConditionalGeneration
from peft import PeftConfig, PeftModel

# ─── 1) Configuration & client OpenAI ─────────────────────────────
load_dotenv()  # charge OPENAI_API_KEY
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL_ID      = "facebook/musicgen-medium"
ADAPTER_DIR   = Path("adapter_checkpoint")
AUDIO_DIR     = Path("audio")
FEEDBACK_FILE = Path("feedback.jsonl")

THRESHOLD     = 0.5  # score minimal pour retenir un feedback
RETRAIN_EVERY = 1    # relancer le fine-tuning après 5 feedbacks positifs

ADAPTER_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)

# ─── 2) Chargement du modèle & du processeur ────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    processor = MusicgenProcessor.from_pretrained(MODEL_ID)
    model = MusicgenForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    # si un adapter LoRA existe, on le charge
    if (ADAPTER_DIR / "adapter_config.json").exists():
        peft_cfg = PeftConfig.from_pretrained(ADAPTER_DIR)
        model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return processor, model

processor, model = load_models()

# ─── 3) Scoring via GPT-3.5 ────────────────────────────────────────
def get_feedback_score(feedback: str) -> float:
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "Vous êtes un évaluateur. "
                    "Notez le commentaire de l’utilisateur sur une échelle "
                    "de 0.0 (très négatif) à 1.0 (très positif). "
                    "Répondez uniquement par un nombre entre 0 et 1."
                )
            },
            {"role": "user", "content": feedback}
        ],
        temperature=0.0,
    )
    texte = resp.choices[0].message.content.strip().split()[0]
    try:
        score = float(texte)
    except:
        score = 0.0
    return max(0.0, min(1.0, score))

# ─── 4) Relance asynchrone du fine-tuning ──────────────────────────
def maybe_retrain():
    if not FEEDBACK_FILE.exists():
        return
    nombre = sum(1 for _ in FEEDBACK_FILE.open("r", encoding="utf-8") if _.strip())
    if nombre and nombre % RETRAIN_EVERY == 0:
        try:
            subprocess.Popen(
                [sys.executable, "train_feedback.py"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=os.environ,
            )
            st.info("🚀 Fine-tuning en cours…")
        except Exception as e:
            st.error(f"Erreur lors du lancement du fine-tuning : {e}")

# ─── 5) Interface Streamlit ────────────────────────────────────────
st.set_page_config(page_title="MusicGen + GPT Feedback", layout="wide")
st.title("🎹 MusicGen + GPT – Boucle de feedback")

# 1️⃣ Saisie du prompt
prompt = st.text_input("Entrez votre prompt texte :", value="Une douce mélodie de piano")

# 2️⃣ Génération audio
if st.button("Générer l’audio"):
    if not prompt.strip():
        st.warning("Veuillez saisir un prompt avant de générer.")
    else:
        inputs   = processor(text=[prompt], return_tensors="pt").to(model.device)
        wav_ids  = model.generate(**inputs, do_sample=True, max_new_tokens=256)[0]
        wav_arr  = wav_ids.cpu().numpy().T
        sr       = processor.feature_extractor.sampling_rate

        # nommage unique
        idx      = len(list(AUDIO_DIR.glob("out_*.wav"))) + 1
        out_path = AUDIO_DIR / f"out_{idx}.wav"
        sf.write(str(out_path), wav_arr, sr)

        st.success(f"Audio généré et sauvegardé : `{out_path.name}`")
        st.audio(str(out_path), format="audio/wav")
        st.session_state["last_audio"] = str(out_path)

# 3️⃣ Feedback & enregistrement
if "last_audio" in st.session_state:
    st.markdown("---")
    st.subheader("Évaluez cet extrait")
    feedback = st.text_area("Votre commentaire :", height=120)

    if st.button("Envoyer et scorer"):
        if not feedback.strip():
            st.error("Veuillez saisir un commentaire pour l’évaluation.")
        else:
            score = get_feedback_score(feedback)
            st.write(f"**Score GPT :** {score:.2f}")

            if score >= THRESHOLD:
                rec = {
                    "text":       prompt,
                    "audio_path": st.session_state["last_audio"]
                }
                with FEEDBACK_FILE.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                st.success("🎉 Feedback positif enregistré !")
                maybe_retrain()
            else:
                st.info("Feedback trop bas, non enregistré.")

        del st.session_state["last_audio"]
