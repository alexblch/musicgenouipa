import os
import sys
import subprocess
from pathlib import Path
from transformers import AutoConfig

MODEL_ID      = "facebook/musicgen-medium"
ADAPTER_DIR   = Path("adapter_checkpoint")
FEEDBACK_FILE = Path("feedback.jsonl")

def main():
    # si pas de feedback, on s'arrête
    if not FEEDBACK_FILE.exists() or FEEDBACK_FILE.stat().st_size == 0:
        print("❌ Aucun feedback positif à entraîner.")
        return

    # on récupère pad_token_id & decoder_start_token_id
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    pad = cfg.pad_token_id or 2048
    dec = cfg.decoder_start_token_id or pad

    cmd = [
        sys.executable,
        "musicgen-dreamboothing/dreambooth_musicgen.py",
        "--use_lora",
        "--model_name_or_path", MODEL_ID,

        # on utilise le loader JSON de 🤗 datasets
        "--dataset_name",     "json",
        "--data_files",       str(FEEDBACK_FILE),
        "--train_split_name", "train",
        "--eval_split_name",  "train",

        "--target_audio_column_name", "audio_path",
        "--text_column_name",         "text",

        "--output_dir", str(ADAPTER_DIR),
        "--do_train", "--fp16",
        "--num_train_epochs",           "1",
        "--gradient_checkpointing",
        "--per_device_train_batch_size", "1",
        "--learning_rate",               "2e-4",
        "--overwrite_output_dir",

        # désactive WandB
        "--report_to", "none",

        # évite l’UnboundLocalError du script officiel
        "--pad_token_id",           str(pad),
        "--decoder_start_token_id", str(dec),
    ]

    print("🚀 Lancement du fine-tuning LoRA…")
    subprocess.check_call(cmd, env=os.environ)

if __name__ == "__main__":
    main()
