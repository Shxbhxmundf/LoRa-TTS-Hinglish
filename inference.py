"""
Inference script for VoxCPM2 + Hinglish LoRA adapters.

Usage:
    python inference.py                                          # defaults
    python inference.py --text "your text here"                  # custom text
    python inference.py --step 50                                # use a different checkpoint
    python inference.py --reference your_speaker.wav             # use your own reference voice
    python inference.py --no-denoiser                            # skip denoiser (faster)
"""

import argparse
import json
import os

import torch
import soundfile as sf
from voxcpm import VoxCPM
from voxcpm.model.voxcpm import LoRAConfig


def load_lora_config(checkpoint_dir: str) -> LoRAConfig:
    """Read lora_config.json from a checkpoint directory and return a LoRAConfig."""
    config_path = os.path.join(checkpoint_dir, "lora_config.json")
    with open(config_path) as f:
        cfg = json.load(f)["lora_config"]

    return LoRAConfig(
        enable_lm=cfg["enable_lm"],
        enable_dit=cfg["enable_dit"],
        enable_proj=cfg.get("enable_proj", False),
        r=cfg["r"],
        alpha=cfg["alpha"],
        dropout=cfg.get("dropout", 0.0),
        target_modules_lm=cfg.get("target_modules_lm", ["q_proj", "v_proj", "k_proj", "o_proj"]),
        target_modules_dit=cfg.get("target_modules_dit", ["q_proj", "v_proj", "k_proj", "o_proj"]),
        target_proj_modules=cfg.get("target_proj_modules", []),
    )


def main():
    parser = argparse.ArgumentParser(description="VoxCPM2 + LoRA inference")
    parser.add_argument("--text", type=str, default=None, help="Text to synthesize")
    parser.add_argument("--reference", type=str, default="reference.wav",
                        help="Path to reference speaker wav (default: reference.wav)")
    parser.add_argument("--step", type=int, default=100,
                        help="Checkpoint step to use (default: 100)")
    parser.add_argument("--output", type=str, default="output.wav",
                        help="Output wav path (default: output.wav)")
    parser.add_argument("--cfg", type=float, default=2.0,
                        help="Classifier-free guidance value (default: 2.0)")
    parser.add_argument("--timesteps", type=int, default=10,
                        help="Diffusion inference timesteps (default: 10)")
    parser.add_argument("--no-denoiser", action="store_true",
                        help="Skip loading the denoiser model")
    args = parser.parse_args()

    # Default demo text (Hinglish)
    if args.text is None:
        args.text = (
            "आज हम डिटरमिनेंट्स पढ़ेंगे, जिनका यूज़ मैट्रिक्स इक्वेशन्स और लीनियर सिस्टम्स को सॉल्व करने के लिए व्यापक रूप से होता है।"
        )

    # Resolve checkpoint directory
    checkpoint_dir = os.path.join("lora_only", f"step_{args.step:07d}")
    if not os.path.isdir(checkpoint_dir):
        available = sorted(os.listdir("lora_only"))
        raise FileNotFoundError(
            f"Checkpoint '{checkpoint_dir}' not found. Available: {available}"
        )

    # Load LoRA config from checkpoint
    lora_config = load_lora_config(checkpoint_dir)
    print(f"LoRA config: r={lora_config.r}, alpha={lora_config.alpha}, "
          f"lm={lora_config.enable_lm}, dit={lora_config.enable_dit}")

    # Load VoxCPM2 base model + inject LoRA adapters
    print("Loading VoxCPM2 with LoRA adapters...")
    model = VoxCPM.from_pretrained(
        "openbmb/VoxCPM2",
        load_denoiser=not args.no_denoiser,
        lora_config=lora_config,
        lora_weights_path=checkpoint_dir,
    )
    print(f"Model loaded. LoRA enabled: {model.lora_enabled}")

    # Generate speech
    print(f"Generating: \"{args.text[:80]}...\"")
    wav = model.generate(
        text=args.text,
        reference_wav_path=args.reference,
        cfg_value=args.cfg,
        inference_timesteps=args.timesteps,
    )
    sf.write(args.output, wav, model.tts_model.sample_rate)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
