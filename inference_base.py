"""
Inference script for vanilla VoxCPM2 — no LoRA, no reference audio.
Uses the model's zero-shot mode to generate speech from text alone.

Usage:
    python inference_base.py                                     # defaults
    python inference_base.py --text "your text here"             # custom text
    python inference_base.py --voice "warm female voice"         # voice design via text description
    python inference_base.py --no-denoiser                       # skip denoiser (faster)
"""

import argparse

import soundfile as sf
from voxcpm import VoxCPM


def main():
    parser = argparse.ArgumentParser(description="VoxCPM2 base inference (no LoRA, no reference audio)")
    parser.add_argument("--text", type=str, default=None, help="Text to synthesize")
    parser.add_argument("--voice", type=str, default=None,
                        help="Voice design description, e.g. 'warm male voice' (optional)")
    parser.add_argument("--output", type=str, default="output_base.wav",
                        help="Output wav path (default: output_base.wav)")
    parser.add_argument("--cfg", type=float, default=2.0,
                        help="Classifier-free guidance value (default: 2.0)")
    parser.add_argument("--timesteps", type=int, default=10,
                        help="Diffusion inference timesteps (default: 10)")
    parser.add_argument("--no-denoiser", action="store_true",
                        help="Skip loading the denoiser model")
    args = parser.parse_args()

    # Same default text as inference.py for comparison
    if args.text is None:
        args.text = (
            "आज हम डिटरमिनेंट्स पढ़ेंगे, जिनका यूज़ मैट्रिक्स इक्वेशन्स और "
            "लीनियर सिस्टम्स को सॉल्व करने के लिए व्यापक रूप से होता है।"
        )

    # Prepend voice design instruction if provided
    if args.voice:
        synth_text = f"({args.voice}){args.text}"
    else:
        synth_text = args.text

    # Load vanilla VoxCPM2 — no LoRA
    print("Loading VoxCPM2 base model (no LoRA)...")
    model = VoxCPM.from_pretrained(
        "openbmb/VoxCPM2",
        load_denoiser=not args.no_denoiser,
    )
    print("Model loaded.")

    # Generate speech — zero-shot, no reference audio
    print(f"Generating (zero-shot): \"{args.text[:80]}...\"")
    wav = model.generate(
        text=synth_text,
        cfg_value=args.cfg,
        inference_timesteps=args.timesteps,
    )
    sf.write(args.output, wav, model.tts_model.sample_rate)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
