# VoxCPM2 Hinglish LoRA Adapters

Fine-tuned LoRA adapters for [VoxCPM2](https://github.com/OpenBMB/VoxCPM) (text-to-speech). These adapters were trained on Hinglish (Hindi + English) data to improve code-mixed speech synthesis.

## What's included

```
.
├── lora_only/              # LoRA checkpoint weights
│   ├── step_0000000/       # Checkpoints at various training steps
│   ├── step_0000010/
│   ├── ...
│   └── step_0000100/       # Final checkpoint (recommended)
│       ├── lora_config.json
│       └── lora_weights.safetensors
├── reference.wav           # Sample reference speaker voice
├── inference.py            # LoRA + reference voice inference
├── inference_base.py       # Base model only (no LoRA, no reference audio)
├── requirements.txt
└── README.md
```

Each checkpoint is ~35 MB. The adapters target attention layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`) in both the language model and the diffusion transformer, with rank 16 and alpha 32.

## Setup

Requires **Python 3.10+** and a working PyTorch installation.

### 1. Clone this repo

```bash
git clone <this-repo-url>
cd "LoRa adapter TTS"
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs `voxcpm` (which pulls in PyTorch, safetensors, etc.) and `soundfile` for wav I/O.

> **Note:** On first run, VoxCPM2 base model weights (~3 GB) will be downloaded from HuggingFace automatically.

### 4. (Linux only) Install libsndfile

```bash
sudo apt-get install libsndfile1
```

## Usage

### Quick start

```bash
python inference.py
```

This loads the step 100 checkpoint, uses the included `reference.wav` for voice cloning, generates a sample Hinglish sentence, and saves to `output.wav`.

### Custom text

```bash
python inference.py --text "अच्छा तो आज हम पढ़ेंगे fractions के बारे में, ready हो?"
```

### Use a different checkpoint step

```bash
python inference.py --step 50
```

### Use your own reference voice

```bash
python inference.py --reference /path/to/your/speaker.wav
```

### All options

```
python inference.py --help

  --text TEXT           Text to synthesize
  --reference PATH      Reference speaker wav (default: reference.wav)
  --step INT            Checkpoint step to use (default: 100)
  --output PATH         Output wav path (default: output.wav)
  --cfg FLOAT           Classifier-free guidance value (default: 2.0)
  --timesteps INT       Diffusion inference steps (default: 10)
  --no-denoiser         Skip loading the denoiser (faster startup)
```

## Base model comparison (no LoRA, no reference audio)

`inference_base.py` runs vanilla VoxCPM2 in zero-shot mode — no adapters, no reference voice. Useful for comparing against the LoRA-enhanced output.

```bash
python inference_base.py
```

This generates the same default text using the base model and saves to `output_base.wav`.

You can also use VoxCPM2's voice design feature to describe a voice via text:

```bash
python inference_base.py --voice "warm male voice"
```

```
python inference_base.py --help

  --text TEXT           Text to synthesize
  --voice TEXT          Voice design description, e.g. 'warm male voice'
  --output PATH         Output wav path (default: output_base.wav)
  --cfg FLOAT           Classifier-free guidance value (default: 2.0)
  --timesteps INT       Diffusion inference steps (default: 10)
  --no-denoiser         Skip loading the denoiser (faster startup)
```

## Loading the adapters in your own code

```python
import json
from voxcpm import VoxCPM
from voxcpm.model.voxcpm import LoRAConfig

# 1. Build the LoRA config matching the training setup
with open("lora_only/step_0000100/lora_config.json") as f:
    cfg = json.load(f)["lora_config"]

lora_config = LoRAConfig(
    enable_lm=cfg["enable_lm"],       # True
    enable_dit=cfg["enable_dit"],      # True
    enable_proj=cfg["enable_proj"],    # False
    r=cfg["r"],                        # 16
    alpha=cfg["alpha"],                # 32
)

# 2. Load VoxCPM2 base + inject LoRA weights
model = VoxCPM.from_pretrained(
    "openbmb/VoxCPM2",
    load_denoiser=False,
    lora_config=lora_config,
    lora_weights_path="lora_only/step_0000100",
)

# 3. Generate
wav = model.generate(
    text="your text here",
    reference_wav_path="reference.wav",
)

# 4. You can toggle LoRA on/off at runtime
model.set_lora_enabled(False)   # base model only
model.set_lora_enabled(True)    # adapters active again
```


## LoRA training config

| Parameter | Value |
|-----------|-------|
| Rank (r) | 16 |
| Alpha | 32 |
| Dropout | 0.0 |
| LM targets | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| DiT targets | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Projection layers | disabled |
| Base model | `openbmb/VoxCPM2` |
