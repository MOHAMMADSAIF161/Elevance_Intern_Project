# Trained LoRA Model

The trained LoRA weights are hosted on Google Drive because they exceed GitHub's file size limit.

## Download LoRA Weights

Google Drive:
https://drive.google.com/drive/folders/1qzrgbGc_tAXLihAVihSTkCirD38BN6D-?usp=sharing

## Files Included

- pytorch_lora_weights.safetensors
- adapter_config.json (if present)
- Other LoRA-related files

## Usage

1. Download the `lora_output` folder.
2. Place it inside the project directory:

```text
model/
└── lora_output/
```

3. Run inference:

```bash
python inference.py
```