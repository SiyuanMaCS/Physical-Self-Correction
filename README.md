# Physical Self-Correction

Physical Self-Correction is a small demo repository for testing whether video generation/editing models can correct physical glitches in generated videos.

The current demo uses one Physion-Eval sample and compares three correction settings:

1. **GT glitch prompt**: the edit model receives the original Physion-Eval glitch annotation.
2. **Blind prompt**: the edit model is only asked to fix physically unrealistic behavior, without the dataset annotation.
3. **GPT-5.5 detected prompt**: GPT-5.5 first detects a Physion-style glitch from the video, then the edit model uses that detected annotation as the repair prompt.

Tested edit models:

- Wan2.7 video edit
- Wan3.0 video prime
- MiniMax-H3

## Dataset

The 100-sample Physion-Eval demo package is hosted at:

<https://huggingface.co/datasets/HuggingFriends/physion-eval-videoedit-100>

The demo sample used here is sample #3:

- `source_item_id`: `1bb7dbd3-23a8-5b4e-aa06-e168c19a1497`
- category: `Contact / Interaction Failure`
- original video: <https://huggingface.co/datasets/HuggingFriends/physion-eval-videoedit-100/resolve/main/videos/006_1bb7dbd3-23a8-5b4e-aa06-e168c19a1497_P35_108_277_2c50f65d7115f9a1_processed_h.mp4>
- contact sheet: <https://huggingface.co/datasets/HuggingFriends/physion-eval-videoedit-100/resolve/main/review10_contact_sheets/03_1bb7dbd3-23a8-5b4e-aa06-e168c19a1497.jpg>

Local copies of the demo video, contact sheet, and generated results are included under `assets/`.

## Demo Report

See [docs/demo_sample3.md](docs/demo_sample3.md) for the full prompt settings, GPT-5.5 annotation, model task IDs, and output video paths.

## Repository Layout

```text
assets/
  contact_sheets/                  # contact sheet for the demo sample
  videos/demo3/original/            # original Physion-Eval sample
  videos/demo3/gt-glitch-prompt/    # outputs using the GT Physion glitch text
  videos/demo3/blind-prompt/        # outputs using a generic physical-repair prompt
  videos/demo3/gpt55-detected-prompt/ # outputs using GPT-5.5 detected glitch text
metadata/
  hf100_manifest.jsonl              # 100-sample package manifest
  hf100_alignment_report.json       # alignment check for the HF package
  gpt55_physion_style_annotation.json
  demo3_tasks.json
scripts/
  annotate_demo3_gpt55.py           # GPT-5.5 frame-based glitch annotation script
  submit_dashscope_*.py             # DashScope submission scripts
  poll_dashscope_*.py               # DashScope polling/downloading scripts
```

## Reproducibility Notes

The scripts read API keys from environment variables or local secret guides. Do not commit credentials. The generated output videos are committed here so the demo remains inspectable after temporary provider URLs expire.

