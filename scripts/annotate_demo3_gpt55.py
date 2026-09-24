#!/usr/bin/env python3
import base64
import json
import mimetypes
import os
import re
import sys
from pathlib import Path

import cv2
from openai import AzureOpenAI

AZURE_ENDPOINT = "https://search.bytedance.net/gpt/openapi/online/multimodal/crawl"
API_GUIDE = Path("/mnt/bn/embodied-lf3/masiyuan/API_GUIDE.md")
MODEL = "gpt-5.5-2026-04-24"
VIDEO = Path("dashscope_wan27_physion_20260924/input/demo3_original.mp4")
OUTDIR = Path("dashscope_wan27_physion_20260924/gpt55_demo3_glitch_annotation_20260924")

HF_URL = "https://huggingface.co/datasets/HuggingFriends/physion-eval-videoedit-100/resolve/main/videos/006_1bb7dbd3-23a8-5b4e-aa06-e168c19a1497_P35_108_277_2c50f65d7115f9a1_processed_h.mp4"

SYSTEM = "You are an expert video-physics annotator for Physion-Eval style physical-glitch analysis."
PROMPT = """Inspect the chronological video frames and identify the main physical glitch in the video.

Write the annotation in the same style as Physion-Eval glitch text: concise but specific, with physical category cues and time intervals when visible. Do not mention that you saw sampled frames. Do not propose an edit. Return JSON only with these fields:
{
  "glitch_category": "...",
  "physion_style_annotation": "...",
  "evidence_intervals": ["..."],
  "short_edit_prompt": "..."
}

The annotation should describe what is physically unrealistic in the video, not score the video. The short_edit_prompt should be suitable for asking a video-editing model to fix the detected glitch while preserving unrelated content."""


def key_from_guide():
    env = os.environ.get("AZURE_GPT55_API_KEY")
    if env:
        return env
    if not API_GUIDE.exists():
        raise RuntimeError("missing API guide")
    for line in API_GUIDE.read_text(encoding="utf-8").splitlines():
        if "| GPT 5.5 |" not in line:
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) >= 3 and parts[-1] and parts[-1] not in ("API key", "AK"):
            return parts[-1].strip("`")
    raise RuntimeError("missing GPT 5.5 key")


def download_video():
    if VIDEO.exists() and VIDEO.stat().st_size > 0:
        return
    import urllib.request
    VIDEO.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(HF_URL, headers={"User-Agent": "mystic-raft"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        VIDEO.write_bytes(resp.read())


def sample_frames(video_path, n=12, max_size=768):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    if total <= 0:
        raise RuntimeError("no frames")
    idxs = list(range(total)) if total <= n else [round(i * (total - 1) / (n - 1)) for i in range(n)]
    frames = []
    frame_dir = OUTDIR / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for seq, idx in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        scale = min(max_size / max(w, h), 1.0)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        frame_path = frame_dir / f"frame_{seq:02d}_t{idx / max(fps, 1e-6):.2f}s.jpg"
        cv2.imwrite(str(frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
        with open(frame_path, "rb") as fh:
            encoded = base64.b64encode(fh.read()).decode("ascii")
        mime = mimetypes.guess_type(str(frame_path))[0] or "image/jpeg"
        frames.append({"path": str(frame_path), "time": idx / max(fps, 1e-6), "data_url": f"data:{mime};base64,{encoded}"})
    cap.release()
    return frames


def extract_json(text):
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        return json.loads(match.group(0))
    raise RuntimeError("response did not contain JSON")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    download_video()
    frames = sample_frames(VIDEO)
    content = [{"type": "text", "text": PROMPT}]
    for frame in frames:
        content.append({"type": "image_url", "image_url": {"url": frame["data_url"]}})
    client = AzureOpenAI(
        api_key=key_from_guide(),
        api_version="2023-07-01-preview",
        azure_endpoint=AZURE_ENDPOINT,
        timeout=180,
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": content},
        ],
        max_completion_tokens=8192,
    )
    text = response.choices[0].message.content or ""
    parsed = extract_json(text)
    result = {
        "model": MODEL,
        "input_video": HF_URL,
        "sampled_frames": [{"path": f["path"], "time": f["time"]} for f in frames],
        "raw_response": text,
        "annotation": parsed,
    }
    (OUTDIR / "gpt55_physion_style_annotation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (OUTDIR / "gpt55_physion_style_annotation.md").write_text(
        "# GPT-5.5 Physion-Style Glitch Annotation\n\n"
        f"Video: <{HF_URL}>\n\n"
        f"Category: {parsed.get('glitch_category', '')}\n\n"
        f"Annotation:\n\n{parsed.get('physion_style_annotation', '')}\n\n"
        f"Short edit prompt:\n\n{parsed.get('short_edit_prompt', '')}\n",
        encoding="utf-8",
    )
    print(json.dumps(parsed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
