#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from openai import AzureOpenAI

MODEL = "gpt-5.5-2026-04-24"
AZURE_ENDPOINT = "https://search.bytedance.net/gpt/openapi/online/multimodal/crawl"
API_GUIDE = Path("/mnt/bn/embodied-lf3/masiyuan/API_GUIDE.md")
ROOT = Path("/home/tiger/.slock/agents/67b6b8d0-4886-4654-b62e-ff04528f88e6")
OUT = ROOT / "artifacts/h3_ref2video_check_20260926"

SOURCE_VIDEO = ROOT / "dashscope_wan27_physion_20260924/input/demo3_original.mp4"
CANDIDATES = [
    {
        "sample_id": "minimax_h3_reference_video_specific_prompt",
        "output_video": ROOT / "dashscope_wan27_physion_20260924/MiniMax_H3_demo3_reference_video_20260924/minimaxh3_reference_video_demo3_result.mp4",
        "request": ROOT / "dashscope_wan27_physion_20260924/MiniMax_H3_demo3_reference_video_20260924/request.redacted.json",
    },
    {
        "sample_id": "minimax_h3_reference_video_blind_prompt",
        "output_video": ROOT / "dashscope_wan27_physion_20260924/MiniMax_H3_demo3_reference_video_blindprompt_20260924/minimaxh3_blind_demo3_result.mp4",
        "request": ROOT / "dashscope_wan27_physion_20260924/MiniMax_H3_demo3_reference_video_blindprompt_20260924/request.redacted.json",
    },
    {
        "sample_id": "minimax_h3_reference_video_gpt55_prompt",
        "output_video": ROOT / "dashscope_wan27_physion_20260924/MiniMax_H3_demo3_reference_video_gpt55prompt_20260924/minimaxh3_gpt55_demo3_result.mp4",
        "request": ROOT / "dashscope_wan27_physion_20260924/MiniMax_H3_demo3_reference_video_gpt55prompt_20260924/request.redacted.json",
    },
]

SYSTEM = "You are a strict video generation quality auditor."
PROMPT = """Review the paired video frame sheet.

Top row: source/reference video supplied to MiniMax-H3 Ref2Video.
Bottom row: generated MiniMax-H3 Ref2Video output.

Judge whether the generated output is a normal usable video for an end-to-end Ref2Video smoke test. This is not asking whether it perfectly fixes the physical glitch; it asks whether the MiniMax-H3 reference-video pipeline is actually usable and not broken.

Return JSON only:
{
  "output_is_decodable_normal_video": true/false,
  "not_static_or_blank": true/false,
  "not_noise_texture_collapse": true/false,
  "preserves_reference_scene": true/false,
  "has_temporal_motion": true/false,
  "usable_ref2video_smoke": true/false,
  "quality": "broken|weak|acceptable|good",
  "main_assessment": "short explanation",
  "reject_reason": "short reason if unusable, otherwise empty"
}

Reject if the output is mostly gray/noisy texture, blank, static, severely corrupted, or loses the source scene completely."""


def key_from_guide() -> str:
    if os.environ.get("AZURE_GPT55_API_KEY"):
        return os.environ["AZURE_GPT55_API_KEY"]
    for line in API_GUIDE.read_text(encoding="utf-8").splitlines():
        if "| GPT 5.5 |" not in line:
            continue
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) >= 3 and parts[-1] and parts[-1] not in {"AK", "API key"}:
            return parts[-1].strip("`")
    raise RuntimeError("GPT-5.5 key not found")


def sample_indices(total: int, n: int) -> list[int]:
    if total <= 1:
        return [0]
    return [round(i * (total - 1) / (n - 1)) for i in range(n)]


def video_stats(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"path": str(path), "opened": False, "exists": path.exists()}
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    sampled = []
    for idx in sample_indices(frames, 5):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            sampled.append(frame)
    cap.release()
    diffs = [float(np.mean(cv2.absdiff(a, b))) for a, b in zip(sampled, sampled[1:])]
    return {
        "path": str(path),
        "opened": True,
        "bytes": path.stat().st_size if path.exists() else 0,
        "frames": frames,
        "fps": fps,
        "duration_sec": frames / fps if fps else None,
        "width": width,
        "height": height,
        "frame_mean_min": min((float(f.mean()) for f in sampled), default=None),
        "frame_mean_max": max((float(f.mean()) for f in sampled), default=None),
        "frame_std_min": min((float(f.std()) for f in sampled), default=None),
        "frame_std_max": max((float(f.std()) for f in sampled), default=None),
        "sample_frame_diff_mean": float(sum(diffs) / len(diffs)) if diffs else None,
    }


def video_frames(path: Path, n: int = 6, size: tuple[int, int] = (224, 126)) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    for idx in sample_indices(total, n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames: {path}")
    return frames


def make_sheet(source: Path, output: Path, out: Path) -> None:
    src = video_frames(source)
    tgt = video_frames(output)
    n = min(len(src), len(tgt))
    src, tgt = src[:n], tgt[:n]
    gap = 8
    h, w = src[0].shape[:2]
    canvas = 255 * np.ones((2 * h + 3 * gap, n * w + (n + 1) * gap, 3), dtype=np.uint8)
    for i in range(n):
        x = gap + i * (w + gap)
        canvas[gap:gap + h, x:x + w] = src[i]
        canvas[2 * gap + h:2 * gap + 2 * h, x:x + w] = tgt[i]
    cv2.putText(canvas, "SOURCE / reference video", (gap, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, "MiniMax-H3 Ref2Video output", (gap, 2 * gap + h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 120, 0), 1, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def extract_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            return json.loads(match.group(0))
    raise RuntimeError("No JSON in GPT response")


def call_gpt(client: AzureOpenAI, sheet: Path, request: dict[str, Any]) -> dict[str, Any]:
    prompt = PROMPT + "\n\nGeneration request, redacted:\n" + json.dumps(request, ensure_ascii=False)[:4000]
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url(sheet)}},
            ]},
        ],
        max_completion_tokens=8192,
    )
    raw = resp.choices[0].message.content or ""
    return {"raw_response": raw, "qc": extract_json(raw)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    client = AzureOpenAI(api_key=key_from_guide(), api_version="2023-07-01-preview", azure_endpoint=AZURE_ENDPOINT, timeout=180)
    results = []
    for cand in CANDIDATES:
        sample_id = cand["sample_id"]
        output = cand["output_video"]
        sheet = OUT / "sheets" / f"{sample_id}.jpg"
        result_path = OUT / f"{sample_id}_gpt55_qc.json"
        request = json.loads(cand["request"].read_text(encoding="utf-8"))
        make_sheet(SOURCE_VIDEO, output, sheet)
        gpt = call_gpt(client, sheet, request)
        result = {
            "sample_id": sample_id,
            "model": MODEL,
            "source_video": str(SOURCE_VIDEO),
            "output_video": str(output),
            "request_file": str(cand["request"]),
            "sheet": str(sheet),
            "source_stats": video_stats(SOURCE_VIDEO),
            "output_stats": video_stats(output),
            **gpt,
        }
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        results.append(result)
        print(json.dumps({"sample_id": sample_id, "qc": result["qc"], "result": str(result_path)}, ensure_ascii=False), flush=True)
    (OUT / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
