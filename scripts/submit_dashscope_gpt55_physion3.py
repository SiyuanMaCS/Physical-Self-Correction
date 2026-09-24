#!/usr/bin/env python3
import json
import os
import time
import urllib.request
from pathlib import Path

ENDPOINT = "https://ws-c8sw7d257aos18yg.cn-beijing.maas.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"
HF_URL = "https://huggingface.co/datasets/HuggingFriends/physion-eval-videoedit-100/resolve/main/videos/006_1bb7dbd3-23a8-5b4e-aa06-e168c19a1497_P35_108_277_2c50f65d7115f9a1_processed_h.mp4"
ANNOTATION_PATH = Path("dashscope_wan27_physion_20260924/gpt55_demo3_glitch_annotation_20260924/gpt55_physion_style_annotation.json")

TASKS = [
    (
        "Wan2.7 GPT55",
        "dashscope_wan27_physion_20260924/demo3_wan27_gpt55prompt_20260924",
        {
            "model": "wan2.7-videoedit",
            "input": {"prompt": None, "media": [{"type": "video", "url": None}]},
            "parameters": {"resolution": "720P", "prompt_extend": True, "watermark": True},
        },
    ),
    (
        "Wan3.0 GPT55",
        "dashscope_wan27_physion_20260924/wan30_video_prime_demo3_reference_video_gpt55prompt_20260924",
        {
            "model": "wan3.0-video-prime",
            "input": {"prompt": None, "media": [{"type": "reference_video", "url": None}]},
            "parameters": {"resolution": "480P", "ratio": "adaptive", "duration": 5},
        },
    ),
    (
        "MiniMax-H3 GPT55",
        "dashscope_wan27_physion_20260924/MiniMax_H3_demo3_reference_video_gpt55prompt_20260924",
        {
            "model": "MiniMax/MiniMax-H3",
            "input": {"prompt": None, "media": [{"type": "reference_video", "url": None}]},
            "parameters": {"resolution": "768P", "ratio": "16:9", "duration": 5, "watermark": True},
        },
    ),
]


def final_url(url):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "mystic-raft"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.geturl()


def submit(api_key, payload):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        ENDPOINT,
        data=data,
        method="POST",
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return {"http_status": resp.status, "response": json.loads(resp.read().decode("utf-8", "replace"))}


def main():
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit("missing DASHSCOPE_API_KEY")
    ann = json.loads(ANNOTATION_PATH.read_text(encoding="utf-8"))["annotation"]
    prompt = (
        "Edit this video to fix the following physical glitch: "
        + ann["physion_style_annotation"].strip()
        + " Keep the camera, scene, hands, sink, faucet, mug, lighting, timing, and all unrelated content unchanged as much as possible."
    )
    submit_url = final_url(HF_URL)
    rows = []
    for label, outdir, payload in TASKS:
        path = Path(outdir)
        path.mkdir(parents=True, exist_ok=True)
        payload = json.loads(json.dumps(payload))
        payload["input"]["prompt"] = prompt
        payload["input"]["media"][0]["url"] = submit_url
        redacted = json.loads(json.dumps(payload))
        redacted["input"]["media"][0]["human_url"] = HF_URL
        redacted["input"]["media"][0]["submitted_url_kind"] = "hf_cdn_redirect"
        redacted["gpt55_annotation"] = ann
        path.joinpath("request.redacted.json").write_text(json.dumps(redacted, ensure_ascii=False, indent=2) + "\n")
        result = submit(api_key, payload)
        result["submitted_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        path.joinpath("submit_response.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        output = result.get("response", {}).get("output", {})
        rows.append({"label": label, "outdir": outdir, "task_id": output.get("task_id"), "task_status": output.get("task_status"), "code": output.get("code", ""), "message": output.get("message", "")})
    print(json.dumps({"gpt55_annotation": ann, "submitted": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
