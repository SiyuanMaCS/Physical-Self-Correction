#!/usr/bin/env python3
import json
import os
import time
import urllib.request
from pathlib import Path

ENDPOINT = "https://ws-c8sw7d257aos18yg.cn-beijing.maas.aliyuncs.com/api/v1/tasks/{task_id}"

TASKS = [
    ("Wan2.7 GPT55", "b35b6edf-dafb-4da2-abb4-e9909254d5cc", "dashscope_wan27_physion_20260924/demo3_wan27_gpt55prompt_20260924"),
    ("Wan3.0 GPT55", "944322e3-3a9d-4ebb-a2f7-f36335362891", "dashscope_wan27_physion_20260924/wan30_video_prime_demo3_reference_video_gpt55prompt_20260924"),
    ("MiniMax-H3 GPT55", "5f9dc8e2-d811-4e11-ae81-67ed596b9275", "dashscope_wan27_physion_20260924/MiniMax_H3_demo3_reference_video_gpt55prompt_20260924"),
]


def fetch_status(api_key, task_id):
    req = urllib.request.Request(
        ENDPOINT.format(task_id=task_id),
        method="GET",
        headers={"Authorization": "Bearer " + api_key},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return {"http_status": resp.status, "response": json.loads(resp.read().decode("utf-8", "replace"))}


def status_of(payload):
    return payload.get("response", {}).get("output", {}).get("task_status")


def video_url_of(payload):
    return payload.get("response", {}).get("output", {}).get("video_url")


def append_history(path, payload):
    history_path = path / "status_history.json"
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text())
        except Exception:
            history = []
    else:
        history = []
    history.append(payload)
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n")
    (path / "status_latest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def download(url, out_path):
    req = urllib.request.Request(url, headers={"User-Agent": "mystic-raft"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = resp.read()
    out_path.write_bytes(data)
    return len(data)


def main():
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit("missing DASHSCOPE_API_KEY")
    rows = []
    for label, task_id, outdir in TASKS:
        path = Path(outdir)
        path.mkdir(parents=True, exist_ok=True)
        payload = fetch_status(api_key, task_id)
        payload["checked_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        append_history(path, payload)
        st = status_of(payload)
        url = video_url_of(payload)
        local = ""
        if st == "SUCCEEDED" and url:
            safe_label = label.lower().replace(".", "").replace("-", "").replace(" ", "_")
            local_path = path / f"{safe_label}_demo3_result.mp4"
            if not local_path.exists() or local_path.stat().st_size == 0:
                size = download(url, local_path)
            else:
                size = local_path.stat().st_size
            local = f"{local_path} ({size} bytes)"
        output = payload.get("response", {}).get("output", {})
        rows.append({
            "label": label,
            "task_id": task_id,
            "status": st,
            "code": output.get("code", ""),
            "message": output.get("message", ""),
            "video_url": url or "",
            "local": local,
        })
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
