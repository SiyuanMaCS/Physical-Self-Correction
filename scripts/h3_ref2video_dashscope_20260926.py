#!/usr/bin/env python3
"""Submit, poll, and QC MiniMax-H3 Ref2Video jobs through DashScope-style API.

Credentials are read only from DASHSCOPE_API_KEY. The script writes redacted
requests and downloaded videos under the output directory.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

ENDPOINT = "https://ws-c8sw7d257aos18yg.cn-beijing.maas.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"
TASK_ENDPOINT = "https://ws-c8sw7d257aos18yg.cn-beijing.maas.aliyuncs.com/api/v1/tasks/{task_id}"
MODEL = "MiniMax/MiniMax-H3"


def require_key() -> str:
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise SystemExit("missing DASHSCOPE_API_KEY")
    return key


def final_url(url: str) -> str:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "mystic-raft"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.geturl()


def submit_job(api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
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


def poll_job(api_key: str, task_id: str) -> dict[str, Any]:
    req = urllib.request.Request(
        TASK_ENDPOINT.format(task_id=task_id),
        method="GET",
        headers={"Authorization": "Bearer " + api_key},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return {"http_status": resp.status, "response": json.loads(resp.read().decode("utf-8", "replace"))}


def output_of(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("response", {}).get("output", {}) or {}


def download_video(url: str, out_path: Path) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": "mystic-raft"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = resp.read()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    return len(data)


def append_history(path: Path, payload: dict[str, Any]) -> None:
    history_path = path / "status_history.json"
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except Exception:
            history = []
    else:
        history = []
    history.append(payload)
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (path / "status_latest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def cmd_submit(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    submit_url = final_url(args.reference_video_url) if args.follow_redirect else args.reference_video_url
    payload = {
        "model": MODEL,
        "input": {
            "prompt": args.prompt,
            "media": [{"type": "reference_video", "url": submit_url}],
        },
        "parameters": {
            "resolution": args.resolution,
            "ratio": args.ratio,
            "duration": args.duration,
            "watermark": args.watermark,
        },
    }
    redacted = json.loads(json.dumps(payload, ensure_ascii=False))
    redacted["input"]["media"][0]["human_url"] = args.reference_video_url
    redacted["input"]["media"][0]["submitted_url_kind"] = "redirect" if args.follow_redirect else "direct"
    (out_dir / "request.redacted.json").write_text(json.dumps(redacted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = submit_job(require_key(), payload)
    result["submitted_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (out_dir / "submit_response.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "task": output_of(result)}, ensure_ascii=False, indent=2))


def cmd_poll(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    submit_response = out_dir / "submit_response.json"
    if args.task_id:
        task_id = args.task_id
    elif submit_response.exists():
        task_id = output_of(json.loads(submit_response.read_text(encoding="utf-8"))).get("task_id")
    else:
        task_id = None
    if not task_id:
        raise SystemExit("missing task id; pass --task-id or keep submit_response.json in --out-dir")
    latest = poll_job(require_key(), task_id)
    latest["checked_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    append_history(out_dir, latest)
    out = output_of(latest)
    local = ""
    if out.get("task_status") == "SUCCEEDED" and out.get("video_url"):
        local_path = out_dir / args.output_name
        size = local_path.stat().st_size if local_path.exists() and local_path.stat().st_size else download_video(out["video_url"], local_path)
        local = f"{local_path} ({size} bytes)"
    print(json.dumps({"task_id": task_id, "status": out.get("task_status"), "code": out.get("code", ""), "message": out.get("message", ""), "video_url_present": bool(out.get("video_url")), "local": local}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    submit = sub.add_parser("submit")
    submit.add_argument("--reference-video-url", required=True)
    submit.add_argument("--prompt", required=True)
    submit.add_argument("--out-dir", required=True)
    submit.add_argument("--resolution", default="768P")
    submit.add_argument("--ratio", default="16:9")
    submit.add_argument("--duration", type=int, default=5)
    submit.add_argument("--watermark", action=argparse.BooleanOptionalAction, default=True)
    submit.add_argument("--follow-redirect", action=argparse.BooleanOptionalAction, default=True)
    submit.set_defaults(func=cmd_submit)

    poll = sub.add_parser("poll")
    poll.add_argument("--out-dir", required=True)
    poll.add_argument("--task-id", default="")
    poll.add_argument("--output-name", default="minimax_h3_ref2video_result.mp4")
    poll.set_defaults(func=cmd_poll)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
