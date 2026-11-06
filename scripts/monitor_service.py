#!/usr/bin/env python3
"""UTF-8 Korean monitor for before/during/after deployment observations."""

import argparse
import json
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

PNG = bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d49444154789c63f8cfc0f01f00050001ff89993d1d0000000049454e44ae426082")


def timestamp():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def check(url):
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"fixed-monitor.png\"\r\nContent-Type: image/png\r\n\r\n").encode() + PNG + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(url.rstrip("/") + "/api/classify", data=body, method="POST")
    request.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    started = time.monotonic()
    started_at = timestamp()
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
            status = response.status
        error = ""
    except Exception as exc:
        payload, status, error = {}, 0, str(exc)
    return {
        "start": started_at,
        "end": timestamp(),
        "response_ms": round((time.monotonic() - started) * 1000, 1),
        "status": status,
        "analysis_ok": bool(payload.get("ok")),
        "version": payload.get("version", "확인불가"),
        "error": error,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--duration", type=int, default=45)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    rows, gaps = [], []
    started = time.monotonic()
    previous = None
    with args.log.open("w", encoding="utf-8") as output:
        output.write("분리수거 서비스 무중단 배포 관측 로그 (UTF-8)\n")
        output.write(f"검증 주소: {args.url}\n관측 시작: {timestamp()}\n요청 간격 목표: 1초\n\n")
        while time.monotonic() - started < args.duration:
            request_started = time.monotonic()
            row = check(args.url)
            if previous is not None:
                gaps.append(round((request_started - previous) * 1000, 1))
            previous = request_started
            rows.append(row)
            line = (f"[요청] 시작={row['start']} 종료={row['end']} 응답시간={row['response_ms']}ms "
                    f"상태={row['status']} 분석={'정상' if row['analysis_ok'] else '실패'} 배포버전={row['version']}")
            if row["error"]:
                line += f" 오류={row['error']}"
            print(line, flush=True)
            output.write(line + "\n")
            output.flush()
            time.sleep(max(0, 1 - (time.monotonic() - request_started)))
    success_count = sum(row["analysis_ok"] for row in rows)
    max_gap = max(gaps, default=0)
    summary = [
        "## 배포 무중단 검증 요약", "",
        f"- 검증 주소: `{args.url}`",
        f"- 관측 종료: `{timestamp()}`",
        f"- 전체 요청: **{len(rows)}건** (성공 {success_count}건 / 실패 {len(rows) - success_count}건)",
        "- 실제 요청 간격 목표: 1초",
        f"- 최대 기록 공백: **{max_gap}ms** (공백 구간은 검증되지 않은 구간)",
        f"- 관측 범위 무중단 판정: **{'통과' if rows and success_count == len(rows) else '실패'}**", "",
        "### 대표 응답", "",
    ]
    for label, row in [("배포 전/첫 요청", rows[0] if rows else None), ("배포 중/중간 요청", rows[len(rows) // 2] if rows else None), ("배포 후/마지막 요청", rows[-1] if rows else None)]:
        summary.append(f"- {label}: " + (f"상태 {row['status']}, {row['response_ms']}ms, 분석={'정상' if row['analysis_ok'] else '실패'}, 버전 `{row['version']}`" if row else "기록 없음"))
    args.summary.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary), flush=True)


if __name__ == "__main__":
    main()
