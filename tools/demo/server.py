"""테스트 프로그램 로컬 서버 — 강조 UI + nuScenes 이미지 + 영상 업로드 처리.

    .venv/bin/python tools/demo/server.py          # → http://127.0.0.1:8765 (WSL이면 Windows 브라우저에서 그대로 열린다)

먼저 `tools/demo/export.py`로 nuScenes 장면 결과와 임계값(results/demo/)을 만들어 둔다.
업로드 영상은 results/demo/uploads/{id}/에 프레임과 결과 JSON으로 남고, 원본 영상은 처리 후 지운다.
로컬 주소(127.0.0.1)에만 열린다 — 업로드한 영상이 이 PC 밖으로 나가지 않는다.

경로
  GET  /                 강조 UI (index.html)
  GET  /exhibit          전시 모드 (exhibit.html) — 먼저 tools/demo/exhibit_build.py
  GET  /api/index        장면 목록 (index.json + 업로드 목록)
  GET  /data/<경로>      results/demo/ 아래 JSON·프레임
  GET  /img/<경로>       nuScenes 이미지(dataroot 기준) — 표시용 축소본을 cache/demo_thumbs에 만들어 둔다
  POST /api/upload       영상 업로드(multipart) → {job}
  GET  /api/job/<id>     처리 진행 상황
  POST /api/delete/<id>  업로드 결과 삭제
"""

from __future__ import annotations

import argparse
import cgi
import json
import os
import re
import shutil
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

sys.path.insert(0, os.path.dirname(__file__))

import core  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TYPES = {".json": "application/json; charset=utf-8", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
         ".png": "image/png", ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8"}

CFG: dict = {}
JOBS: dict[str, core.Job] = {}
WORK = threading.Lock()                 # 업로드는 한 번에 하나씩 (GPU 하나)
_FX: list = []


def inside(base: str, rel: str) -> str | None:
    """base 밖으로 나가는 경로(../)는 거절한다."""
    base = os.path.realpath(base)
    p = os.path.realpath(os.path.join(base, rel))
    return p if p == base or p.startswith(base + os.sep) else None


def thumb(rel: str) -> str | None:
    """nuScenes 원본(1600×900) → 표시용 축소본. 12 Hz × 6채널을 원본으로 보내면 브라우저가 버거워한다."""
    import cv2
    src = inside(CFG["dataroot"], rel)
    if not src or not os.path.isfile(src):
        return None
    dst = os.path.join(CFG["thumb_dir"], rel)
    if not os.path.exists(dst):
        img = cv2.imread(src)
        if img is None:
            return None
        h, w = img.shape[:2]
        tw = int(CFG["thumb_width"])
        img = cv2.resize(img, (tw, int(round(h * tw / w))), interpolation=cv2.INTER_AREA)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = f"{dst}.{threading.get_ident()}.tmp.jpg"
        cv2.imwrite(tmp, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        os.replace(tmp, dst)
    return dst


def uploads_index() -> list[dict]:
    d = os.path.join(CFG["out_dir"], "uploads")
    out = []
    if not os.path.isdir(d):
        return out
    for jid in sorted(os.listdir(d), reverse=True):
        p = os.path.join(d, jid, "scene.json")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            s = json.load(f)
        kfs = s["keyframes"]
        out.append(dict(name=s["scene"], id=jid, file=f"data/uploads/{jid}/scene.json", n_kf=len(kfs),
                        dur=kfs[-1]["t"] if kfs else 0, missing=s.get("missing_cams", []),
                        can=s.get("can_given", False),
                        created=time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(p)))))
    return out


def run_job(job: core.Job, job_dir: str, spec: dict):
    with WORK:
        try:
            if not _FX:
                job.set("영상 특징 추출기(MobileNetV3) 불러오는 중")
                _FX.append(core.FeatureExtractor(core.device_name()))
            path = core.process_upload(job, job_dir, spec, CFG, _FX[0], core.device_name())
            job.file = "data/" + os.path.relpath(path, CFG["out_dir"]).replace(os.sep, "/")
            job.state, job.msg = "done", "완료"
        except Exception as e:  # noqa: BLE001 — 오류는 화면에 그대로 보여준다
            job.state, job.error, job.msg = "error", f"{type(e).__name__}: {e}", "실패"
            print(f"[업로드 {job.id}] {job.error}", file=sys.stderr)


class Handler(BaseHTTPRequestHandler):
    server_version = "ViewSelectDemo/1.0"

    def log_message(self, fmt, *args):     # 이미지 요청이 많아 로그는 오류만
        if args and str(args[1])[:1] in "45":
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", TYPES[".json"])
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path, cache=False):
        if not path or not os.path.isfile(path):
            self.send_error(404)
            return
        with open(path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", TYPES.get(os.path.splitext(path)[1].lower(), "application/octet-stream"))
        self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            return self.send_file(os.path.join(HERE, "index.html"))
        if path in ("/exhibit", "/exhibit.html"):
            return self.send_file(os.path.join(HERE, "exhibit.html"))
        if path == "/api/index":
            p = os.path.join(CFG["out_dir"], "index.json")
            idx = {"sources": {}}
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    idx = json.load(f)
            for k, v in idx["sources"].items():          # 제목·설명은 설정 파일이 우선 (내보내기 뒤에 바꿔도 반영)
                if k in CFG["sources"]:
                    v.update(title=CFG["sources"][k]["title"], note=CFG["sources"][k]["note"])
            up = CFG["sources"]["upload"]
            idx["sources"]["upload"] = dict(title=up["title"], note=up["note"], default=up["default"],
                                            scenes=uploads_index())
            idx["cfg"] = dict(display_fps=CFG["upload"]["display_fps"], key_hz=CFG["upload"]["key_hz"],
                              max_seconds=CFG["upload"]["max_seconds"],
                              methods={m: CFG["methods"][m]["name"] for m in up["methods"]},
                              thresholds=sorted(core.load_thresholds(CFG)))
            return self.send_json(idx)
        if path.startswith("/api/job/"):
            job = JOBS.get(path.rsplit("/", 1)[-1])
            return self.send_json(job.as_dict() if job else {"state": "unknown"}, 200 if job else 404)
        if path.startswith("/data/"):
            p = inside(CFG["out_dir"], path[len("/data/"):])
            return self.send_file(p, cache=bool(p) and p.endswith(".jpg"))
        if path.startswith("/img/"):
            return self.send_file(thumb(path[len("/img/"):]), cache=True)
        self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path.startswith("/api/delete/"):
            jid = path.rsplit("/", 1)[-1]
            if not re.fullmatch(r"[0-9a-z_-]+", jid):
                return self.send_json({"ok": False}, 400)
            d = inside(os.path.join(CFG["out_dir"], "uploads"), jid)
            if d and os.path.isdir(d):
                shutil.rmtree(d)
            return self.send_json({"ok": True})
        if path != "/api/upload":
            return self.send_error(404)
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > int(CFG["upload"]["max_bytes"]):
            return self.send_json({"error": "업로드 크기가 허용 범위를 벗어났습니다"}, 413)
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers,
                                environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers["Content-Type"],
                                         "CONTENT_LENGTH": str(n)})
        jid = time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
        job_dir = os.path.join(CFG["out_dir"], "uploads", jid)
        src = os.path.join(job_dir, "src")
        os.makedirs(src, exist_ok=True)

        def save(field, stem):
            if field not in form or not getattr(form[field], "filename", None):
                return None
            ext = os.path.splitext(form[field].filename)[1].lower()[:6] or ".bin"
            dst = os.path.join(src, stem + re.sub(r"[^.a-z0-9]", "", ext))
            with open(dst, "wb") as f:
                shutil.copyfileobj(form[field].file, f)
            return dst

        mode = form.getfirst("mode", "six")
        videos = {}
        if mode == "mosaic":
            v = save("mosaic", "mosaic")
            if v:
                videos["mosaic"] = v
        else:
            for c in range(6):
                v = save(f"cam{c}", f"cam{c}")
                if v:
                    videos[c] = v
        if not videos:
            shutil.rmtree(job_dir, ignore_errors=True)
            return self.send_json({"error": "영상 파일이 없습니다"}, 400)
        name = (form.getfirst("name", "") or "").strip()[:60]
        spec = dict(mode=mode, videos=videos, flip_back=form.getfirst("flip_back", "") == "1",
                    can_csv=save("can", "can"), name=name or jid)
        job = core.Job(jid)
        JOBS[jid] = job
        threading.Thread(target=run_job, args=(job, job_dir, spec), daemon=True).start()
        return self.send_json({"job": jid})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/demo.yaml")
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()
    CFG.update(core.load_cfg(args.config))
    for k in ("out_dir", "thumb_dir"):
        os.makedirs(CFG[k], exist_ok=True)
    host, port = CFG["host"], args.port or int(CFG["port"])
    if not core.load_thresholds(CFG):
        print("⚠ results/demo/thresholds.json이 없습니다 — 먼저 python tools/demo/export.py --thresholds", file=sys.stderr)
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"테스트 프로그램: http://{host}:{port}  ·  전시 모드: http://{host}:{port}/exhibit  (Ctrl+C로 종료)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
