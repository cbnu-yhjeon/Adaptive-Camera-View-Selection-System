"""포스터·논문용 화면 캡처 — WSL에서 Windows Edge 헤드리스로 찍는다.

  fig/pipeline_poster.png   diagrams.html?d=poster (가로 파이프라인)
  fig/pipeline_paper.png    diagrams.html?d=paper  (세로, 논문 단 폭)
  fig/ui_goldset.png        골드셋 검수 화면 — 실제 판정(data/goldset_v1.json)을 그대로 띄운 장면
  fig/ui_review.png         라벨 표적 검수 2차 페이지의 첫 카드

사용자의 브라우저 저장소를 건드리지 않도록 임시 폴더 사본과 임시 프로필을 쓴다.

사용:
    python scripts/colloquium/shoot_ui.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG = os.path.join(ROOT, "results", "colloquium", "fig")
EDGE = "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
TMP = "/mnt/c/Users/USER/AppData/Local/Temp/acvs_shots"
UI = os.path.join(ROOT, "tools", "goldset_ui")
GOLD_ITEM = "b72f2f5db001"       # scene-0066: 좌측 지시등, 정답 좌후·좌전 (연구자 판정)


def winpath(p: str) -> str:
    return subprocess.check_output(["wslpath", "-w", p], text=True).strip()


def shoot(html: str, out_png: str, w: int, h: int, scale: float = 2.0, query: str = "") -> str:
    url = "file:///" + winpath(html).replace("\\", "/") + query
    shot = os.path.join(TMP, os.path.basename(out_png))
    subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    f"--user-data-dir={winpath(TMP)}\\profile", f"--window-size={w},{h}",
                    f"--force-device-scale-factor={scale}", "--virtual-time-budget=6000",
                    f"--screenshot={winpath(TMP)}\\{os.path.basename(out_png)}", url],
                   check=True, capture_output=True)
    return shot


def goldset_copy() -> str:
    """판정을 미리 넣은 골드셋 UI 사본 (대상 장면 이미지만 복사)."""
    dst = os.path.join(TMP, "goldset")
    os.makedirs(os.path.join(dst, "img"), exist_ok=True)
    s = open(os.path.join(UI, "manifest.js"), encoding="utf-8").read()
    j = json.loads(s[s.index("{"):s.rindex("}") + 1])
    toks = [it["token"] for it in j["items"]]
    gold = json.load(open(os.path.join(ROOT, "data", "goldset_v1.json"), encoding="utf-8"))
    g, meta = gold["gold"], gold.get("item_meta", {})
    target = next(i for i, t in enumerate(toks) if t.startswith(GOLD_ITEM))
    it = j["items"][target]
    files = {it["route"]} | {f for fr in it["frames"] for f in fr["img"].values()}
    for f in files:
        shutil.copy(os.path.join(UI, f), os.path.join(dst, f))
    shutil.copy(os.path.join(UI, "manifest.js"), os.path.join(dst, "manifest.js"))
    key = "goldset_%s_%s_%s_%d" % (j["meta"]["version"], j["meta"]["split"], j["meta"]["seed"], len(toks))
    seed = {t: {"sel": v, "done": True, "none": len(v) == 0, "hmm": bool(meta.get(t, {}).get("hmm")),
                "note": meta.get(t, {}).get("note", "")} for t, v in g.items()}
    first = next(i for i in range(len(toks)) if toks[i] not in g)
    html = open(os.path.join(UI, "index.html"), encoding="utf-8").read()
    # 판정과 함께 '파일 백업 완료' 기록도 넣는다 — 없으면 화면 위에 백업 알림 띠가 찍힌다
    pre = ("<script>try{localStorage.setItem(%s,%s);localStorage.setItem(%s,%s)}catch(e){}</script>\n"
           % (json.dumps(key), json.dumps(json.dumps(seed)), json.dumps(key + "_exported"), json.dumps(str(len(g)))))
    html = html.replace('<script src="manifest.js"></script>', '<script src="manifest.js"></script>\n' + pre, 1)
    step = "ArrowRight" if target >= first else "ArrowLeft"
    # 미판정 장면에서는 →를 두 번 눌러야 넘어가므로(건너뛰기 경고) 대상 장면에 닿을 때까지 누른다
    nav = ('<script>(function(){const want="%d / ";for(let i=0;i<%d&&!document.getElementById("meta").textContent'
           '.startsWith(want);i++)document.dispatchEvent(new KeyboardEvent("keydown",{key:"%s"}));})();</script>\n'
           % (target + 1, 2 * abs(target - first) + 4, step))
    html = html.replace("</body>", nav + "</body>")
    out = os.path.join(dst, "index.html")
    open(out, "w", encoding="utf-8").write(html)
    return out


def main() -> int:
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(TMP, exist_ok=True)
    dg = os.path.join(TMP, "diagrams.html")
    shutil.copy(os.path.join(os.path.dirname(__file__), "diagrams.html"), dg)
    for name, (w, h, sc) in {"poster": (1200, 500, 3), "paper": (440, 540, 5)}.items():
        shot = shoot(dg, f"pipeline_{name}.png", w, h, sc, f"?d={name}")
        Image.open(shot).convert("RGB").save(os.path.join(FIG, f"pipeline_{name}.png"))

    shot = shoot(goldset_copy(), "ui_goldset_full.png", 1600, 1400, 1.5)
    im = Image.open(shot).convert("RGB")
    s = im.width / 1600
    # 시각 칸(프레임별 속도·지시등) 바로 아래까지 — 그 아래는 안내문이라 잘린 문장이 남는다
    im.crop((int(56 * s), int(10 * s), int(1544 * s), int(694 * s))).save(os.path.join(FIG, "ui_goldset.png"))

    rv = os.path.join(TMP, "review")
    os.makedirs(rv, exist_ok=True)
    shutil.copy(os.path.join(ROOT, "results", "label_review", "round2.html"), os.path.join(rv, "round2.html"))
    shot = shoot(os.path.join(rv, "round2.html"), "ui_review_full.png", 1500, 2300, 1.5)
    im = Image.open(shot).convert("RGB")
    s = im.width / 1500
    im.crop((int(150 * s), int(1142 * s), int(1350 * s), int(1920 * s))).save(os.path.join(FIG, "ui_review.png"))
    print("✅", FIG)
    return 0


if __name__ == "__main__":
    sys.exit(main())
