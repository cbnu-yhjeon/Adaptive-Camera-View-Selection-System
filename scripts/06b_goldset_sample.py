"""골드셋 후보 추출 + 검수 이미지 내보내기 (CLAUDE.md B.8 골드셋 절차, M6).

test 분할에서 층화 추출한다. 층은 서로 겹치지 않게 나눈다 — 그래야 층마다 추출 확률이
하나로 정해져 역확률 가중으로 전체 분포 기준 추정치를 복원할 수 있다.

  S1 위험       '실제 위험'(is_hazard) 채널이 하나라도 있는 프레임
  S2 비직진 기동 S1이 아니고 기동이 TURN·LANE_CHANGE·REVERSE인 프레임
  S3 나머지     S1·S2가 아닌 프레임 (직진·정지)

층마다 n장을 단순 무작위로 뽑는다. 가중치 = 층 크기 / n.
명세는 세 번째 층을 '무작위 100'으로 적었지만, 전체에서 무작위로 뽑으면 S1·S2와 겹쳐
추출 확률이 프레임마다 달라진다. 겹치지 않는 '나머지' 층이 같은 목적(자연 분포의 대부분을
차지하는 평범한 장면)을 달성하면서 가중치를 정확하게 만든다.

출력
  data/goldset/candidates_v1.json   후보 목록(토큰·층·가중치·scene·진행 방향)
  <ui_dir>/manifest.js              UI가 읽는 목록 (file://에서는 fetch가 막혀 스크립트로 싣는다)
  <ui_dir>/img/                     6채널 현재 프레임 + 과거 2 keyframe 축소 이미지

사용:
    python scripts/06b_goldset_sample.py --per_stratum 100
    python scripts/06b_goldset_sample.py --per_stratum 2 --ui_dir /tmp/goldset_dry   # 드라이런
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.contrib_cache import ContribCache, cache_path  # noqa: E402
from src.data.nusc_wrapper import NuscWrapper  # noqa: E402
from src.data.splits import load_split  # noqa: E402
from src.labeling.label_function import LabelConfig, is_hazard  # noqa: E402
from src.utils.config import load_cfgs  # noqa: E402
from src.utils.constants import CAMERAS  # noqa: E402

GRID = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
NON_STRAIGHT = ("TURN_LEFT", "TURN_RIGHT", "LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT", "REVERSE")


def export_image(src: str, dst: str, width: int, quality: int = 80) -> None:
    if os.path.exists(dst):
        return
    im = Image.open(src).convert("RGB")
    im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    im.save(dst, "JPEG", quality=quality, optimize=True)


def main() -> int:
    base, lab, _ = load_cfgs("configs/train_proposed.yaml")
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=base["version"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--per_stratum", type=int, default=100)
    ap.add_argument("--seed", type=int, default=base.get("seed", 42))
    ap.add_argument("--out", default="data/goldset/candidates_v1.json")
    ap.add_argument("--ui_dir", default="tools/goldset_ui")
    ap.add_argument("--width", type=int, default=800, help="현재 프레임 이미지 폭(px)")
    ap.add_argument("--past", type=int, default=2, help="함께 보여줄 과거 keyframe 수")
    args = ap.parse_args()

    cfg = LabelConfig.from_dict(lab)
    w = NuscWrapper(base["dataroot"], args.version,
                    require_image_files=base.get("require_image_files", True))
    df = pd.read_parquet(f"data/labels_{args.version}.parquet")
    cc = ContribCache(cache_path(os.path.join(base["cache_dir"], args.version), cfg))
    names = set(load_split(base["splits_file"])[args.split])
    df = df[df["scene_token"].map(lambda t: w.scene_name(t) in names)].copy()
    df = df[df["sample_token"].isin(set(cc.tokens))].reset_index(drop=True)

    def hazard(tok):
        return any(is_hazard(c, cfg) and c["vis"] for c in cc.get(tok))
    df["haz"] = df["sample_token"].map(hazard)
    df["stratum"] = np.where(df["haz"], "S1_hazard",
                             np.where(df["maneuver"].isin(NON_STRAIGHT), "S2_maneuver", "S3_rest"))

    rng = np.random.default_rng(args.seed)
    rows = []
    sizes = df["stratum"].value_counts().to_dict()
    for s in ("S1_hazard", "S2_maneuver", "S3_rest"):
        pool = df[df["stratum"] == s]
        k = min(args.per_stratum, len(pool))
        pick = pool.iloc[rng.choice(len(pool), size=k, replace=False)]
        for r in pick.itertuples():
            rows.append(dict(token=r.sample_token, stratum=s, n_stratum=int(sizes[s]),
                             weight=float(sizes[s] / k), scene=w.scene_name(r.scene_token),
                             scene_token=r.scene_token, timestamp=float(r.timestamp),
                             primary=int(r.primary_view)))
    rng.shuffle(rows)                       # 층이 순서로 드러나지 않게 섞는다

    # ---- 이미지 내보내기: 현재 프레임 6채널 + 과거 keyframe ----
    img_dir = os.path.join(args.ui_dir, "img")
    os.makedirs(img_dir, exist_ok=True)
    for i, r in enumerate(rows):
        toks = w.sample_tokens_in_scene(r["scene_token"])
        pos = toks.index(r["token"])
        frames = [toks[pos - k] for k in range(args.past, 0, -1) if pos - k >= 0] + [r["token"]]
        r["frames"] = []
        for j, t in enumerate(frames):
            cur = (t == r["token"])
            paths = w.get_image_paths(t)
            entry = dict(token=t, dt=round(w.sample_timestamp(t) - r["timestamp"], 2), img={})
            for cam in GRID:
                name = f"{t[:12]}_{cam}{'' if cur else '_s'}.jpg"
                export_image(paths[cam], os.path.join(img_dir, name),
                             args.width if cur else args.width // 2)
                entry["img"][cam] = f"img/{name}"
            r["frames"].append(entry)
        if (i + 1) % 25 == 0 or i + 1 == len(rows):
            print(f"  이미지 {i + 1}/{len(rows)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    meta = dict(version=args.version, split=args.split, seed=args.seed,
                per_stratum=args.per_stratum, stratum_sizes=sizes,
                cameras=CAMERAS, n=len(rows))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(dict(meta=meta, items=[{k: v for k, v in r.items() if k != "frames"} for r in rows]),
                  f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.ui_dir, "manifest.js"), "w", encoding="utf-8") as f:
        f.write("window.GOLDSET = " + json.dumps(dict(meta=meta, items=rows), ensure_ascii=False) + ";\n")

    here = os.path.join(os.path.dirname(__file__), "..", "tools", "goldset_ui", "index.html")
    if os.path.abspath(args.ui_dir) != os.path.abspath(os.path.dirname(here)) and os.path.exists(here):
        shutil.copy(here, os.path.join(args.ui_dir, "index.html"))

    print(f"\n층 크기 {sizes}")
    print(f"후보 {len(rows)}장 → {args.out}")
    print(f"검수 UI → {os.path.join(args.ui_dir, 'index.html')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
