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
  <ui_dir>/img/                     6채널 현재 프레임 + 과거·미래 프레임 축소 이미지 + 경로도

검수자에게 주는 정보 — 운전자가 아는 것만, 라벨은 주지 않는다
  원격 운전자는 자기가 어디로 갈지 알고 운전한다. 현재·과거 프레임만 보여주면 회전·차선변경
  직전 장면에서 무엇을 봐야 할지 판단할 근거가 없다(1차 골드셋 시도에서 검수자가 지적).
  그래서 운전자의 의도에 해당하는 원자료를 함께 보여준다:
    · 미래 프레임(+1~+5초) — 실제로 어디로 갔는가
    · 경로도 — 차량 기준 위에서 본 과거 1초·미래 5초 경로 + 차로·교차로
    · 현재 속도와 방향지시등
  우리 기동 라벨(TURN_LEFT 등)은 보여주지 않는다. 라벨을 보여주면 골드셋이 의사 라벨을
  따라가 평가가 순환한다(원칙 1). 원자료를 보고 판단하는 것은 사람의 몫이다.

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
from src.utils.plotting import use_korean_font  # noqa: E402

GRID = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
NON_STRAIGHT = ("TURN_LEFT", "TURN_RIGHT", "LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT", "REVERSE")


def export_image(src: str, dst: str, width: int, quality: int = 80) -> None:
    if os.path.exists(dst):
        return
    im = Image.open(src).convert("RGB")
    im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    im.save(dst, "JPEG", quality=quality, optimize=True)


def ego_frame(points, origin, yaw):
    """global (x,y) → 차량 기준 (오른쪽 +x, 앞 +y). 그림에서 진행 방향이 위가 되게 한다."""
    import numpy as np
    d = np.asarray(points, dtype=float) - np.asarray(origin, dtype=float)
    c, s = np.cos(yaw), np.sin(yaw)
    fwd = d[:, 0] * c + d[:, 1] * s
    left = -d[:, 0] * s + d[:, 1] * c
    return np.stack([-left, fwd], axis=1)


def draw_route_map(w, scene_token, token, path):
    """경로도 — 차량 기준 과거 1초·미래 경로와 주변 차로·교차로·주차장."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from shapely.geometry import box

    toks = w.sample_tokens_in_scene(scene_token)
    t0 = w.sample_timestamp(token)
    ego = w.get_ego_pose(token)["translation"][:2]
    yaw = w.get_ego_yaw(token)
    hist = [t for t in toks if -1.0 <= w.sample_timestamp(t) - t0 <= 5.0]
    pts = ego_frame([w.get_ego_pose(t)["translation"][:2] for t in hist], ego, yaw)
    dts = [w.sample_timestamp(t) - t0 for t in hist]
    # 축척은 경로 길이에 맞춘다 — 저속이면 5초 경로가 10 m 남짓이라 고정 축척으론 점이 된다
    import numpy as np
    extent = float(np.abs(pts).max()) if len(pts) else 0.0
    radius = float(np.clip(extent * 1.35 + 8.0, 15.0, 60.0))

    idx = w.map_index(scene_token)
    use_korean_font()      # nuScenes 맵 모듈을 처음 불러올 때 matplotlib 글꼴 설정이 덮어써진다
    fig, ax = plt.subplots(figsize=(3.6, 4.4), dpi=110)
    if idx is not None:
        area = box(ego[0] - radius * 1.5, ego[1] - radius * 1.5, ego[0] + radius * 1.5, ego[1] + radius * 1.5)
        style = {"carpark_area": ("#cfe3f5", None), "lane": ("#e6e9ec", "#c3c9ce"),
                 "lane_connector": ("#eef0f2", "#d5d9dd"), "road_segment": (None, None)}
        for layer in ("carpark_area", "lane", "lane_connector"):
            tree, polys, ltoks = idx._trees[layer]
            for i in tree.query(area):
                poly = polys[i]
                geoms = getattr(poly, "geoms", [poly])
                for g in geoms:
                    xy = ego_frame(list(g.exterior.coords), ego, yaw)
                    fc, ec = style[layer]
                    ax.add_patch(MplPolygon(xy, closed=True, facecolor=fc,
                                            edgecolor=ec or "none", linewidth=0.5))
        tree, polys, ltoks = idx._trees["road_segment"]
        for i in tree.query(area):
            if ltoks[i] in idx.intersections:
                for g in getattr(polys[i], "geoms", [polys[i]]):
                    xy = ego_frame(list(g.exterior.coords), ego, yaw)
                    ax.add_patch(MplPolygon(xy, closed=True, facecolor="#f6e7b8", alpha=0.55,
                                            edgecolor="none"))
    past = [p for p, d in zip(pts, dts) if d <= 0]
    fut = [p for p, d in zip(pts, dts) if d >= 0]
    if len(past) > 1:
        ax.plot(*zip(*past), color="#8a939a", linewidth=2.5)
    if len(fut) > 1:
        ax.plot(*zip(*fut), color="#0d6b73", linewidth=3)
    for p, d in zip(pts, dts):
        if d > 0 and abs(d - round(d)) < 0.26:
            ax.plot(*p, "o", color="#0d6b73", markersize=4)
            ax.annotate(f"+{round(d):.0f}s", p, textcoords="offset points", xytext=(5, -3),
                        fontsize=7, color="#0d6b73")
    ax.add_patch(MplPolygon([(-0.95, -2.3), (0.95, -2.3), (0.95, 1.6), (0, 2.4), (-0.95, 1.6)],
                            closed=True, facecolor="#b8422a", edgecolor="none"))
    ax.set_xlim(-radius * 0.6, radius * 0.6)
    ax.set_ylim(-radius * 0.3, radius)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#d8dee2")
    ax.set_title("경로 (위 = 현재 진행 방향)", fontsize=9)
    fig.tight_layout(pad=0.4)
    fig.savefig(path)
    plt.close(fig)


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
    ap.add_argument("--future_s", type=float, default=5.0, help="함께 보여줄 미래 구간(초)")
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

    # ---- 이미지 내보내기: 과거 · 현재 · 미래 프레임 6채널 + 경로도 ----
    img_dir = os.path.join(args.ui_dir, "img")
    os.makedirs(img_dir, exist_ok=True)
    use_korean_font()
    for i, r in enumerate(rows):
        toks = w.sample_tokens_in_scene(r["scene_token"])
        pos = toks.index(r["token"])
        past = [toks[pos - k] for k in range(args.past, 0, -1) if pos - k >= 0]
        # 미래는 1초 간격으로 (keyframe 2 Hz 중 정수 초에 가까운 것)
        future, want = [], 1.0
        for t in toks[pos + 1:]:
            d = w.sample_timestamp(t) - r["timestamp"]
            if d > args.future_s + 0.3:
                break
            if d >= want - 0.26:
                future.append(t)
                want += 1.0
        frames = past + [r["token"]] + future
        r["now_index"] = len(past)
        route = f"{r['token'][:12]}_route.png"
        draw_route_map(w, r["scene_token"], r["token"], os.path.join(img_dir, route))
        r["route"] = f"img/{route}"
        vm = w._latest_can(r["scene_token"], "vehicle_monitor",
                           int(w.nusc.get("sample", r["token"])["timestamp"]))
        r["signal"] = (None if vm is None else
                       ("좌" if vm.get("left_signal") and not vm.get("right_signal") else
                        "우" if vm.get("right_signal") and not vm.get("left_signal") else
                        "비상" if vm.get("left_signal") and vm.get("right_signal") else "꺼짐"))
        r["speed_kmh"] = round(3.6 * float(np.linalg.norm(w.get_ego_velocity(r["token"]))), 1)
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
