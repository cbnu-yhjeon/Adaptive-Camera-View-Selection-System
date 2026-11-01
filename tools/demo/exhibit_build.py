"""전시 모드 준비 — 하이라이트 장면 자동 선정 + 수치 모음 + 표시용 축소본 미리 만들기.

    .venv/bin/python tools/demo/exhibit_build.py      # → results/demo/exhibit.json
    .venv/bin/python tools/demo/server.py             # → http://127.0.0.1:8765/exhibit

먼저 `tools/demo/export.py --source split_test`로 test 분할 결과가 있어야 하고,
수치는 `scripts/colloquium/make_assets.py`가 만든 results/colloquium/numbers.json에서만 읽는다(원칙 4).

장면 선정 (configs/demo.yaml `exhibit`): test 분할(학습에 쓰지 않은 장면)에서 사건 구간을 찾는다.
  좌·우회전 / 좌·우 차선변경  라벨 기동 구간. 기동 쪽 측방 채널을 강조한 keyframe 비율로 점수
  측방 위험                   진행 방향이 아닌 채널에 '실제 위험'(B.4-(2))이 있는 구간. 포착 비율로 점수
  강조 불필요                 위험·기동 없이 보조 뷰 0장이 이어지는 구간 — 필요 없을 땐 끈다
점수가 `min_hit` 이상이고 `clip_min_s` 이상인 구간만 쓴다. 차선변경은 test에 22 keyframe뿐이고 모델의
측방 강조 일치가 0~50%라 기준을 넘지 못한다 — 억지로 넣지 않는다(약점은 연구용 화면에서 그대로 볼 수 있다).
점수 기준으로 고른다 — 전시용으로 고른 장면이라는 점은 화면에도 적는다.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))

import core  # noqa: E402
from core import CAMERA_INDEX  # noqa: E402

FL, FR = CAMERA_INDEX["CAM_FRONT_LEFT"], CAMERA_INDEX["CAM_FRONT_RIGHT"]
BL, BR = CAMERA_INDEX["CAM_BACK_LEFT"], CAMERA_INDEX["CAM_BACK_RIGHT"]

# 기동 → 그 기동에서 강조돼야 할 측방 채널 (constants의 m(c) 최댓값 채널 + 차선변경은 전측방도 인정)
KINDS = {
    "TURN_LEFT": dict(kind="turn_left", title="좌회전", want={FL}),
    "TURN_RIGHT": dict(kind="turn_right", title="우회전", want={FR}),
    "LANE_CHANGE_LEFT": dict(kind="lane_left", title="왼쪽 차선 변경", want={BL, FL}),
    "LANE_CHANGE_RIGHT": dict(kind="lane_right", title="오른쪽 차선 변경", want={BR, FR}),
}
ORDER = ["turn_left", "hazard", "lane_right", "quiet", "turn_right", "lane_left"]


def runs(flags: list[bool]) -> list[tuple[int, int]]:
    """True가 이어지는 [시작, 끝] keyframe 구간 목록."""
    out, s = [], None
    for i, f in enumerate(flags + [False]):
        if f and s is None:
            s = i
        elif not f and s is not None:
            out.append((s, i - 1))
            s = None
    return out


def side_hazard(k) -> list[int]:
    return [c for c in k["hazard"] if c != k["primary"]]


def candidates(scene: dict, file: str, ex: dict, method: str) -> list[dict]:
    kfs, hz = scene["keyframes"], float(scene["key_hz"])
    n = len(kfs)
    lead, tail, cap = (int(round(ex[k] * hz)) for k in ("lead_s", "tail_s", "clip_max_s"))

    def clip(kind, title, a, b, hit, event):
        k0, k1 = max(0, a - lead), min(n - 1, b + tail)
        if k1 - k0 + 1 > cap:                      # 너무 길면 사건 시작 쪽을 남긴다 (선제 전환이 보이도록)
            k1 = k0 + cap - 1
        return dict(kind=kind, title=title, scene=scene["scene"], file=file, location=scene["location"],
                    description=scene.get("description", ""), k0=k0, k1=k1,
                    t0=kfs[k0]["t"], t1=kfs[k1]["t"] + 1.0 / hz, hit=round(hit, 3),
                    event=[a, b], n_event=b - a + 1)

    out = []
    for man, spec in KINDS.items():
        for a, b in runs([k["label"]["maneuver"] == man for k in kfs]):
            if b - a + 1 < 3:
                continue
            hit = sum(bool(spec["want"] & set(kfs[i]["m"][method]["sel"])) for i in range(a, b + 1)) / (b - a + 1)
            out.append(clip(spec["kind"], spec["title"], a, b, hit, None))
    for a, b in runs([bool(side_hazard(k)) for k in kfs]):
        if b - a + 1 < 2:
            continue
        hits = [all(c in kfs[i]["m"][method]["sel"] for c in side_hazard(kfs[i])) for i in range(a, b + 1)]
        out.append(clip("hazard", "뒤·옆에서 다가오는 물체", a, b, sum(hits) / len(hits), None))
    quiet = [not k["hazard"] and k["label"]["maneuver"] == "STRAIGHT" and not k["m"][method]["sel"]
             and (k["can"] or {}).get("speed", 0) > 3.0 for k in kfs]
    for a, b in runs(quiet):
        if (b - a + 1) / hz >= ex["quiet_min_s"]:
            out.append(clip("quiet", "강조할 것이 없을 때", a, b, 1.0, None))
    return [c for c in out if c["hit"] >= ex["min_hit"] and c["t1"] - c["t0"] >= ex["clip_min_s"]]


def pick(cands: list[dict], per_kind: int) -> list[dict]:
    """종류별로 점수·사건 길이 순 상위 per_kind개, 장면 중복 없이. 지역이 겹치지 않게 번갈아 고른다."""
    chosen, used = [], set()
    by = {k: sorted([c for c in cands if c["kind"] == k], key=lambda c: (-c["hit"], -c["n_event"], c["scene"]))
          for k in ORDER}
    for rnd in range(per_kind):
        for kind in ORDER:
            locs = {c["location"] for c in chosen if c["kind"] == kind}
            pool = [c for c in by[kind] if c["scene"] not in used]
            pool.sort(key=lambda c: c["location"] in locs)     # 안정 정렬 — 새 지역 우선
            if pool:
                chosen.append(pool[0])
                used.add(pool[0]["scene"])
    for i, c in enumerate(chosen):
        c["id"] = f"{i + 1:02d}_{c['kind']}"
    return chosen


def numbers(path: str) -> dict:
    """전시 카드에 쓰는 수치 — numbers.json 그대로 (소수 반올림만)."""
    with open(path, encoding="utf-8") as f:
        n = json.load(f)
    rows = {r["key"]: r for r in n["table1"]["rows"]}
    lab, rob = n["label"], n["robust"]
    return dict(
        source=os.path.relpath(path, core.ROOT),
        n_keyframes=lab["n_keyframes"], n_test=lab["n_split"]["test"], hazard_pct=lab["hazard_pct"],
        n_gold=n["table1"]["n_labeled"],
        model=dict(mean_aux=round(rows["model"]["mean_aux"], 2), recall=round(rows["model"]["recall"], 1),
                   precision=round(rows["model"]["precision"], 1), top1=round(100 * rows["model"]["top1"], 1)),
        fixed=dict(recall=round(rows["b1"]["recall"], 1)),
        # 영상 단일과의 차이는 같은 표시량에서 잰다 (논문·포스터와 같은 값, B.8)
        img_only=dict(recall_at_load=round(rows["b3"]["recall_at_load"], 1),
                      gain_at_load=round(rows["model"]["recall_at_load"] - rows["b3"]["recall_at_load"], 1)),
        rule=dict(top1=round(100 * rows["b2c"]["top1"], 1)),
        claim3=dict(gold_diff=[round(x, 1) for x in rob["claim3"]["gold_diff"]],
                    recall_diff=[round(x, 1) for x in rob["claim3"]["recall_diff"]]),
        params_m=round(n["setup"]["params_m"], 2), seq_s=n["setup"]["seq_len"] * 0.5,
    )


def warm_thumbs(cfg: dict, clips: list[dict]) -> int:
    """재생할 프레임의 표시용 축소본을 미리 만든다 — 전시 중 첫 재생이 끊기지 않게."""
    import server
    server.CFG.update(cfg)
    rels = []
    for c in clips:
        with open(os.path.join(cfg["out_dir"], c["file"][len("data/"):]), encoding="utf-8") as f:
            frames = json.load(f)["frames"]
        for fr in frames:
            if c["t0"] - 0.1 <= fr["t"] <= c["t1"] + 0.1:
                rels += [u[len("img/"):] for u in fr["img"] if u]
    with ThreadPoolExecutor(8) as pool:
        ok = sum(p is not None for p in pool.map(server.thumb, rels))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/demo.yaml")
    ap.add_argument("--no_thumbs", action="store_true")
    args = ap.parse_args()
    cfg = core.load_cfg(args.config)
    ex = cfg["exhibit"]
    src_dir = os.path.join(cfg["out_dir"], ex["source"])
    files = sorted(glob.glob(os.path.join(src_dir, "*.json")))
    if not files:
        print(f"{src_dir}가 비어 있습니다 — 먼저 tools/demo/export.py --source {ex['source']}", file=sys.stderr)
        return 1
    cands = []
    for p in files:
        with open(p, encoding="utf-8") as f:
            s = json.load(f)
        cands += candidates(s, f"data/{ex['source']}/{os.path.basename(p)}", ex, ex["method"])
    clips = pick(cands, int(ex["per_kind"]))
    out = dict(
        generated=time.strftime("%Y-%m-%d %H:%M"), source=ex["source"], method=ex["method"],
        n_scenes=len(files), n_candidates=len(cands),
        cfg={k: ex[k] for k in ("idle_resume_s", "card_s", "playback", "min_hit")},
        numbers=numbers(os.path.join(core.ROOT, "results", "colloquium", "numbers.json")),
        clips=clips,
    )
    dst = os.path.join(cfg["out_dir"], "exhibit.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"후보 {len(cands)}개 (scene {len(files)}개) → 장면 {len(clips)}개")
    for c in clips:
        print(f"  {c['id']:<16} {c['scene']}  {c['t0']:5.1f}~{c['t1']:5.1f}초  강조 일치 {c['hit']:.0%}  {c['location']}")
    if not args.no_thumbs:
        print(f"축소본 {warm_thumbs(cfg, clips)}장 준비")
    print(f"→ {os.path.relpath(dst, core.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
