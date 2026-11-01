"""테스트 프로그램 데이터 내보내기 — nuScenes 장면마다 모델·규칙의 선택을 미리 계산해 JSON으로 쓴다.

소스 (configs/demo.yaml `sources`)
  split_test  우리 test 분할 127 scene. 평가(06_evaluate.py)와 **같은 Dataset 아이템**을 거치므로
              선택이 표 1과 같다. GT로 '실제 위험' 객체 박스·의사 라벨·골드 정답을 함께 싣는다.
  v1_test     공식 v1.0-test 150 scene — 학습·평가 어디에도 쓰지 않은 장면. 주석이 비공개라
              위험도 입력이 없다(위험도를 쓰지 않는 모델 A2·B3가 기본). 특징 캐시를 먼저 만든다.

임계값 τ는 평가와 같이 **val에서** 라벨 표시량에 맞춘다(--thresholds, 없으면 자동) →
results/demo/thresholds.json. 장면 JSON은 results/demo/{소스}/{scene}.json, 목록은 index.json.

재생 프레임: 그 scene의 12 Hz sweeps가 디스크에 있으면 그것으로, 없으면 keyframe(2 Hz)으로.
모델은 keyframe마다 한 번 판단하고, 사이 프레임은 직전 keyframe의 판단을 그대로 쓴다(과거만).

사용
    python tools/demo/export.py --thresholds
    python tools/demo/export.py --source split_test
    python tools/demo/export.py --source v1_test --limit 10
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import core  # noqa: E402
from core import ROOT, load_cfg, load_model, predict, primary_from_speed, r3, select  # noqa: E402

from src.utils.constants import CAMERAS, CAN_FEATURES, CAN_NORM, D_CAN, NUM_CAMERAS  # noqa: E402

KIND = (("human.pedestrian", "보행자"), ("vehicle.bicycle", "자전거"), ("vehicle.motorcycle", "오토바이"),
        ("vehicle.bus", "버스"), ("vehicle.truck", "트럭"), ("vehicle", "차량"))


def kind_of(cat: str) -> str:
    return next((v for k, v in KIND if cat.startswith(k)), cat.split(".")[-1])


# ---------------------------------------------------------------------
# 임계값
# ---------------------------------------------------------------------
def compute_thresholds(env, cfg, device) -> dict:
    """학습 모델마다 val에서 라벨 표시량에 맞춘 τ — 06_evaluate.evaluate()와 같은 절차."""
    ev = core.ev()
    aux_max = env.cfg.aux_max
    out = {}
    for m, mc in cfg["methods"].items():
        if "ckpt" not in mc:
            continue
        model = load_model(mc["ckpt"], device)
        ds = env.dataset("val", mc["risk_source"], mc.get("noise", "clean"), mc.get("seq_len"))
        Pv, _, mv = ev.run_model(model, ds, device)
        load = float((mv["aux"] > 0.5).sum(1).mean())
        tau = ev.threshold_for_load(Pv, load, mv["mask"] > 0.5, aux_max)
        sel = ev.select_from_probs(Pv, mv["primary"].astype(int), tau, aux_max)
        out[m] = dict(tau=round(float(tau), 3), ckpt=mc["ckpt"], risk_source=mc["risk_source"],
                      noise=mc.get("noise", "clean"), label_load=round(load, 4),
                      val_mean_aux=round(float(sel.sum(1).mean()), 4))
        print(f"  τ[{m}] = {tau:.2f}  (val 라벨 표시량 {load:.3f}, 모델 {sel.sum(1).mean():.3f})")
    path = os.path.join(cfg["out_dir"], "thresholds.json")
    os.makedirs(cfg["out_dir"], exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"✅ {os.path.relpath(path, ROOT)}")
    return out


# ---------------------------------------------------------------------
# 재생 프레임
# ---------------------------------------------------------------------
def play_frames(nusc, dataroot: str, toks: list[str]) -> tuple[list[dict], bool]:
    """(프레임 목록, sweeps 사용 여부). 각 프레임: {t, k(유효 keyframe), img[6]}.

    sweeps는 카메라마다 시각이 어긋나 있다 — CAM_FRONT를 기준 시계로 두고 다른 카메라는
    가장 가까운 프레임을 붙인다. keyframe 번호 k는 그 시각 **이전**의 마지막 keyframe(과거만).
    """
    samples = [nusc.get("sample", t) for t in toks]
    t0 = samples[0]["timestamp"]
    kf_ts = [nusc.get("sample_data", s["data"]["CAM_FRONT"])["timestamp"] for s in samples]
    chains = {}
    for cam in CAMERAS:
        sd = nusc.get("sample_data", samples[0]["data"][cam])
        end = nusc.get("sample_data", samples[-1]["data"][cam])["timestamp"]
        chain = []
        while sd is not None and sd["timestamp"] <= end:
            chain.append((sd["timestamp"], sd["filename"]))
            sd = nusc.get("sample_data", sd["next"]) if sd["next"] else None
        chains[cam] = chain
    have = all(os.path.exists(os.path.join(dataroot, fn)) for ch in chains.values() for _, fn in ch)
    if have and len(chains["CAM_FRONT"]) > len(toks):
        frames = []
        cts = {cam: [ts for ts, _ in ch] for cam, ch in chains.items()}
        for ts, _ in chains["CAM_FRONT"]:
            k = max(0, bisect.bisect_right(kf_ts, ts) - 1)
            img = []
            for cam in CAMERAS:
                a = cts[cam]
                j = bisect.bisect_left(a, ts)
                j = min(range(max(0, j - 1), min(len(a), j + 1)), key=lambda q: abs(a[q] - ts))
                img.append("img/" + chains[cam][j][1])
            frames.append(dict(t=round((ts - t0) / 1e6, 3), k=k, img=img))
        return frames, True
    frames = []
    for k, s in enumerate(samples):
        img = ["img/" + nusc.get("sample_data", s["data"][cam])["filename"] for cam in CAMERAS]
        frames.append(dict(t=round((s["timestamp"] - t0) / 1e6, 3), k=k, img=img))
    return frames, False


# ---------------------------------------------------------------------
# GT 객체 박스 (test 분할만)
# ---------------------------------------------------------------------
def box_rect(box, ego, cal):
    """3D 박스 → 카메라 화면의 2D 외접 사각형 (0~1 정규화). 화면 밖이면 None."""
    from nuscenes.utils.data_classes import Box
    from pyquaternion import Quaternion

    from src.utils.geometry import global_to_camera, project_to_image
    b = Box(box["global_center"], box["size"], Quaternion(box["rotation"]))
    pc = global_to_camera(b.corners().T, ego["translation"], ego["rotation"],
                          cal["translation"], cal["rotation"])
    front = pc[:, 2] > 0.1
    if not front.any():
        return None
    uv, _ = project_to_image(pc[front], cal["intrinsic"])
    W, H = cal["width"], cal["height"]
    x0, y0 = np.clip(uv.min(0), 0, [W, H])
    x1, y1 = np.clip(uv.max(0), 0, [W, H])
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return [round(x0 / W, 4), round(y0 / H, 4), round(x1 / W, 4), round(y1 / H, 4)]


def frame_boxes(env, tok) -> list:
    """'실제 위험' 객체와 위험 슬롯 임계값 이상을 기여한 객체의 박스.

    [카메라, x0, y0, x1, y1, 실제위험(0/1), 종류, TTC, 최접근거리] — 기여 캐시와 박스 목록은
    같은 순서다(channel_risk_contributors가 get_boxes를 그대로 돈다).
    """
    from src.labeling.label_function import is_hazard
    lc = env.cfg
    boxes = env.w.get_boxes(tok, lc.categories, lc.max_range)
    contribs = env.cc.get(tok)
    if len(boxes) != len(contribs):
        return []
    ego = env.w.get_ego_pose(tok)
    cals = {}
    out = []
    for b, c in zip(boxes, contribs):
        haz = is_hazard(c, lc)
        if not haz and c["w"] < lc.aux_threshold:
            continue
        for ci in c["vis"]:
            if ci not in cals:
                cals[ci] = env.w.get_camera_calib(tok, CAMERAS[ci])
            rect = box_rect(b, ego, cals[ci])
            if rect:
                ttc = c["ttc"] if np.isfinite(c["ttc"]) else None
                out.append([ci, *rect, int(haz), kind_of(b["category"]),
                            None if ttc is None else round(ttc, 1), round(c["dca"], 1)])
    return out


def can_shown(feat, mask) -> dict | None:
    if mask < 0.5:
        return None
    raw = {k: float(feat[i]) * CAN_NORM[k] for i, k in enumerate(CAN_FEATURES)}
    return dict(speed=round(raw["speed"], 2), steer=round(raw["steering"], 1),
                left=int(raw["left_signal"] > 0.5), right=int(raw["right_signal"] > 0.5))


# ---------------------------------------------------------------------
# 소스별 내보내기
# ---------------------------------------------------------------------
def write_scene(cfg, source: str, name: str, data: dict) -> str:
    d = os.path.join(cfg["out_dir"], source)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    return f"data/{source}/{name}.json"


def method_meta(cfg, src, th) -> dict:
    return {m: dict(name=cfg["methods"][m]["name"], tau=th.get(m, {}).get("tau"),
                    rule="rule" in cfg["methods"][m], risk=cfg["methods"][m]["risk_source"])
            for m in src["methods"]}


def export_split_test(cfg, env, th, device, only=None, limit=None) -> list[dict]:
    from src.baselines.rule_based import RuleBased
    ev = core.ev()
    src = cfg["sources"]["split_test"]
    lc = env.cfg
    dsets = {}

    def ds(rs, noise):
        if (rs, noise) not in dsets:
            dsets[(rs, noise)] = env.dataset(src["split"], rs, noise)
        return dsets[(rs, noise)]

    base = ds("noisy", "medium")
    scene_names = {st: env.w.scene_name(st) for st in base.scene_tokens}
    scenes = [st for st in base.scene_tokens if not only or scene_names[st] in only]
    if limit:
        scenes = scenes[:limit]
    keep = {t for st in scenes for t in base.scene_tokens[st]}

    res: dict[str, dict] = {}            # 방법 → 토큰 → {p, sel}
    est: dict[str, str] = {}
    for m in src["methods"]:
        mc = cfg["methods"][m]
        d = ds(mc["risk_source"], mc.get("noise", "clean"))
        if "rule" in mc:
            rule = RuleBased(lc, mc["rule"], wrapper=env.w)
            sub = _SubsetDS(d, scenes)
            pred = rule.run(sub)
            res[m] = {t: dict(p=r3(np.clip(p["score"], -1, 9)), sel=p["aux"]) for t, p in pred.items()}
            est.update(rule._causal)
        else:
            P, _, meta = ev.run_model(load_model(mc["ckpt"], device), _SubsetDS(d, scenes), device)
            sel = select(P, meta["primary"], th[m]["tau"], lc.aux_max)
            res[m] = {t: dict(p=r3(P[i]), sel=sel[i]) for i, t in enumerate(meta["token"]) if t in keep}
        print(f"  {m}: {len(res[m])} keyframe")

    gt = ds("gt", "clean")
    summaries = []
    for st in scenes:
        toks = base.scene_tokens[st]
        name = scene_names[st]
        kfs, n_haz, n_gold, mans = [], 0, 0, {}
        for i, t in enumerate(toks):
            lab = base.labels.loc[t]
            feat, mask = env.cc.can(t)
            haz = sorted(env.hazard(t))
            g = env.gold.get(t) if env.gold else None
            n_haz += bool(haz)
            n_gold += g is not None
            mv = str(lab["maneuver"])
            mans[mv] = mans.get(mv, 0) + 1
            kfs.append(dict(
                t=round(env.w.sample_timestamp(t) - env.w.sample_timestamp(toks[0]), 3), token=t,
                primary=int(lab["primary_view"]), can=can_shown(feat, mask),
                m={m: res[m][t] for m in src["methods"]},
                risk=dict(noisy=r3(base.frame_risk(t)), gt=r3(gt.frame_risk(t))),
                label=dict(aux=[c for c in range(NUM_CAMERAS) if lab[f"aux_{c}"] > 0.5], maneuver=mv),
                est=est.get(t), hazard=haz, boxes=frame_boxes(env, t),
                gold=(dict(set=sorted(g["set"]), stratum=g["stratum"]) if g else None)))
        frames, sweeps = play_frames(env.w.nusc, env.w.dataroot, toks)
        sc = env.w.nusc.get("scene", st)
        data = dict(source="split_test", scene=name, has_gt=True, location=env.w.scene_location(st),
                    description=sc["description"], methods=method_meta(cfg, src, th),
                    default=src["default"], key_hz=2.0, sweeps=sweeps, keyframes=kfs, frames=frames)
        file = write_scene(cfg, "split_test", name, data)
        summaries.append(dict(name=name, file=file, location=data["location"], desc=sc["description"],
                              n_kf=len(toks), dur=kfs[-1]["t"], n_haz=n_haz, n_gold=n_gold,
                              maneuvers=mans, sweeps=sweeps))
    return summaries


class _SubsetDS:
    """평가 Dataset에서 고른 scene만 — 아이템·프레임 위험은 원래 Dataset의 것을 그대로 쓴다
    (프레임 난수가 원래 인덱스로 정해지므로 표 1과 같은 손상 입력이 들어간다)."""

    def __init__(self, ds, scenes):
        self.ds = ds
        self.scene_tokens = {st: ds.scene_tokens[st] for st in scenes}
        self.labels = ds.labels
        keep = set(scenes)
        self.idx = [i for i, (st, _) in enumerate(ds.flat) if st in keep]
        self.flat = [ds.flat[i] for i in self.idx]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return self.ds[self.idx[i]]

    def frame_risk(self, *a, **k):
        return self.ds.frame_risk(*a, **k)


class _ZeroRiskDS:
    """위험도 입력이 없는 장면용 규칙 입력 — B2c(검출 없음)는 기동 슬롯만 남는다."""

    def __init__(self, scene_tokens):
        self.scene_tokens = scene_tokens

    def frame_risk(self, token, *a, **k):
        return np.zeros(NUM_CAMERAS, dtype=np.float32)


def export_v1_test(cfg, th, device, only=None, limit=None) -> list[dict]:
    from src.baselines.rule_based import RuleBased
    from src.data.feature_cache import FeatureCache, build_feature_cache
    from src.data.nusc_wrapper import NuscWrapper
    src = cfg["sources"]["v1_test"]
    lc = core.label_cfg(cfg)
    w = NuscWrapper(cfg["dataroot"], src["version"], require_image_files=True)
    scenes = [st for st in w.scene_tokens() if not only or w.scene_name(st) in only]
    if limit:
        scenes = scenes[:limit]
    print(f"  v1.0-test: 이미지가 있는 scene {len(w.scene_tokens())}개 중 {len(scenes)}개")
    cache_dir = os.path.join(ROOT, cfg["base"].get("cache_dir", "cache"), src["version"])
    build_feature_cache(w, cache_dir, img_size=cfg["base"].get("img_size", 224), device=device)
    fc = FeatureCache(cache_dir)

    rules = {m: RuleBased(lc, cfg["methods"][m]["rule"], wrapper=w)
             for m in src["methods"] if "rule" in cfg["methods"][m]}
    summaries = []
    for n_done, st in enumerate(scenes):
        toks = w.sample_tokens_in_scene(st)
        name = w.scene_name(st)
        img = np.stack([fc.get(t) for t in toks]).astype(np.float32)
        can = np.zeros((len(toks), D_CAN), np.float32)
        shown, speeds = [], []
        for i, t in enumerate(toks):
            feat, mask = w.get_can_features(t)
            can[i, :len(feat)] = feat
            can[i, -1] = mask
            shown.append(can_shown(feat, mask))
            # CAN이 없으면 속도는 ego_pose 차분의 크기라 부호가 없다 — 후진 판정 불가
            speeds.append(float(feat[0]) * CAN_NORM["speed"] if mask > 0.5 else None)
        primary = primary_from_speed(speeds)
        risk = np.zeros((len(toks), NUM_CAMERAS), np.float32)
        per = {}
        for m in src["methods"]:
            mc = cfg["methods"][m]
            if m in rules:
                pred = rules[m].run(_ZeroRiskDS({st: toks}))
                per[m] = [dict(p=r3(np.clip(pred[t]["score"], -1, 9)), sel=pred[t]["aux"]) for t in toks]
                continue
            P = predict(load_model(mc["ckpt"], device), img, can, risk, int(mc.get("seq_len", 4)), device)
            sel = select(P, primary, th[m]["tau"], lc.aux_max)
            per[m] = [dict(p=r3(P[i]), sel=sel[i]) for i in range(len(toks))]
        est = next(iter(rules.values()))._causal if rules else {}
        t0 = w.sample_timestamp(toks[0])
        kfs = [dict(t=round(w.sample_timestamp(t) - t0, 3), token=t, primary=int(primary[i]), can=shown[i],
                    m={m: per[m][i] for m in per}, est=est.get(t)) for i, t in enumerate(toks)]
        frames, sweeps = play_frames(w.nusc, w.dataroot, toks)
        sc = w.nusc.get("scene", st)
        data = dict(source="v1_test", scene=name, has_gt=False, location=w.scene_location(st),
                    description=sc["description"], methods=method_meta(cfg, src, th),
                    default=src["default"], key_hz=2.0, sweeps=sweeps, keyframes=kfs, frames=frames)
        file = write_scene(cfg, "v1_test", name, data)
        mans = {}
        for k in kfs:
            if k["est"]:
                mans[k["est"]] = mans.get(k["est"], 0) + 1
        summaries.append(dict(name=name, file=file, location=data["location"], desc=sc["description"],
                              n_kf=len(toks), dur=kfs[-1]["t"], n_haz=None, n_gold=0,
                              maneuvers=mans, sweeps=sweeps))
        if (n_done + 1) % 10 == 0:
            print(f"  {n_done + 1}/{len(scenes)}")
    return summaries


# ---------------------------------------------------------------------
# 주행 기록(log) 단위 — 긴 영상
# ---------------------------------------------------------------------
CUT_SLOT_S = 1.0     # 공개되지 않은 구간을 접어 넣는 재생 시간(초) — 화면에 '끊김'으로 표시한다


def log_groups(nusc, scene_tokens, max_gap_s: float) -> dict:
    """log → (연속 그룹 목록, 그룹 앞 끊김 초). 그룹 = 간격 max_gap_s 이하로 이어지는 scene들.

    nuScenes는 20초 scene만 공개한다. 같은 log에서 다음 scene 첫 keyframe이 이번 scene 마지막
    keyframe 뒤 1초 안에 오면 끊김 없는 주행이다(keyframe 하나를 빠뜨린 정도).
    """
    have = set(scene_tokens)
    by_log: dict[str, list] = {}
    for sc in nusc.scene:
        if sc["token"] in have:
            by_log.setdefault(sc["log_token"], []).append(
                (nusc.get("sample", sc["first_sample_token"])["timestamp"],
                 nusc.get("sample", sc["last_sample_token"])["timestamp"], sc["token"]))
    out = {}
    for log, ss in by_log.items():
        ss.sort()
        groups, cuts = [[ss[0][2]]], [None]
        for a, b in zip(ss[:-1], ss[1:]):
            gap = (b[0] - a[1]) / 1e6
            if gap <= max_gap_s:
                groups[-1].append(b[2])
            else:
                groups.append([b[2]])
                cuts.append(round(gap, 1))
        out[log] = (groups, cuts)
    return out


class _GroupDS:
    """규칙 입력 — 이어지는 scene들을 한 줄로 돈다(위험 EMA가 scene 경계에서 끊기지 않게)."""

    def __init__(self, scene_tokens, risk_fn):
        self.scene_tokens = scene_tokens
        self._risk = risk_fn

    def frame_risk(self, token, *a, **k):
        return self._risk(token)


def export_long(cfg, source, th, device, env=None, only=None, limit=None) -> list[dict]:
    """log 하나를 영상 하나로. 모델의 2초 창은 연속 그룹 안에서 scene 경계를 넘어 이어지고,
    끊김(공개되지 않은 구간) 뒤에서는 새로 시작한다(장면 시작과 같은 패딩).

    trainval: 검출 오류는 각 프레임이 속한 분할의 평가 Dataset을 거친다 — test 프레임은 표 1과 같은
    손상 입력이다. 학습 장면이 섞여 있으므로 구간마다 분할을 싣고 요약은 eval_split 구간만 센다.
    """
    from src.baselines.rule_based import RuleBased
    src = cfg["sources"][source]
    gt_mode = env is not None
    if gt_mode:
        w, fc, lc = env.w, env.fc, env.cfg
        can_of = env.cc.can
        split_of = {n: s for s, names in env.split.items() for n in names}
        labels = env.df.set_index("sample_token")
        dsets: dict = {}

        def frisk(tok, rs, noise):
            if rs == "none":
                return np.zeros(NUM_CAMERAS, np.float32)
            sp = split_of[w.scene_name(w.sample_scene_token(tok))]
            if (sp, rs, noise) not in dsets:
                dsets[(sp, rs, noise)] = env.dataset(sp, rs, noise)
            return dsets[(sp, rs, noise)].frame_risk(tok)
    else:
        from src.data.feature_cache import FeatureCache, build_feature_cache
        from src.data.nusc_wrapper import NuscWrapper
        lc = core.label_cfg(cfg)
        w = NuscWrapper(cfg["dataroot"], src["version"], require_image_files=True)
        cache_dir = os.path.join(ROOT, cfg["base"].get("cache_dir", "cache"), src["version"])
        build_feature_cache(w, cache_dir, img_size=cfg["base"].get("img_size", 224), device=device)
        fc = FeatureCache(cache_dir)
        can_of = w.get_can_features
        split_of = {}

        def frisk(tok, rs, noise):
            return np.zeros(NUM_CAMERAS, np.float32)

    nusc = w.nusc
    logs = log_groups(nusc, w.scene_tokens(), float(src.get("max_gap_s", 1.0)))
    order = sorted(logs, key=lambda lg: nusc.get("log", lg)["logfile"])
    if only:
        order = [lg for lg in order if nusc.get("log", lg)["logfile"] in only]
    if limit:
        order = order[:limit]
    rules = {m: RuleBased(lc, cfg["methods"][m]["rule"], wrapper=w)
             for m in src["methods"] if "rule" in cfg["methods"][m]}
    summaries = []
    for n_done, lg in enumerate(order):
        groups, cuts = logs[lg]
        logrec = nusc.get("log", lg)
        name = logrec["logfile"]
        kfs, segs, frames = [], [], []
        t_off, sweeps_all, longest = 0.0, True, 0.0
        log_t0 = nusc.get("sample", nusc.get("scene", groups[0][0])["first_sample_token"])["timestamp"]
        for gi, group in enumerate(groups):
            gtoks = [t for st in group for t in w.sample_tokens_in_scene(st)]
            g_t0 = nusc.get("sample", gtoks[0])["timestamp"]
            if gi > 0:
                t_off = kfs[-1]["t"] + CUT_SLOT_S
            # 입력 (그룹 전체를 한 줄로)
            img = np.stack([fc.get(t) for t in gtoks]).astype(np.float32)
            can = np.zeros((len(gtoks), D_CAN), np.float32)
            shown, speeds = [], []
            for i, t in enumerate(gtoks):
                feat, mask = can_of(t)
                can[i, :len(feat)] = feat
                can[i, -1] = mask
                shown.append(can_shown(feat, mask))
                speeds.append(float(feat[0]) * CAN_NORM["speed"] if mask > 0.5 else None)
            if gt_mode:
                primary = [int(labels.loc[t, "primary_view"]) for t in gtoks]
            else:
                primary = primary_from_speed(speeds)
            per = {}
            for m in src["methods"]:
                mc = cfg["methods"][m]
                rs, noise = mc["risk_source"], mc.get("noise", "clean")
                if m in rules:
                    pred = rules[m].run(_GroupDS({f"{name}:{gi}": gtoks}, lambda t, rs=rs, noise=noise: frisk(t, rs, noise)))
                    per[m] = [dict(p=r3(np.clip(pred[t]["score"], -1, 9)), sel=pred[t]["aux"]) for t in gtoks]
                    continue
                risk = np.stack([frisk(t, rs, noise) for t in gtoks]).astype(np.float32)
                P = predict(load_model(mc["ckpt"], device), img, can, risk, int(mc.get("seq_len", 4)), device)
                sel = select(P, primary, th[m]["tau"], lc.aux_max)
                per[m] = [dict(p=r3(P[i]), sel=sel[i]) for i in range(len(gtoks))]
            est = {}
            for r in rules.values():
                est.update(r._causal)
            # 구간(scene)·재생 프레임
            seg_ids = []
            for si, st in enumerate(group):
                stoks = w.sample_tokens_in_scene(st)
                k0 = len(kfs) + sum(len(w.sample_tokens_in_scene(x)) for x in group[:si])
                sname = w.scene_name(st)
                s_t0 = t_off + (nusc.get("sample", stoks[0])["timestamp"] - g_t0) / 1e6
                seg_ids += [len(segs)] * len(stoks)
                segs.append(dict(scene=sname, split=split_of.get(sname, ""),
                                 k0=k0, k1=k0 + len(stoks), t0=round(s_t0, 3),
                                 cut=(cuts[gi] if si == 0 and gi > 0 else None)))
                fr, sw = play_frames(nusc, w.dataroot, stoks)
                sweeps_all &= sw
                frames += [dict(t=round(s_t0 + f["t"], 3), k=k0 + f["k"], img=f["img"]) for f in fr]
            for i, t in enumerate(gtoks):
                ts = nusc.get("sample", t)["timestamp"]
                rec = dict(t=round(t_off + (ts - g_t0) / 1e6, 3), real=round((ts - log_t0) / 1e6, 1),
                           seg=seg_ids[i], token=t, primary=int(primary[i]), can=shown[i],
                           m={m: per[m][i] for m in per}, est=est.get(t))
                if gt_mode:
                    lab = labels.loc[t]
                    g = env.gold.get(t) if env.gold else None
                    rec.update(risk=dict(noisy=r3(frisk(t, "noisy", "medium")), gt=r3(frisk(t, "gt", "clean"))),
                               label=dict(aux=[c for c in range(NUM_CAMERAS) if lab[f"aux_{c}"] > 0.5],
                                          maneuver=str(lab["maneuver"])),
                               hazard=sorted(env.hazard(t)), boxes=frame_boxes(env, t),
                               gold=(dict(set=sorted(g["set"]), stratum=g["stratum"]) if g else None))
                kfs.append(rec)
            longest = max(longest, (nusc.get("sample", gtoks[-1])["timestamp"] - g_t0) / 1e6)
        frames.sort(key=lambda f: f["t"])
        mans = {}
        for k in kfs:
            mv = k["label"]["maneuver"] if gt_mode else k["est"]
            if mv:
                mans[mv] = mans.get(mv, 0) + 1
        sc0 = nusc.get("scene", groups[0][0])
        loc = logrec["location"]
        eval_split = src.get("eval_split")
        data = dict(source=source, scene=name, has_gt=gt_mode, location=loc,
                    description=f"{logrec['date_captured']} · scene {len(segs)}개",
                    methods=method_meta(cfg, src, th), default=src["default"], key_hz=2.0,
                    sweeps=sweeps_all, long=True, eval_split=eval_split, segments=segs,
                    keyframes=kfs, frames=frames)
        file = write_scene(cfg, source, name, data)
        sp_count = {}
        for s in segs:
            sp_count[s["split"]] = sp_count.get(s["split"], 0) + 1
        summaries.append(dict(name=name, file=file, location=loc,
                              desc=f"{logrec['date_captured']} · {len(segs)} scene · 최장 연속 {longest:.0f}초",
                              n_kf=len(kfs), dur=kfs[-1]["t"], n_scenes=len(segs), longest=round(longest),
                              splits=sp_count, n_cuts=len(groups) - 1,
                              n_haz=(sum(bool(k["hazard"]) for k in kfs) if gt_mode else None),
                              n_gold=(sum(k["gold"] is not None for k in kfs) if gt_mode else 0),
                              maneuvers=mans, sweeps=sweeps_all, first_scene=w.scene_name(sc0["token"])))
        print(f"  [{n_done + 1}/{len(order)}] {name}: {len(segs)} scene, 재생 {kfs[-1]['t'] / 60:.1f}분, 프레임 {len(frames)}")
    return summaries


def update_index(cfg, source: str, summaries: list[dict], merge: bool):
    path = os.path.join(cfg["out_dir"], "index.json")
    idx = {"sources": {}}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            idx = json.load(f)
    src = cfg["sources"][source]
    old = idx["sources"].get(source, {}).get("scenes", []) if merge else []
    by = {s["name"]: s for s in old}
    by.update({s["name"]: s for s in summaries})
    idx["sources"][source] = dict(title=src["title"], note=src["note"], default=src["default"],
                                  scenes=sorted(by.values(), key=lambda s: s["name"]),
                                  updated=time.strftime("%Y-%m-%d %H:%M"))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=1)
    print(f"✅ {os.path.relpath(path, ROOT)} — {source} {len(by)} scene")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/demo.yaml")
    ap.add_argument("--source", choices=["split_test", "v1_test", "v1_test_long", "trainval_long"])
    ap.add_argument("--thresholds", action="store_true", help="val에서 τ를 다시 정한다")
    ap.add_argument("--scenes", nargs="*", help="이 scene만 (긴 영상 소스는 log 이름, 목록에 덧붙인다)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    os.chdir(ROOT)
    cfg = load_cfg(args.config)
    device = args.device or core.device_name()
    th = core.load_thresholds(cfg)
    need_th = args.thresholds or any(m not in th for m, mc in cfg["methods"].items() if "ckpt" in mc)

    env = None
    if need_th or args.source in ("split_test", "trainval_long"):
        from src.utils.config import load_cfgs
        base, lab, tc = load_cfgs(os.path.join(ROOT, "configs", "train_proposed.yaml"))
        gold = os.path.join(ROOT, "data", "goldset_v1.json")
        print("[env] trainval 로드 중…")
        env = core.ev().Env(base, lab, tc, cfg["sources"]["split_test"]["version"], gold_path=gold)
    if need_th:
        print("[임계값] val에서 라벨 표시량에 맞춤")
        th = compute_thresholds(env, cfg, device)
    if args.source == "split_test":
        sums = export_split_test(cfg, env, th, device, args.scenes, args.limit)
    elif args.source == "v1_test":
        sums = export_v1_test(cfg, th, device, args.scenes, args.limit)
    elif args.source == "v1_test_long":
        sums = export_long(cfg, "v1_test_long", th, device, None, args.scenes, args.limit)
    elif args.source == "trainval_long":
        sums = export_long(cfg, "trainval_long", th, device, env, args.scenes, args.limit)
    else:
        return 0
    update_index(cfg, args.source, sums, merge=bool(args.scenes or args.limit))
    return 0


if __name__ == "__main__":
    sys.exit(main())
