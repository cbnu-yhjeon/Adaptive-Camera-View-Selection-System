"""논문 수치 모음 (CLAUDE.md M8) → results/paper_numbers.md

논문 2.4~2.5절에 옮길 수치를 **결과 파일에서만** 모은다(원칙 4 — 손으로 옮기지 않는다).
파일이 없으면 그 절은 '미생성'으로 남긴다. 실험 설정도 실제로 쓴 설정 파일에서 읽는다.

  데이터·라벨     results/label_report.json, results/operating_curve.json
  입력 검증       results/can_check.json
  표 1           results/eval/{방법}_test_noisy-medium.json (06_evaluate --all)
  ablation       results/ablation.json (07b_ablation)
  강건성         results/robustness/robustness_test.json (07_robustness)
  설정           configs/train_proposed.yaml · label_v1.yaml · base.yaml, splits

사용: python scripts/08b_paper_numbers.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_cfgs  # noqa: E402

_spec = importlib.util.spec_from_file_location("ev", os.path.join(os.path.dirname(__file__), "06_evaluate.py"))
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

METHODS = ["b1", "b2", "b2c", "b3", "b4", "model"]


def jload(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


pct = lambda v, d=1: "—" if v is None else f"{100 * v:.{d}f}%"
p1 = lambda v: "—" if v is None else f"{v:.1f}%"
f3 = lambda v: "—" if v is None else f"{v:.3f}"


def sec_data(L):
    rep, oc = jload("results/label_report.json"), jload("results/operating_curve.json")
    if rep is None:
        return ["## 1. 데이터·의사 라벨\n", "_미생성 — scripts/02c_label_report.py_\n"]
    h, lb, sg = rep["hazard"], rep["labels"], rep["signal"]["by_maneuver"]
    mv = lb["maneuver"]
    out = ["## 1. 데이터·의사 라벨\n",
           f"- nuScenes v1.0-trainval keyframe **{rep['n_keyframes']:,}장** "
           f"(분할 train {h['n_by_split']['train']:,} / val {h['n_by_split']['val']:,} / test {h['n_by_split']['test']:,})",
           f"- '실제 위험'(TTC < {h['ttc']:.0f}초 ∧ DCA < {h['dca']:.0f} m): 위험 채널 **{h['channels']:,}개**, "
           f"위험 프레임 **{h['frames']:,}장 ({h['frame_pct']}%)**, test {h['by_split']['test']}장",
           "- 기동 분포: " + " · ".join(f"{k} {v:,}" for k, v in sorted(mv.items(), key=lambda x: -x[1])),
           f"- 보조 뷰 수 분포: 0장 {lb['aux_share']['0']}% · 1장 {lb['aux_share']['1']}% · 2장 {lb['aux_share']['2']}% "
           f"(평균 {lb['mean_aux']}장), 보조 뷰 집합 flicker {lb['flicker_aux_set']}",
           "- 방향지시등 일치(라벨 독립 검증, 창 [−1, +5]초): " + " · ".join(
               f"{k} 같은 방향 {v['same_pct']}% / 반대 {v['opposite_pct']}%" for k, v in sg.items()
               if v.get("same_pct") is not None)]
    if oc is not None:
        ad = next((p for p in oc["curve"] if abs(p.get("theta", -1) - oc["adopted"]) < 1e-9), None)
        a1 = oc["always_one"]
        if ad:
            out.append(f"- 운영점 θ={oc['adopted']}: 평균 표시 **{ad['load']:.2f}장**, 위험 포착률 **{ad['recall']:.1f}%**, "
                       f"정밀도 **{ad['precision']:.1f}%** (항상 1장: {a1['recall']:.1f}% / {a1['precision']:.1f}%)")
    return out + [""]


def sec_can():
    c = jload("results/can_check.json")
    if c is None:
        return []
    s, g = c["speed"], c["signal"]
    return ["## 2. 입력 검증 (CAN)\n",
            f"- 속도: ego_pose 차분 vs CAN 상관 {s['pose_diff_vs_can_pose']['corr']:.4f}, 정지 판정 불일치 "
            f"{s['stopped_disagree']['pose_stop_can_moving'] + s['stopped_disagree']['pose_moving_can_stop']}건",
            f"- 방향지시등: 값이 램프 점멸 상태라 직전 값만 쓰면 켜진 keyframe의 {pct(g['missed_by_latest_only'])}를 놓친다 "
            f"→ 직전 {g['hold_s']}초 유지 판정", ""]


def sec_table1():
    rows = {m: jload(f"results/eval/{m}_test_noisy-medium.json") for m in METHODS}
    if not any(rows.values()):
        return ["## 3. 표 1 (test, noisy-medium)\n", "_미생성 — scripts/06_evaluate.py --all_\n"]
    has_gold = any(r and "gold" in r for r in rows.values())
    out = ["## 3. 표 1 (test, 입력 noisy-medium)\n"]
    if has_gold:
        g0 = next(r["gold"] for r in rows.values() if r and "gold" in r)
        out.append(f"골드셋 {g0['n']}장(애매 {g0['n_hmm']}장). Top-1(가중)은 역확률 가중 — 전체 분포 추정치. "
                   "비가중 Top-1에 95% Wilson 구간.\n")
    out += ["| 방법 | Top-1 gold (가중) | Top-1 gold (비가중, 95% CI) | Top-2 gold | Macro-F1 gold | 정숙률 | "
            "Macro-F1 pseudo | 평균 표시 | 포착률 | 정밀도 | Flicker | 포착률 @같은 표시량 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in METHODS:
        r = rows[m]
        if not r:
            continue
        g = r.get("gold", {}); w = g.get("weighted", {}); u = g.get("unweighted", {})
        ci = ev.wilson(u.get("top1"), u.get("n_nonempty"))
        ut = "—" if ci is None else f"{pct(u['top1'])} [{100 * ci[0]:.1f}–{100 * ci[1]:.1f}]"
        out.append(f"| {ev.METHOD_KO[m]} | {pct(w.get('top1'))} | {ut} | {pct(w.get('top2'))} | {f3(w.get('f1'))} | "
                   f"{pct(w.get('quiet'))} | {f3(r['pseudo_f1'])} | {r['mean_aux']:.2f}장 | {p1(r['recall'])} | "
                   f"{p1(r['precision'])} | {f3(r['flicker'])} | {p1(r.get('recall_at_load'))} |")
    return out + [""]


def sec_ablation():
    a = jload("results/ablation.json")
    if a is None:
        return ["## 4. Ablation\n", "_미생성 — scripts/07b_ablation.py_\n"]
    cc, cn = a["conds"]
    ref = a["rows"][0]["res"][cn]["at_load"]["recall"]
    out = ["## 4. Ablation (test, 같은 표시량 {:.2f}장)\n".format(a["load"]),
           "| 태그 | 구성 | val AP | test AP (noisy) | 포착률 clean | 포착률 noisy | Δ | flicker |", "|---|---|---|---|---|---|---|---|"]
    for r in a["rows"]:
        n, c = r["res"][cn], r["res"][cc]
        rec = n["at_load"]["recall"]
        d = "—" if r is a["rows"][0] or rec is None else f"{rec - ref:+.1f}%p"
        out.append(f"| {r['tag']} | {r['desc']} | {f3(r['val_ap'])} | {f3(n['ap'])} | {p1(c['at_load']['recall'])} | "
                   f"{p1(rec)} | {d} | {f3(n['at_load']['flicker'])} |")
    return out + [""]


def sec_robust():
    d = jload("results/robustness/robustness_test.json")
    if d is None:
        return ["## 5. 강건성\n", "_미생성 — scripts/07_robustness.py --split test_\n"]
    names = list(d["results"][0]["at_load"])
    rows = {(r["type"], r["level"]): r for r in d["results"]}
    clean = rows[("all", "clean")]
    out = [f"## 5. 강건성 (test, 같은 표시량 {d['load']:.2f}장의 위험 포착률 · 골드 Top-1)\n",
           "| 조건 | " + " | ".join(f"{n} 포착률" for n in names) + " | " + " | ".join(f"{n} 골드 Top-1" for n in names) + " |",
           "|---|" + "---|" * (2 * len(names))]
    for key in [("all", "clean")] + [(t, "heavy") for t in ("miss", "fp", "pos", "drop")]:
        r = rows.get(key)
        if r is None:
            continue
        g = r.get("gold", {})
        out.append(f"| {key[0]} {key[1]} | " + " | ".join(p1(r["at_load"][n]["recall"]) for n in names) + " | "
                   + " | ".join(pct((g.get(n) or {}).get("top1")) for n in names) + " |")
    out.append("")
    out.append("열화(clean → heavy, 포착률): " + " · ".join(
        f"{t}: " + ", ".join(f"{n} {rows[(t, 'heavy')]['at_load'][n]['recall'] - clean['at_load'][n]['recall']:+.1f}%p"
                             for n in names if rows[(t, 'heavy')]['at_load'][n]['recall'] is not None
                             and clean['at_load'][n]['recall'] is not None)
        for t in ("miss", "fp", "pos", "drop") if (t, "heavy") in rows))
    return out + [""]


def sec_setup():
    base, lab, tc = load_cfgs("configs/train_proposed.yaml")
    sp = jload(base["splits_file"]) or {}
    m, lo, tr = tc.get("model", {}), tc.get("loss", {}), tc.get("train", {})
    lb = lab if "tau" in lab else lab.get("label", lab)
    return ["## 6. 실험 설정 (설정 파일 그대로)\n",
            f"- 분할: scene 단위 train {len(sp.get('train', []))} / val {len(sp.get('val', []))} / "
            f"test {len(sp.get('test', []))}, seed {base.get('seed', 42)}",
            f"- 모델: Transformer {m.get('tf_layers')}층·{m.get('heads')}헤드·d={m.get('d')}·ff={m.get('ff')}, 채널별 GRU, "
            f"입력 {tr.get('seq_len')}프레임(≈{tr.get('seq_len', 4) / 2:.0f}초), 백본 MobileNetV3-small 특징(576) 캐시",
            f"- 학습: AdamW lr {tr.get('lr')}, weight decay {tr.get('weight_decay')}, batch {tr.get('batch_size')}, "
            f"최대 {tr.get('epochs')} epoch, early stopping val macro-AP patience {tr.get('patience')}, "
            f"노이즈 증강 {tr.get('noise_mix')}",
            f"- 손실: focal BCE(γ={lo.get('focal_gamma')}) + KL α={lo.get('alpha')} + layout CE γ_l={lo.get('gamma_layout')} "
            f"+ temporal β={lo.get('beta')}",
            f"- 라벨: τ={lb.get('tau')}s, λ={lb.get('lambda_m')}, θ={lb.get('aux_threshold')}, m_guard={lb.get('m_guard')}, "
            f"aux_max={lb.get('aux_max')}, EMA α={lb.get('ema_alpha')}, d_near={lb.get('d_near')} m, d_scale={lb.get('d_scale')} m, "
            f"위험 정의 TTC<{lb.get('hazard_ttc')}s ∧ DCA<{lb.get('d_hazard')} m", ""]


def main() -> int:
    lines = ["# 논문 수치 모음\n",
             "`scripts/08b_paper_numbers.py`가 결과 파일에서 생성한다 — 손으로 고치지 말고 다시 생성할 것.\n"]
    for part in (sec_data(None), sec_can(), sec_table1(), sec_ablation(), sec_robust(), sec_setup()):
        lines += part
    os.makedirs("results", exist_ok=True)
    with open("results/paper_numbers.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("✅ results/paper_numbers.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
