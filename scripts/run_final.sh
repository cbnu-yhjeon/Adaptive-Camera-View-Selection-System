#!/bin/bash
# 골드셋 완료 후 최종 산출물 일괄 생성 (M6~M8).
#   표 1(본·부록) → 강건성(골드 Top-1 포함) → 그림 2 → 논문 수치
# 사용: bash scripts/run_final.sh [골드셋 결과 파일]   (기본 data/goldset_v1.json)
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
GOLD=${1:-data/goldset_v1.json}
if [ "$GOLD" != "data/goldset_v1.json" ]; then cp "$GOLD" data/goldset_v1.json; fi
$PY -c "import json;d=json.load(open('data/goldset_v1.json',encoding='utf-8'));print('골드셋', d['meta']['n_done'], '/', d['meta']['n_items'], '장')"

# A2(위험도 제거) 최신 trainval 실행 — 강건성 비교 행
A2=$($PY - <<'PY'
import importlib.util
s = importlib.util.spec_from_file_location("ev", "scripts/06_evaluate.py"); ev = importlib.util.module_from_spec(s); s.loader.exec_module(ev)
print(ev.latest_ckpt("ablation_a2_no_risk", "v1.0-trainval"))
PY
)

$PY scripts/06_evaluate.py --all --split test                     # → results/table1.md
$PY scripts/06_evaluate.py --all --split test --risk_source gt   # → results/table1_gt.md (부록)
$PY scripts/07_robustness.py --split test --extra_ckpt "$A2"     # → results/robustness/robustness_test.*
$PY scripts/08_make_figures.py                                    # → results/fig2/
$PY scripts/08b_paper_numbers.py                                  # → results/paper_numbers.md
echo "✅ 최종 산출물 생성 완료"
