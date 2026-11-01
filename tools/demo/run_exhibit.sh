#!/usr/bin/env bash
# 전시 모드 실행 — 서버를 띄우고 Windows 브라우저를 전체 화면(키오스크)으로 연다.
#
#   bash tools/demo/run_exhibit.sh              # 전시용: 키오스크 전체 화면
#   bash tools/demo/run_exhibit.sh --window     # 일반 창으로 (점검·리허설)
#   bash tools/demo/run_exhibit.sh --rebuild    # 장면·수치를 다시 고른 뒤 실행
#   bash tools/demo/run_exhibit.sh --no-browser # 서버만
#   bash tools/demo/run_exhibit.sh --port 8800
#
# 종료: 이 터미널에서 Ctrl+C (서버가 내려간다). 키오스크 창은 Alt+F4로 닫는다.
# Windows에서는 같은 폴더의 run_exhibit.bat을 더블클릭해도 된다.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"

PORT=""; KIOSK=1; REBUILD=0; BROWSER=1
while [ $# -gt 0 ]; do
  case "$1" in
    --window) KIOSK=0 ;;
    --rebuild) REBUILD=1 ;;
    --no-browser) BROWSER=0 ;;
    --port) PORT="$2"; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "알 수 없는 옵션: $1 (--help)" >&2; exit 2 ;;
  esac
  shift
done

[ -x "$PY" ] || { echo "가상환경이 없습니다: $PY (README의 설치 절차 참조)" >&2; exit 1; }
# 포트는 configs/demo.yaml이 기본값 (하드코딩하지 않는다)
[ -n "$PORT" ] || PORT="$("$PY" -c "import yaml;print(yaml.safe_load(open('configs/demo.yaml'))['port'])")"
URL="http://127.0.0.1:${PORT}/exhibit"

# 1) 장면·수치 준비 — 없거나 --rebuild일 때만 (축소본까지 만들어 두므로 처음엔 몇 분 걸린다)
if [ "$REBUILD" = 1 ] || [ ! -f results/demo/exhibit.json ]; then
  echo "▶ 전시 장면 준비 (exhibit_build.py)"
  "$PY" tools/demo/exhibit_build.py
fi

# 2) 서버 — 이미 떠 있으면 그대로 쓴다
up() { curl -s -o /dev/null -m 1 -w '%{http_code}' "$URL" 2>/dev/null | grep -q 200; }
SRV=""
if up; then
  echo "▶ 서버가 이미 실행 중입니다: $URL"
else
  echo "▶ 서버 시작 (포트 $PORT)"
  mkdir -p logs
  "$PY" tools/demo/server.py --port "$PORT" > logs/exhibit_server.log 2>&1 &
  SRV=$!
  trap '[ -n "$SRV" ] && kill "$SRV" 2>/dev/null; echo; echo "■ 서버 종료"' EXIT INT TERM
  for _ in $(seq 1 40); do
    up && break
    kill -0 "$SRV" 2>/dev/null || { echo "서버가 시작되지 않았습니다 — logs/exhibit_server.log:" >&2; tail -20 logs/exhibit_server.log >&2; exit 1; }
    sleep 0.25
  done
  up || { echo "서버 응답 없음 — logs/exhibit_server.log 확인" >&2; exit 1; }
fi

# 3) 브라우저 — WSL이면 Windows의 Chrome(없으면 Edge)으로 연다
open_browser() {
  if command -v powershell.exe >/dev/null 2>&1; then
    local exe=""
    for c in "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe" \
             "/mnt/c/Program Files (x86)/Google/Chrome/Application/chrome.exe" \
             "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe" \
             "/mnt/c/Program Files/Microsoft/Edge/Application/msedge.exe"; do
      [ -f "$c" ] && { exe="$(wslpath -w "$c")"; break; }
    done
    if [ -z "$exe" ]; then
      cmd.exe /c start "" "$URL" >/dev/null 2>&1; return
    fi
    # 별도 프로필 — 이미 열린 브라우저 창에 붙으면 --kiosk가 무시된다
    local args="--user-data-dir=\$env:LOCALAPPDATA\\acvs-exhibit --no-first-run --disable-session-crashed-bubble --autoplay-policy=no-user-gesture-required"
    if [ "$KIOSK" = 1 ]; then args="$args --kiosk --edge-kiosk-type=fullscreen"; else args="$args --new-window --start-maximized"; fi
    powershell.exe -NoProfile -Command "Start-Process -FilePath '$exe' -ArgumentList \"$args $URL\"" >/dev/null 2>&1
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$URL" >/dev/null 2>&1 &
  else
    echo "브라우저에서 직접 여세요: $URL"
  fi
}
[ "$BROWSER" = 1 ] && { echo "▶ 브라우저 열기 ($([ "$KIOSK" = 1 ] && echo 키오스크 전체 화면 || echo 일반 창))"; open_browser; }

cat <<EOF

  전시 모드: $URL
  연구용 화면: http://127.0.0.1:${PORT}/
  조작: F 전체 화면 · Space 정지 · ←/→ 이동 · H AI 강조 켜고 끄기 · T 밝은 화면 · 0 처음으로
  키오스크 창 닫기: Alt+F4

EOF

if [ -n "$SRV" ]; then
  echo "서버 실행 중 — 종료하려면 Ctrl+C"
  wait "$SRV"
fi
