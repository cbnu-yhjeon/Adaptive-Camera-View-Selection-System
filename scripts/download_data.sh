#!/usr/bin/env bash
# nuScenes 데이터 다운로드 (카메라 keyframe 전용) — CLAUDE.md B.2 단계 1·2
#
# nuScenes 공개 파일은 계정 로그인 없이 공식 CloudFront CDN에서 직접 받을 수 있다.
# 이 프로젝트는 LiDAR/RADAR와 sweeps(비-keyframe)를 쓰지 않으므로:
#   - 전체 센서 blob(~300GB) 대신 카메라 전용 blob(합 165GB)만 받고,
#   - 그중 samples/(keyframe)만 풀어서 디스크에는 ~27GB만 남긴다.
#
# 사용법:
#   bash scripts/download_data.sh                    # 기본 ~/data/nuscenes
#   DATAROOT=/path/to/nuscenes bash scripts/download_data.sh
#   BLOBS="01 02 03" bash scripts/download_data.sh   # 부분집합만
#
# 중단해도 안전하다. 다시 실행하면 완료된 파일은 건너뛰고, 받다 만 파일은 이어받는다.
set -euo pipefail

BASE="https://d36yt3mvayqw5m.cloudfront.net/public/v1.0"
DATAROOT="${DATAROOT:-$HOME/data/nuscenes}"
BLOBS="${BLOBS:-01 02 03 04 05 06 07 08 09 10}"
WORK="$DATAROOT/.download"          # 임시 tgz + 완료 마커
mkdir -p "$DATAROOT" "$WORK"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

fetch() {   # fetch <파일명> [tar 추가 인자...]
  local f="$1"; shift
  local marker="$WORK/$f.done"
  if [ -f "$marker" ]; then log "건너뜀  $f (이미 완료)"; return; fi

  log "받는 중  $f"
  curl -fL --retry 10 --retry-delay 5 --no-progress-meter -C - \
       -o "$WORK/$f" "$BASE/$f"

  log "푸는 중  $f"
  case "$f" in
    *.tgz) tar -xzf "$WORK/$f" -C "$DATAROOT" "$@" ;;
    *.zip) unzip -q -o "$WORK/$f" -d "$DATAROOT" ;;
  esac

  touch "$marker"
  rm -f "$WORK/$f"                  # 원본 압축파일은 즉시 삭제(디스크 절약)
  log "완료    $f"
}

log "저장 위치: $DATAROOT"

# 1) 메타데이터 + CAN expansion (필수)
fetch v1.0-trainval_meta.tgz
fetch can_bus.zip

# 2) mini — 스모크 테스트용. sweeps는 제외
fetch v1.0-mini.tgz --exclude 'sweeps/*'

# 3) trainval 카메라 blob — keyframe(samples/)만 추출
for i in $BLOBS; do
  fetch "v1.0-trainval${i}_blobs_camera.tgz" --wildcards 'samples/*'
done

log "===== 전체 완료 ====="
du -sh "$DATAROOT"/samples "$DATAROOT"/v1.0-* "$DATAROOT"/can_bus 2>/dev/null || true
