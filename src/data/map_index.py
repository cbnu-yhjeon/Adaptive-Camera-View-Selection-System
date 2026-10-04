"""nuScenes 맵 확장 조회 — ego 위치가 어느 차로·교차로·주차장에 있는가.

기동 라벨을 궤적만으로 판정하면 "옆으로 2 m 움직였다"와 "차선을 바꿨다"를 구분하지 못한다.
주차장·진입로 통과, 주차 차량 회피도 같은 궤적을 만든다(2차 검수 scene-0972: 승하차 구역을
빠져나오는 장면이 LANE_CHANGE_RIGHT였다). 차로가 어디 있는지 알아야 가를 수 있다.

devkit의 `record_on_point`·`get_closest_lane`은 한 점에 약 78 ms라 trainval 34,149 keyframe에
44분이 걸린다. 폴리곤마다 STRtree 공간 인덱스를 직접 만들어 조회한다.

맵 데이터: nuScenes-map-expansion-v1.3 (`{dataroot}/maps/expansion/*.json`).
"""

from __future__ import annotations

import os
from collections import deque
from functools import lru_cache

from shapely import STRtree
from shapely.geometry import Point

LAYERS = ("lane", "lane_connector", "road_segment", "carpark_area")


class MapIndex:
    """한 지역(location) 맵의 공간 인덱스."""

    def __init__(self, dataroot: str, location: str):
        from nuscenes.map_expansion.map_api import NuScenesMap
        m = NuScenesMap(dataroot=os.path.expanduser(dataroot), map_name=location)
        self.location = location
        self._trees: dict[str, tuple[STRtree, list, list[str]]] = {}
        for layer in LAYERS:
            recs = getattr(m, layer)
            polys, toks = [], []
            for r in recs:
                poly = m.extract_polygon(r["polygon_token"])
                if poly.is_valid and not poly.is_empty:
                    polys.append(poly)
                    toks.append(r["token"])
            self._trees[layer] = (STRtree(polys), polys, toks)
        self.intersections = {r["token"] for r in m.road_segment if r["is_intersection"]}
        self.connectivity = m.connectivity

    def _hit(self, layer: str, pt: Point) -> str | None:
        tree, polys, toks = self._trees[layer]
        for i in tree.query(pt):
            if polys[i].contains(pt):
                return toks[i]
        return None

    def query(self, x: float, y: float) -> dict:
        """한 점의 맵 맥락. lane과 lane_connector는 겹치지 않는 것이 원칙이나 경계에서는
        둘 다 걸릴 수 있어 각각 돌려준다."""
        pt = Point(float(x), float(y))
        seg = self._hit("road_segment", pt)
        return dict(lane=self._hit("lane", pt),
                    connector=self._hit("lane_connector", pt),
                    intersection=seg in self.intersections if seg else False,
                    carpark=self._hit("carpark_area", pt) is not None)

    def connected(self, a: str, b: str, max_hops: int = 4) -> bool:
        """a에서 b로(또는 b에서 a로) 진행 방향 연결을 따라 max_hops 안에 닿는가.

        이어지는 차로(앞 차로 → 연결부 → 다음 차로)로 넘어간 것은 차선변경이 아니다.
        나란한 이웃 차로는 진행 방향으로 연결돼 있지 않다.
        """
        return self._reach(a, b, max_hops) or self._reach(b, a, max_hops)

    def _reach(self, src: str, dst: str, max_hops: int) -> bool:
        seen, q = {src}, deque([(src, 0)])
        while q:
            node, d = q.popleft()
            if node == dst:
                return True
            if d >= max_hops:
                continue
            for nxt in self.connectivity.get(node, {}).get("outgoing", []):
                if nxt not in seen:
                    seen.add(nxt)
                    q.append((nxt, d + 1))
        return False


@lru_cache(maxsize=None)
def get_index(dataroot: str, location: str) -> MapIndex:
    return MapIndex(dataroot, location)
