from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
from scipy.spatial import distance as dist

Centroid = Tuple[int, int]
Rect = Tuple[int, int, int, int]


class CentroidTracker:
    def __init__(self, max_disappeared: int = 40) -> None:
        self.next_object_id = 0
        self.objects: Dict[int, Centroid] = {}
        self.disappeared: Dict[int, int] = {}
        self.max_disappeared = max_disappeared

    def register(self, centroid: Centroid) -> None:
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id: int) -> None:
        self.objects.pop(object_id, None)
        self.disappeared.pop(object_id, None)

    def update(self, rects: Iterable[Rect]) -> Dict[int, Centroid]:
        rects_list = list(rects)

        if len(rects_list) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return dict(self.objects)

        input_centroids = np.zeros((len(rects_list), 2), dtype="int")
        for (i, (start_x, start_y, end_x, end_y)) in enumerate(rects_list):
            center_x = int((start_x + end_x) / 2.0)
            center_y = int((start_y + end_y) / 2.0)
            input_centroids[i] = (center_x, center_y)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register((int(input_centroids[i][0]), int(input_centroids[i][1])))
            return dict(self.objects)

        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()))

        distance_matrix = dist.cdist(object_centroids, input_centroids)
        rows = distance_matrix.min(axis=1).argsort()
        cols = distance_matrix.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            object_id = object_ids[row]
            self.objects[object_id] = (int(input_centroids[col][0]), int(input_centroids[col][1]))
            self.disappeared[object_id] = 0

            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(0, distance_matrix.shape[0])).difference(used_rows)
        unused_cols = set(range(0, distance_matrix.shape[1])).difference(used_cols)

        if distance_matrix.shape[0] >= distance_matrix.shape[1]:
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
        else:
            for col in unused_cols:
                self.register((int(input_centroids[col][0]), int(input_centroids[col][1])))

        return dict(self.objects)
