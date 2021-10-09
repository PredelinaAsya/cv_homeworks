#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'calc_track_interval_mappings',
    'calc_track_len_array_mapping',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import (
    FrameCorners,
    CornerStorage,
    StorageImpl,
    dump,
    load,
    draw,
    calc_track_interval_mappings,
    calc_track_len_array_mapping,
    without_short_tracks,
    create_cli
)


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:

    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 0.01))

    image_0 = (frame_sequence[0] * 255.0).astype(np.uint8)

    m, n = image_0.shape
    max_corners_count = m * n // 5000

    feature_params = dict(maxCorners=max_corners_count,
                          qualityLevel=0.1,
                          minDistance=15,
                          blockSize=15)

    p0 = cv2.goodFeaturesToTrack(image_0, mask=None, **feature_params).reshape((-1, 2))

    corners = FrameCorners(
        np.arange(len(p0)),
        p0,
        np.array([feature_params['blockSize']] * len(p0))
    )
    builder.set_corners_at_frame(0, corners)

    max_id = len(p0) - 1

    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        image_1 = (image_1 * 255.0).astype(np.uint8)
        p1, st, err = cv2.calcOpticalFlowPyrLK(image_0, image_1, corners.points,
                                               None, **lk_params, minEigThreshold=1.5 * 1e-3)
        corners = FrameCorners(
            corners.ids[st == 1],
            p1[np.hstack((st, st)) == 1].reshape((-1, 2)),
            corners.sizes[st == 1]
        )

        new_mask = np.full(image_1.shape, 255).astype(np.uint8)
        curr_corners = corners.points.astype(int)
        for i in range(curr_corners.shape[0]):
            cv2.circle(new_mask, (curr_corners[i, 0], curr_corners[i, 1]), corners.sizes[i, 0], 0, -1)

        p2 = cv2.goodFeaturesToTrack(image_1, mask=new_mask, **feature_params)
        if p2 is not None:
            p2 = p2.reshape((-1, 2))
            new_ids = np.arange(max_id + 1, len(p2) + max_id + 1).reshape((-1, 1))
            new_sz = np.array([feature_params['blockSize']] * len(p2)).reshape((-1, 1))
            max_id += (min(max_corners_count, len(corners.ids) + len(new_ids)) - len(corners.ids))
            corners = FrameCorners(
                np.concatenate((corners.ids, new_ids))[:max_corners_count],
                np.concatenate((corners.points, p2))[:max_corners_count],
                np.concatenate((corners.sizes, new_sz))[:max_corners_count]
            )
        builder.set_corners_at_frame(frame, corners)
        image_0 = image_1


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.
    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
