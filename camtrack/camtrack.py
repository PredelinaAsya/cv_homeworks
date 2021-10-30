#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import cv2
import sortednp as snp
import sys

from corners import CornerStorage, FrameCorners
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    build_correspondences,
    triangulate_correspondences,
    rodrigues_and_translation_to_view_mat3x4,
    TriangulationParameters,
    Correspondences
)


def build_correspondences_corners_cloud(corners: FrameCorners,
                                        point_cloud: PointCloud) -> \
        Correspondences:
    ids_1 = corners.ids.flatten()
    ids_2 = point_cloud.ids.flatten()
    _, (indices_1, indices_2) = snp.intersect(ids_1, ids_2, indices=True)
    corrs = Correspondences(
        ids_1[indices_1],
        corners.points[indices_1],
        point_cloud.points[indices_2]
    )
    return corrs


def calc_view_matr_for_new_frame(corners: FrameCorners, point_cloud: PointCloud,
                                 intrinsic_mat: np.ndarray,
                                 stdout_file) -> np.array:
    correspondences_corners_cloud = build_correspondences_corners_cloud(
        corners, point_cloud)
    _, r_vec, t_vec, inliers = cv2.solvePnPRansac(
        np.array(correspondences_corners_cloud.points_2).astype(np.float32),
        np.array(correspondences_corners_cloud.points_1).astype(np.float32),
        intrinsic_mat, None)
    stdout_file.write(f'Number of inliers: {inliers.shape[0]}\n')
    stdout_file.write(
        f'Number of cloud points: {point_cloud.points.shape[0]}\n')
    return rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec)


def add_points_to_point_cloud(frame_id_1: int, frame_id_2: int,
                              point_cloud: PointCloudBuilder,
                              intrinsic_mat: np.ndarray,
                              corner_storage: CornerStorage,
                              view_matr_1: np.ndarray, view_matr_2: np.ndarray,
                              stdout_file):
    correspondences = build_correspondences(corner_storage[frame_id_1],
                                            corner_storage[frame_id_2])
    new_cloud_points, ids, _ = triangulate_correspondences(
        correspondences, view_matr_1, view_matr_2,
        intrinsic_mat, TriangulationParameters(1, 0, 0))
    stdout_file.write(
        f'Number of triangulated points: {new_cloud_points.shape[0]}\n')
    point_cloud.add_points(ids, new_cloud_points)


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    # TODO: implement
    stdout_file = sys.stdout
    frame_count = len(corner_storage)
    view_mats = [pose_to_view_mat3x4(known_view_1[1])] * frame_count
    view_mats[known_view_2[0]] = pose_to_view_mat3x4(known_view_2[1])
    correspondences = build_correspondences(corner_storage[known_view_1[0]],
                                            corner_storage[known_view_2[0]])
    cloud_points, ids, _ = triangulate_correspondences(
        correspondences, view_mats[known_view_1[0]], view_mats[known_view_2[0]],
        intrinsic_mat, TriangulationParameters(1, 0, 0))
    stdout_file.write(
        f'Number of triangulated points: {cloud_points.shape[0]}\n')
    point_cloud_builder = PointCloudBuilder(ids, cloud_points)
    point_cloud = point_cloud_builder.build_point_cloud()
    known_frames = 2
    for i in range(min(known_view_1[0], known_view_2[0]) + 1,
                   max(known_view_1[0], known_view_2[0])):
        stdout_file.write('Current frame: ' + str(i) + '\n')
        view_mats[i] = calc_view_matr_for_new_frame(
            corner_storage[i], point_cloud, intrinsic_mat, stdout_file)
        known_frames += 1

    interval_len = abs(known_view_2[0] - known_view_1[0])
    curr_index = 1
    while known_frames < frame_count:
        new_frame_left = min(known_view_1[0],
                             known_view_2[0]) - curr_index * interval_len
        if new_frame_left >= 0:
            stdout_file.write('Current frame: ' + str(new_frame_left) + '\n')
            view_mats[new_frame_left] = calc_view_matr_for_new_frame(
                corner_storage[new_frame_left],
                point_cloud_builder.build_point_cloud(),
                intrinsic_mat, stdout_file)
            add_points_to_point_cloud(new_frame_left,
                                      new_frame_left + interval_len,
                                      point_cloud_builder, intrinsic_mat,
                                      corner_storage, view_mats[new_frame_left],
                                      view_mats[new_frame_left + interval_len],
                                      stdout_file)
            known_frames += 1
            point_cloud = point_cloud_builder.build_point_cloud()
            for i in range(new_frame_left + 1, new_frame_left + interval_len):
                stdout_file.write('Current frame: ' + str(i) + '\n')
                view_mats[i] = calc_view_matr_for_new_frame(
                    corner_storage[i], point_cloud, intrinsic_mat, stdout_file)
                known_frames += 1
        new_frame_right = max(known_view_1[0],
                              known_view_2[0]) + curr_index * interval_len
        if new_frame_right < frame_count:
            stdout_file.write('Current frame: ' + str(new_frame_right) + '\n')
            view_mats[new_frame_right] = calc_view_matr_for_new_frame(
                corner_storage[new_frame_right],
                point_cloud_builder.build_point_cloud(),
                intrinsic_mat, stdout_file)
            add_points_to_point_cloud(new_frame_right,
                                      new_frame_right - interval_len,
                                      point_cloud_builder, intrinsic_mat,
                                      corner_storage,
                                      view_mats[new_frame_right],
                                      view_mats[new_frame_right - interval_len],
                                      stdout_file)
            known_frames += 1
            point_cloud = point_cloud_builder.build_point_cloud()
            for i in range(new_frame_right - interval_len + 1, new_frame_right):
                stdout_file.write('Current frame: ' + str(i) + '\n')
                view_mats[i] = calc_view_matr_for_new_frame(
                    corner_storage[i], point_cloud, intrinsic_mat, stdout_file)
                known_frames += 1
        curr_index += 1
        if 0 < new_frame_left < interval_len:
            stdout_file.write('Current frame: ' + str(0) + '\n')
            view_mats[0] = calc_view_matr_for_new_frame(
                corner_storage[0], point_cloud_builder.build_point_cloud(),
                intrinsic_mat, stdout_file)
            add_points_to_point_cloud(0, new_frame_left, point_cloud_builder,
                                      intrinsic_mat, corner_storage,
                                      view_mats[0], view_mats[new_frame_left],
                                      stdout_file)
            known_frames += 1
            point_cloud = point_cloud_builder.build_point_cloud()
            for i in range(new_frame_left):
                stdout_file.write('Current frame: ' + str(i) + '\n')
                view_mats[i] = calc_view_matr_for_new_frame(
                    corner_storage[i], point_cloud, intrinsic_mat, stdout_file)
                known_frames += 1
        if frame_count - interval_len < new_frame_right < frame_count:
            stdout_file.write('Current frame: ' + str(frame_count - 1) + '\n')
            view_mats[-1] = calc_view_matr_for_new_frame(
                corner_storage[-1], point_cloud_builder.build_point_cloud(),
                intrinsic_mat, stdout_file)
            add_points_to_point_cloud(new_frame_right, frame_count - 1,
                                      point_cloud_builder, intrinsic_mat,
                                      corner_storage,
                                      view_mats[new_frame_right],
                                      view_mats[-1], stdout_file)
            known_frames += 1
            point_cloud = point_cloud_builder.build_point_cloud()
            for i in range(new_frame_right, frame_count - 1):
                stdout_file.write('Current frame: ' + str(i) + '\n')
                view_mats[i] = calc_view_matr_for_new_frame(
                    corner_storage[i], point_cloud, intrinsic_mat, stdout_file)
                known_frames += 1

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
