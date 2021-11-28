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
        intrinsic_mat, TriangulationParameters(3.0, 0.2, 0.2),
    )
    stdout_file.write(
        f'Number of triangulated points: {new_cloud_points.shape[0]}\n')
    point_cloud.add_points(ids, new_cloud_points)


def get_best_rotation_and_translation(
        intrinsic_mat: np.ndarray,
        correspondences: Correspondences,
        rotation1: np.ndarray,
        rotation2: np.ndarray,
        translation: np.ndarray,
):
    matr_for_first = np.hstack((np.eye(3), np.zeros(3).reshape(-1, 1)))
    cases = [(rotation1, translation),
             (rotation1, (-1) * translation),
             (rotation2, translation),
             (rotation2, (-1) * translation)]
    max_points_count = 0
    ans_rot, ans_t = None, None
    for rot, t in cases:
        cloud_points, _, _ = triangulate_correspondences(
            correspondences, matr_for_first, np.hstack((rot, t)),
            intrinsic_mat, TriangulationParameters(3.0, 0.2, 0.2)
        )
        if len(cloud_points) >= max_points_count:
            max_points_count = len(cloud_points)
            ans_rot, ans_t = rot, t
    return ans_rot, ans_t


def init_positions_for_known_frames(intrinsic_mat: np.ndarray,
                                    first_corners: FrameCorners,
                                    second_corners: FrameCorners
                                    ):
    correspondences = build_correspondences(first_corners, second_corners)
    points1, points2 = correspondences.points_1, correspondences.points_2
    ess_mat, ess_mask = cv2.findEssentialMat(
        points1, points2, intrinsic_mat, method=cv2.RANSAC, prob=0.999,
        threshold=1.0, maxIters=2000,
    )
    _, homo_mask = cv2.findHomography(
        points1, points2, method=cv2.RANSAC, ransacReprojThreshold=5.0,
    )
    rotation1, rotation2, t = cv2.decomposeEssentialMat(ess_mat)
    rotation, translation = get_best_rotation_and_translation(
        intrinsic_mat, correspondences, rotation1, rotation2, t,
    )
    _, homography_mask = cv2.findHomography(points1, points2, method=cv2.RANSAC)
    homography_inliers_part = np.sum(homography_mask) / len(homography_mask)
    ess_matrix_inliers_part = np.sum(ess_mask) / len(ess_mask)
    error = homography_inliers_part / ess_matrix_inliers_part
    return (np.hstack((rotation, translation)), error)


def find_and_init_positions(intrinsic_mat: np.ndarray,
                            corner_storage: CornerStorage,
                            k=8):
    best_err, camera_matr = np.inf, None
    first_index, second_index = None, None
    for i in range(k):
        for j in range(i + k, 2 * k):
            curr_camera_matr, curr_err = init_positions_for_known_frames(
                intrinsic_mat, corner_storage[i], corner_storage[j],
            )
            if curr_err < best_err or np.isinf(best_err):
                best_err, camera_matr = curr_err, curr_camera_matr
                first_index, second_index = i, j
    matr_for_first = np.hstack((np.eye(3), np.zeros(3).reshape((-1, 1))))
    print(f'first_index = {first_index}, second_index = {second_index}')
    return ((first_index, view_mat3x4_to_pose(matr_for_first)),
            (second_index, view_mat3x4_to_pose(camera_matr)))


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    if known_view_1 is None or known_view_2 is None:
        known_view_1, known_view_2 = find_and_init_positions(
            intrinsic_mat, corner_storage,
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
        intrinsic_mat, TriangulationParameters(3.0, 0.2, 0.2))
    stdout_file.write(
        f'Number of triangulated points: {cloud_points.shape[0]}\n')

    point_cloud_builder = PointCloudBuilder(ids, cloud_points)
    point_cloud = point_cloud_builder.build_point_cloud()
    known_frames = 2

    def process_current_frame(i: int, cloud: PointCloud):
        stdout_file.write('Current frame: ' + str(i) + '\n')
        view_mats[i] = calc_view_matr_for_new_frame(
            corner_storage[i], cloud, intrinsic_mat, stdout_file)

    for i in range(min(known_view_1[0], known_view_2[0]) + 1,
                   max(known_view_1[0], known_view_2[0])):
        process_current_frame(i, point_cloud)
        known_frames += 1

    interval_len = abs(known_view_2[0] - known_view_1[0])
    curr_index = 1
    while known_frames < frame_count:
        new_frame_left = min(known_view_1[0],
                             known_view_2[0]) - curr_index * interval_len
        if new_frame_left >= 0:
            process_current_frame(new_frame_left,
                                  point_cloud_builder.build_point_cloud())
            add_points_to_point_cloud(new_frame_left,
                                      new_frame_left + interval_len,
                                      point_cloud_builder, intrinsic_mat,
                                      corner_storage, view_mats[new_frame_left],
                                      view_mats[new_frame_left + interval_len],
                                      stdout_file)
            known_frames += 1
            point_cloud = point_cloud_builder.build_point_cloud()
            for i in range(new_frame_left + 1, new_frame_left + interval_len):
                process_current_frame(i, point_cloud)
                known_frames += 1
        new_frame_right = max(known_view_1[0],
                              known_view_2[0]) + curr_index * interval_len
        if new_frame_right < frame_count:
            process_current_frame(new_frame_right,
                                  point_cloud_builder.build_point_cloud())
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
                process_current_frame(i, point_cloud)
                known_frames += 1
        curr_index += 1
        if 0 < new_frame_left < interval_len or new_frame_left < 0:
            process_current_frame(0, point_cloud_builder.build_point_cloud())
            add_points_to_point_cloud(0, new_frame_left, point_cloud_builder,
                                      intrinsic_mat, corner_storage,
                                      view_mats[0], view_mats[new_frame_left],
                                      stdout_file)
            known_frames += 1
            point_cloud = point_cloud_builder.build_point_cloud()
            last_index = new_frame_left if new_frame_left > 0 \
                else min(known_view_1[0], known_view_2[0])
            for i in range(1, last_index):
                process_current_frame(i, point_cloud)
                known_frames += 1
        if frame_count - interval_len < new_frame_right < frame_count:
            process_current_frame(frame_count - 1,
                                  point_cloud_builder.build_point_cloud())
            add_points_to_point_cloud(new_frame_right, frame_count - 1,
                                      point_cloud_builder, intrinsic_mat,
                                      corner_storage,
                                      view_mats[new_frame_right],
                                      view_mats[-1], stdout_file)
            known_frames += 1
            point_cloud = point_cloud_builder.build_point_cloud()
            for i in range(new_frame_right, frame_count - 1):
                process_current_frame(i, point_cloud)
                known_frames += 1
        if new_frame_right >= frame_count:
            process_current_frame(frame_count - 1,
                                  point_cloud_builder.build_point_cloud())
            add_points_to_point_cloud(
                new_frame_right - interval_len, frame_count - 1,
                point_cloud_builder, intrinsic_mat, corner_storage,
                view_mats[max(new_frame_right - interval_len, 0)],
                view_mats[-1], stdout_file,
            )
            known_frames += 1
            point_cloud = point_cloud_builder.build_point_cloud()
            for i in range(new_frame_right - interval_len, frame_count - 1):
                process_current_frame(i, point_cloud)
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
