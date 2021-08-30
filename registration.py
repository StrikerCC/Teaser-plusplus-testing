# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 8/19/21 5:32 PM
"""
import time

import numpy as np
import transforms3d as t3d
import open3d as o3

from helpers import find_correspondences, get_teaser_solver, Rt2T
from dataset import Dataset
from vis import draw_registration_result


TRANSLATION_FAILURE_TOLERANCE = 3.0
ORIENTATION_FAILURE_TOLERANCE = 1.0

# statistic_detail = dataset().data_info
# statistic_detail['fail'] = True
# statistic_detail['tf'] = None
# # statistic_detail['']


def format_statistic(statistic):
    statistic['time_global'] /= statistic['#case']
    statistic['time_local'] /= statistic['#case']
    if statistic['#case'] - statistic['#failure'] == 0:
        statistic['error_t'] = np.nan
        statistic['error_o'] = np.nan
    else:
        statistic['error_t'] /= (statistic['#case'] - statistic['#failure'])
        statistic['error_o'] /= (statistic['#case'] - statistic['#failure'])
    assert statistic['#failure'] == len(statistic['index_failure']) == len(statistic['tf_failure']), str(statistic['#failure']) + ' ' + str(len(statistic['index_failure'])) + ' ' + str(len(statistic['tf_failure']))


def record(reg, i, statistic, voxel_size_reg, pose_gt, tf_final, time_global, time_local):
    orientation_gt, translation_gt = np.asarray(t3d.euler.mat2euler(pose_gt[:3, :3])), pose_gt[:3, 3]
    orientation, translation = t3d.euler.mat2euler(tf_final[:3, :3]), tf_final[:3, 3]
    error_t, error_o = np.rad2deg(np.abs(orientation_gt - orientation)), np.abs(translation_gt - translation)
    error_t, error_o = np.linalg.norm(error_t), np.linalg.norm(error_o)
    # # if differ from gt to much, count it as failure, not going to error statics
    statistic['#case'] += 1
    statistic['time_global'] += time_global
    statistic['time_local'] += time_local
    # statistic['detail']['tf'] = tf_final
    if error_o > ORIENTATION_FAILURE_TOLERANCE or error_t > TRANSLATION_FAILURE_TOLERANCE:
        statistic['#failure'] += 1
        statistic['index_failure'].append(i)
        statistic['pose_failure'].append(pose_gt.tolist())
        statistic['tf_failure'].append(tf_final.tolist())
        statistic['error_t_failure'].append(error_t)
        statistic['error_o_failure'].append(error_o)
        statistic['voxel_size_reg_failure'].append(voxel_size_reg)
    else:
        statistic['error_t'] += error_t
        statistic['error_o'] += error_o

    # output
    if i % 100 == 0:
        print('iter', i, '\n   Method', reg,
              '\n   Time average',
              statistic['time_global'] / statistic['#case'] + statistic['time_local'] / statistic['#case'])
        if statistic['#case'] - statistic['#failure'] == 0:
            print('   No successful case')
        else:
            print(
                '   Translation rms', statistic['error_t'] / (statistic['#case'] - statistic['#failure']),
                '\n   Orientation rms', statistic['error_o'] / (statistic['#case'] - statistic['#failure']),
                '\n   Failure percent', statistic['#failure'] / statistic['#case'])


def output(statistic):
    """output"""
    print('ransac_icp')
    print('Translation rms', statistic['error_t'])
    print('Orientation rms', statistic['error_o'])
    print('Time average', (statistic['time_global'] + statistic['time_local']))
    print('Failure percent', statistic['#failure'] / statistic['#case'])


def ransac_icp(dataloader, voxel_size_global, voxel_size_local, statistic, show_flag=False):
    # VOXEL_SIZE_GLOBAL = 5
    # VOXEL_SIZE_LOCAL = 3

    # read source and target pc
    for i in range(len(dataloader)):
        source = dataloader[i]
        pose_gt = source['pose']
        # orientation_gt, translation_gt = np.asarray(t3d.euler.mat2euler(pose_gt[:3, :3])), pose_gt[:3, 3]

        # pc_src, pc_tar = o3.io.read_point_cloud(source['pc_model']), o3.io.read_point_cloud(source['pc_artificial'])
        pc_src, pc_tar = source['pc_model'], source['pc_artificial']

        # preprocessing include, down sampling, feature computation, tree building
        time_0 = time.time()
        pc_src_global, pc_tar_global = pc_src.voxel_down_sample(voxel_size_global), pc_tar.voxel_down_sample(voxel_size_global)
        pc_src_local, pc_tar_local = pc_src.voxel_down_sample(voxel_size_local), pc_tar.voxel_down_sample(voxel_size_local)

        radius_normal = voxel_size_global * 2
        pc_src_global.estimate_normals(o3.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        pc_tar_global.estimate_normals(o3.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        radius_normal = voxel_size_local * 2
        pc_src_local.estimate_normals(o3.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        pc_tar_local.estimate_normals(o3.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        # global registration
        radius_feature = voxel_size_global * 5
        pc_src_fpfh = o3.pipelines.registration.compute_fpfh_feature(pc_src_global, o3.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature, max_nn=100))
        pc_tar_fpfh = o3.pipelines.registration.compute_fpfh_feature(pc_tar_global, o3.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature, max_nn=100))
        # transformation_init = np.eye(4)
        distance_threshold = voxel_size_global * 1.5
        result_global = o3.pipelines.registration.registration_ransac_based_on_feature_matching(
            source=pc_src_global, target=pc_tar_global, source_feature=pc_src_fpfh, target_feature=pc_tar_fpfh, mutual_filter=True,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3.pipelines.registration.TransformationEstimationPointToPoint(False), ransac_n=4,
            checkers=[o3.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                      o3.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
            criteria=o3.pipelines.registration.RANSACConvergenceCriteria(10000000, 0.999)
        )
        time_global = time.time() - time_0

        # local registration
        time_0 = time.time()
        tf_global = result_global.transformation
        distance_threshold = voxel_size_global
        result_local = o3.pipelines.registration.registration_icp(
            source=pc_src_local, target=pc_tar_local, max_correspondence_distance=distance_threshold,
            init=tf_global,
            estimation_method=o3.pipelines.registration.TransformationEstimationPointToPlane(),
            # criteria=
        )
        time_local = time.time() - time_0

        # record statics and output in screen
        tf_final = result_local.transformation
        record('ransac_icp', i, statistic, voxel_size_global, pose_gt, tf_final, time_global, time_local)

        # vis
        if show_flag:
            print('time_global, time_local', time_global, time_local)
            print('icp tf', np.matmul(tf_final, np.linalg.inv(tf_global)))
            draw_registration_result(source=pc_src_global, target=pc_tar_global, window_name='init')
            draw_registration_result(source=pc_src_global, target=pc_tar_global, transformation=tf_global,
                                     window_name='global reg')
            draw_registration_result(source=pc_src_global, target=pc_tar_global, transformation=tf_final,
                                     window_name='local reg')


    format_statistic(statistic)
    output(statistic)

    # return statistic


def fgr_icp(dataloader, voxel_size_global, voxel_size_local, statistic, show_flag=False):
    # VOXEL_SIZE_GLOBAL = 10
    # VOXEL_SIZE_LOCAL = 3

    # read source and target pc
    for i in range(len(dataloader)):
        source = dataloader[i]
        pose_gt = source['pose']

        # pc_src, pc_tar = o3.io.read_point_cloud(source['pc_model']), o3.io.read_point_cloud(source['pc_artificial'])
        pc_src, pc_tar = source['pc_model'], source['pc_artificial']

        # preprocessing include, down sampling, feature computation, tree building
        time_0 = time.time()
        pc_src_global, pc_tar_global = pc_src.voxel_down_sample(voxel_size_global), pc_tar.voxel_down_sample(voxel_size_global)
        pc_src_local, pc_tar_local = pc_src.voxel_down_sample(voxel_size_local), pc_tar.voxel_down_sample(voxel_size_local)

        radius_normal = voxel_size_global * 2
        pc_src_global.estimate_normals(o3.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        pc_tar_global.estimate_normals(o3.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        radius_normal = voxel_size_local * 2
        pc_src_local.estimate_normals(o3.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        pc_tar_local.estimate_normals(o3.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size_global * 5
        pc_src_fpfh = o3.pipelines.registration.compute_fpfh_feature(pc_src_global, o3.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature, max_nn=100))
        pc_tar_fpfh = o3.pipelines.registration.compute_fpfh_feature(pc_src_global, o3.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature, max_nn=100))

        # global registration
        transformation_init = np.eye(4)
        distance_threshold = voxel_size_global * 1.5
        result_global = o3.pipelines.registration.registration_fast_based_on_feature_matching(
            source=pc_src_global, target=pc_tar_global, source_feature=pc_src_fpfh, target_feature=pc_tar_fpfh,
            option=o3.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=distance_threshold)
        )
        time_global = time.time() - time_0

        # local registration
        time_0 = time.time()
        transformation_init = result_global.transformation
        distance_threshold = voxel_size_local * 15
        result_local = o3.pipelines.registration.registration_icp(
            source=pc_src_local, target=pc_tar_local, max_correspondence_distance=distance_threshold, init=transformation_init,
            estimation_method=o3.pipelines.registration.TransformationEstimationPointToPlane(),
            # criteria=
        )
        time_local = time.time() - time_0

        # record statics
        tf_final = result_local.transformation
        record('fgr_icp', i, statistic, voxel_size_global, pose_gt, tf_final, time_global, time_local)

        # vis
        if show_flag:
            # print(time_global, time_local)
            draw_registration_result(source=pc_src_global, target=pc_src_global)
            draw_registration_result(source=pc_src_global, target=pc_src_global, transformation=tf_final)

    format_statistic(statistic)
    """output"""
    output(statistic)


def fpfh_teaser_icp(dataloader, voxel_size_global, voxel_size_local, statistic, show_flag=False):
    # VOXEL_SIZE_GLOBAL = 7
    # VOXEL_SIZE_LOCAL = 3

    # read source and target pc
    for i in range(len(dataloader)):
        source = dataloader[i]
        pose_gt = source['pose']
        # orientation_gt, translation_gt = np.asarray(t3d.euler.mat2euler(pose_gt[:3, :3])), pose_gt[:3, 3]

        # pc_src, pc_tar = o3.io.read_point_cloud(source['pc_model']), o3.io.read_point_cloud(source['pc_artificial'])
        pc_src, pc_tar = source['pc_model'], source['pc_artificial']

        # preprocessing include, down sampling, feature computation, tree building
        time_0 = time.time()
        pc_src_global, pc_tar_global = pc_src.voxel_down_sample(voxel_size_global), pc_tar.voxel_down_sample(voxel_size_global)
        pc_src_local, pc_tar_local = pc_src.voxel_down_sample(voxel_size_local), pc_tar.voxel_down_sample(voxel_size_local)
        array_src_global, array_tar_global = np.asarray(pc_src_global.points).T, np.asarray(pc_tar_global.points).T

        # # extract FPFH features
        radius_normal = voxel_size_global * 2
        pc_src_global.estimate_normals(o3.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        pc_tar_global.estimate_normals(o3.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        radius_normal = voxel_size_local * 2
        pc_src_local.estimate_normals(o3.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        pc_tar_local.estimate_normals(o3.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size_global * 5
        pc_src_fpfh = o3.pipelines.registration.compute_fpfh_feature(pc_src_global, o3.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature, max_nn=100))
        pc_tar_fpfh = o3.pipelines.registration.compute_fpfh_feature(pc_tar_global, o3.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature, max_nn=100))
        feature_src_fpfh, feature_tar_fpfh = np.array(pc_src_fpfh.data).T, np.array(pc_tar_fpfh.data).T

        # global registration
        # establish correspondences by nearest neighbour search in feature space
        src_corrs_mask, tar_corrs_mask = find_correspondences(feature_src_fpfh, feature_tar_fpfh, mutual_filter=True)
        array_src_global, array_tar_global = array_src_global[:, src_corrs_mask], array_tar_global[:, tar_corrs_mask]  # np array of size 3 by num_corrs

        # line
        # robust global registration using TEASER++
        NOISE_BOUND = voxel_size_global
        teaser_solver = get_teaser_solver(NOISE_BOUND)
        teaser_solver.solve(array_src_global, array_tar_global)
        solution = teaser_solver.getSolution()
        R_teaser = solution.rotation
        t_teaser = solution.translation
        T_teaser = Rt2T(R_teaser, t_teaser)

        time_global = time.time() - time_0

        # local registration
        time_0 = time.time()
        tf_global = T_teaser
        distance_threshold = voxel_size_global * 10
        result_local = o3.pipelines.registration.registration_icp(
            source=pc_src_local, target=pc_tar_local, max_correspondence_distance=distance_threshold,
            init=tf_global,
            estimation_method=o3.pipelines.registration.TransformationEstimationPointToPlane(),
            # criteria=
        )
        time_local = time.time() - time_0

        # record statics
        tf_final = result_local.transformation
        record('fpfh_teaser_icp', i, statistic, voxel_size_global, pose_gt, tf_final, time_global, time_local)

        # vis
        if show_flag:
            num_corrs = array_src_global.shape[1]

            # visualize the point clouds together with feature correspondences
            points = np.concatenate((array_src_global.T, array_tar_global.T), axis=0)
            lines = []
            for i in range(num_corrs):
                lines.append([i, i + num_corrs])
            colors = [[0, 1, 0] for i in range(len(lines))]  # lines are shown in green
            line_set = o3.geometry.LineSet(
                points=o3.utility.Vector3dVector(points),
                lines=o3.utility.Vector2iVector(lines),
            )
            line_set.colors = o3.utility.Vector3dVector(colors)

            print('time_global, time_local, num_corrs', time_global, time_local, num_corrs)
            print('icp tf', np.matmul(tf_final, np.linalg.inv(tf_global)))

            draw_registration_result(source=pc_src_global, target=pc_tar_global, window_name='init')
            o3.visualization.draw_geometries([pc_src_global, pc_tar_global, line_set])
            draw_registration_result(source=pc_src_global, target=pc_tar_global, transformation=tf_global, window_name='global reg')
            draw_registration_result(source=pc_src_global, target=pc_tar_global, transformation=tf_final, window_name='local reg')

    format_statistic(statistic)
    """output"""
    output(statistic)


# VOXEL_SIZE_GLOBAL = [5, 10, 7]
# VOXEL_SIZE_GLOBAL = [5, 5, 5]
VOXEL_SIZE_GLOBAL = [3]
VOXEL_SIZE_LOCAL = [3]
# registrations = [ransac_icp, fgr_icp, fpfh_teaser_icp]
registrations = [fpfh_teaser_icp]

