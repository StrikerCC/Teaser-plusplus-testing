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
from vis import draw_registration_result, draw_correspondence


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


def rigid_error(t1, t2):
    orientation_1, translation_1 = t1[:3, :3], t1[:3, 3]
    orientation_2, translation_2 = t2[:3, :3], t2[:3, 3]
    orientation_diff = np.matmul(orientation_2, orientation_1.T)
    error_o, error_t = np.rad2deg(np.asarray(t3d.euler.mat2euler(orientation_diff))), np.abs(
        translation_2 - translation_1)
    error_o, error_t = np.linalg.norm(error_o), np.linalg.norm(error_t)
    return error_o, error_t


def record(reg, i, statistic, voxel_size_reg, correspondence_set, tf_gt, tf_final, time_global, time_local, pc_src_global, pc_tar_global, pc_src_local, pc_tar_local):
    failure = False
    correspondence_set = np.asarray(correspondence_set)
    error_o, error_t = rigid_error(tf_gt, tf_final)

    statistic['#case'] += 1
    statistic['time_global'] += time_global
    statistic['time_local'] += time_local
    # # if differ from gt to much, count it as failure, not going to error statics
    if error_o > ORIENTATION_FAILURE_TOLERANCE or error_t > TRANSLATION_FAILURE_TOLERANCE:
        failure = True
        statistic['#failure'] += 1
        statistic['index_failure'].append(i)
        statistic['#points'].append((
            len(pc_src_global.points),
            len(pc_tar_global.points),
            len(pc_src_local.points),
            len(pc_tar_local.points),
        ))
        statistic['correspondence_set_failure'].append(correspondence_set.tolist())
        statistic['pose_failure'].append(tf_gt.tolist())
        statistic['tf_failure'].append(tf_final.tolist())
        statistic['error_t_failure'].append(error_t)
        statistic['error_o_failure'].append(error_o)
        statistic['voxel_size_reg_failure'].append(voxel_size_reg)
    else:
        statistic['error_t'] += error_t
        statistic['error_o'] += error_o

    # output
    if i % 50 == 0:
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
    return failure


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

        radius_normal = voxel_size_global * 2
        pc_src_global.estimate_normals(o3.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        pc_tar_global.estimate_normals(o3.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        # global registration
        radius_feature = voxel_size_global * 5
        pc_src_fpfh = o3.pipelines.registration.compute_fpfh_feature(pc_src_global, o3.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature, max_nn=100))
        pc_tar_fpfh = o3.pipelines.registration.compute_fpfh_feature(pc_tar_global, o3.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature, max_nn=100))
        # distance_threshold = voxel_size_local * 1.5
        distance_threshold = voxel_size_global
        result_global = o3.pipelines.registration.registration_ransac_based_on_feature_matching(
            source=pc_src_global, target=pc_tar_global, source_feature=pc_src_fpfh, target_feature=pc_tar_fpfh,
            mutual_filter=True, max_correspondence_distance=distance_threshold,
            estimation_method=o3.pipelines.registration.TransformationEstimationPointToPoint(False), ransac_n=3,
            checkers=[o3.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                      o3.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
            criteria=o3.pipelines.registration.RANSACConvergenceCriteria(10000000, 0.999)
        )
        time_global = time.time() - time_0

        # local registration
        time_0 = time.time()
        pc_src_local, pc_tar_local = pc_src.voxel_down_sample(voxel_size_local), pc_tar.voxel_down_sample(voxel_size_local)
        radius_normal_local = voxel_size_local * 2
        pc_src_local.estimate_normals(o3.geometry.KDTreeSearchParamHybrid(radius=radius_normal_local, max_nn=30))
        pc_tar_local.estimate_normals(o3.geometry.KDTreeSearchParamHybrid(radius=radius_normal_local, max_nn=30))


        tf_global = result_global.transformation
        distance_threshold_local = voxel_size_local
        result_local = o3.pipelines.registration.registration_icp(
            source=pc_src_local, target=pc_tar_local, max_correspondence_distance=distance_threshold_local,
            init=tf_global,
            estimation_method=o3.pipelines.registration.TransformationEstimationPointToPlane(),
            # criteria=
        )
        time_local = time.time() - time_0

        # record statics and output in screen
        tf_final = result_local.transformation
        record('ransac_icp', i, statistic, voxel_size_global, result_global.correspondence_set, pose_gt, tf_final, time_global, time_local, pc_src_global, pc_tar_global, pc_src_local, pc_tar_local)

        # vis
        if show_flag:
            show_per_reg_iter(method='ransac_icp', source=source, pc_src_global=pc_src_global,
                              pc_tar_global=pc_tar_global, correspondence_set_global=result_global.correspondence_set,
                              pc_src_local=pc_src_local, pc_tar_local=pc_tar_local,
                              time_global=time_global, time_local=time_local, tf_global=tf_global, tf_final=tf_final)
    format_statistic(statistic)


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

        # global registration
        time_0 = time.time()    # preprocessing include, down sampling, feature computation, tree building
        # establish correspondences by nearest neighbour search in feature space
        pc_src_global, pc_tar_global = pc_src.voxel_down_sample(voxel_size_global), pc_tar.voxel_down_sample(voxel_size_global)
        array_src_global, array_tar_global = np.asarray(pc_src_global.points).T, np.asarray(pc_tar_global.points).T

        # # extract FPFH features
        radius_normal_global = voxel_size_global * 2
        pc_src_global.estimate_normals(o3.geometry.KDTreeSearchParamHybrid(radius=radius_normal_global, max_nn=30))
        pc_tar_global.estimate_normals(o3.geometry.KDTreeSearchParamHybrid(radius=radius_normal_global, max_nn=30))

        radius_feature = voxel_size_global * 5
        pc_src_fpfh = o3.pipelines.registration.compute_fpfh_feature(pc_src_global, o3.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature, max_nn=100))
        pc_tar_fpfh = o3.pipelines.registration.compute_fpfh_feature(pc_tar_global, o3.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature, max_nn=100))
        feature_src_fpfh, feature_tar_fpfh = np.array(pc_src_fpfh.data).T, np.array(pc_tar_fpfh.data).T

        src_corrs_mask, tar_corrs_mask = find_correspondences(feature_src_fpfh, feature_tar_fpfh, mutual_filter=True)
        array_src_global, array_tar_global = array_src_global[:, src_corrs_mask], array_tar_global[:, tar_corrs_mask]  # np array of size 3 by num_corrs
        correspondence_set = np.hstack([np.expand_dims(src_corrs_mask, axis=-1), np.expand_dims(tar_corrs_mask, axis=-1)])

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
        pc_src_local, pc_tar_local = pc_src.voxel_down_sample(voxel_size_local), pc_tar.voxel_down_sample(
            voxel_size_local)
        radius_normal_local = voxel_size_local * 2
        pc_src_local.estimate_normals(o3.geometry.KDTreeSearchParamHybrid(radius=radius_normal_local, max_nn=30))
        pc_tar_local.estimate_normals(o3.geometry.KDTreeSearchParamHybrid(radius=radius_normal_local, max_nn=30))

        tf_global = T_teaser
        distance_threshold_local = voxel_size_local
        result_local = o3.pipelines.registration.registration_icp(
            source=pc_src_local, target=pc_tar_local, max_correspondence_distance=distance_threshold_local,
            init=tf_global,
            estimation_method=o3.pipelines.registration.TransformationEstimationPointToPlane(),
            # criteria=
        )
        time_local = time.time() - time_0

        # record statics
        tf_final = result_local.transformation
        record('ransac_icp', i, statistic, voxel_size_global, correspondence_set, pose_gt, tf_final, time_global,
               time_local, pc_src_global, pc_tar_global, pc_src_local, pc_tar_local)

        # vis
        if show_flag:
            show_per_reg_iter(method='fpfh_teaser_icp', source=source, pc_src_global=pc_src_global,
                              pc_tar_global=pc_tar_global, correspondence_set_global=correspondence_set,
                              pc_src_local=pc_src_local, pc_tar_local=pc_tar_local,
                              time_global=time_global, time_local=time_local,
                              tf_global=tf_global, tf_final=tf_final)

    format_statistic(statistic)


def show_per_reg_iter(method, source, pc_src_global, pc_tar_global, correspondence_set_global, pc_src_local, pc_tar_local, time_global, time_local, tf_global, tf_final):
    # visualize the point clouds together with feature correspondences

    print(method, 'instance: ', source['instance'], ', original voxel size: ', source['voxel_size'], ', noise sigma: ',
          source['sigma'], ', plane factor: ', source['plane'])
    print('#source_points_global:', len(np.asarray(pc_tar_global.points)), '#target_points_global', len(np.asarray(pc_src_global.points)), 'num_correspondence: ',
          len(correspondence_set_global))
    print('#source_points_local:', len(np.asarray(pc_tar_local.points)), '#target_points_local', len(np.asarray(pc_src_local.points)), 'num_correspondence: ',
          len(correspondence_set_global))
    print('time_global:', time_global, 'time_local', time_local)
    print('global error', rigid_error(source['pose'], tf_global))
    print('local error', rigid_error(source['pose'], tf_final))
    print()

    draw_registration_result(source=pc_src_global, target=pc_tar_global, window_name='init')
    draw_correspondence(pc_src_global, pc_tar_global, correspondence_set_global, window_name='correspondence')
    draw_registration_result(source=pc_src_global, target=pc_tar_global, transformation=tf_global,
                             window_name='global reg')
    draw_registration_result(source=pc_src_global, target=pc_tar_global, transformation=tf_final,
                             window_name='local reg')
    

VOXEL_SIZE_GLOBAL = [5, 5]
VOXEL_SIZE_LOCAL = [1, 1]

# VOXEL_SIZE_GLOBAL = [10]
# VOXEL_SIZE_LOCAL = [3]

# registrations = [ransac_icp, fgr_icp, fpfh_teaser_icp]
registrations = [ransac_icp, fpfh_teaser_icp]
# registrations = [fpfh_teaser_icp]
