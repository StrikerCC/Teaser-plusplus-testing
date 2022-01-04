# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 8/26/21 2:07 PM
"""
import copy
import glob
import json
import os
import time

import open3d as o3
import numpy as np
import transforms3d as t3d

from vis import draw_registration_result

"""
data/human_models/head_models/model_women/views/20210826133341-1.ply
data/human_models/head_models/model_women/views/20210826133425-1.ply
"""


def get_dataset(model_dir_path):
    """

    Args:
        model_dir_path (): dataset dir
            dataset dir organized as following
            model_dir_path/
                views/
                    *.ply
                    *.jpg
                    *.txt
                    dataset.json
                3D_model.pcd

            refer data/human_models/head_models/model_man for detail

    Returns:
        dataset
    """
    '''check point cloud folder exist'''
    if not model_dir_path[-1] == '/':
        model_dir_path += '/'
    views_dir_path = model_dir_path + 'views/'
    views_modified_dir_path = model_dir_path + 'views_modified/'

    assert os.path.isdir(views_dir_path), 'Dictionary for scan view not found'
    if not os.path.isdir(views_modified_dir_path): os.mkdir(views_modified_dir_path)

    with open(views_dir_path + 'dataset.json') as f:  # read dataset json
        dataset = json.load(f)
    for data in dataset:
        data['path'] = views_dir_path + data['path']
    return dataset


def preprocess(dataset, flag_vis=False):

    '''get point cloud in dataset for multi registration'''
    # original pcs, filtered pcs, down sampled pcs for global reg, down sampled pcs for local reg
    views_src, views_filtered, views_global_down_sample, views_local_down_sample = [], [], [], []
    for data in dataset:
        # load ply from different view
        view_file_path = data['path']
        view_original = o3.io.read_point_cloud(view_file_path)

        # preprocess each pc, filtering, cutting
        view_filtered = filter_pc(data, view_original)
        views_filtered.append(view_filtered)

        print('Data scanned at time: ', view_file_path[-16:-12], view_file_path[-12:-8], view_file_path[-8:-6], '\n',
              ' ', view_original, '\n'
                                  '     range from', view_original.get_max_bound(), view_original.get_min_bound())
        print('After outlier removal\n',
              ' ', view_filtered, '\n'
                                  '     range from',
              view_filtered.get_max_bound(),
              view_filtered.get_min_bound(), '\n')

        if flag_vis:  # and len(views_filtered) > 1:
            o3.visualization.draw_geometries([view_filtered], window_name='preprocess')
    return views_filtered


def compute_odometry(views_filtered, voxel_sizes_down_sample_global, voxel_sizes_down_sample_local, flag_vis=False):
    """"""
    '''compute transformations between point cloud in differnt view'''
    
    # align pc from different views together, global reg: icp, local reg: icp
    with o3.utility.VerbosityContextManager(o3.utility.VerbosityLevel.Debug) as mm:
        tf_list = reg_onebyone(views_filtered, voxel_sizes_down_sample_global, voxel_sizes_down_sample_local, flag_vis)
    
    # align pc from different views together, global reg: icp, local reg: icp, build pose graph and optimize pose using pose graph
    with o3.utility.VerbosityContextManager(o3.utility.VerbosityLevel.Debug) as mm:
        odometries = compute_odometry_with_pose_graph_opt(views_filtered, voxel_sizes_down_sample_global, voxel_sizes_down_sample_local,
                                                          flag_vis_reg=flag_vis)
    return odometries


def combine_pcs(views_filtered, tf_list, voxel_size_output):
    """

    Args:
        dataset (): dataset
        voxel_size_output (): merged point cloud voxel size
        flag_vis (): show registration detail?

    Returns:
        merged point cloud
    """
    ''''''
    views_filtered = copy.deepcopy(views_filtered)

    '''merge those pcs from different views'''
    pc_combined = o3.geometry.PointCloud()
    for view_id in range(len(views_filtered)):
        views_filtered[view_id].transform(tf_list[view_id])
        pc_combined += views_filtered[view_id]

    '''process the merged point cloud for proper output'''
    pcd_combined_down = pc_combined.voxel_down_sample(voxel_size=voxel_size_output)

    return pcd_combined_down


def compute_odometry_with_pose_graph_opt(pcs, voxel_sizes_down_sample_global, voxel_sizes_down_sample_local, flag_vis_reg=False):
    """"""
    tf_list = []

    '''build pose graph'''
    correspondence_distance_min = voxel_sizes_down_sample_local[-1]
    pose_graph = o3.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3.pipelines.registration.PoseGraphNode(
        odometry))  # the really first node, world coordinate in this registration

    # fill pose graph using tf between views

    for tgt_id in range(len(pcs)):
        odometry_tgt = copy.deepcopy(odometry)  # odemetry of current tgt
        for src_id in range(tgt_id + 1, min(tgt_id + 3, len(pcs))):     # only use next 2 view as non-neighbor node, more views are too hard to get registered with current one
            pc_src, pc_tar = pcs[src_id], pcs[tgt_id]
            tf, info, correspondence_distance_min = registration_pairwise(pc_src, pc_tar,
                                                                          voxel_sizes_down_sample_global,
                                                                          voxel_sizes_down_sample_local,
                                                                          flag_vis_reg)
            if src_id == tgt_id + 1:    # src pc and tgt pc are neighbor
                odometry = np.dot(odometry, tf)     # update current src odemetry
                odometry_src = odometry
                pose_graph.nodes.append(o3.pipelines.registration.PoseGraphNode(odometry))
                pose_graph.edges.append(o3.pipelines.registration.PoseGraphEdge(src_id,
                                                                                tgt_id,
                                                                                tf,
                                                                                info,
                                                                                uncertain=False))

            else:   # src pc and tgt pc are not neighbor
                odometry_src = np.dot(odometry_tgt, tf)
                pose_graph.edges.append(o3.pipelines.registration.PoseGraphEdge(src_id,
                                                                                tgt_id,
                                                                                tf,
                                                                                info,
                                                                                uncertain=True))

            print('tgt id:', tgt_id)
            print('src id:', src_id)
            # print('odometry :', odometry)
            if flag_vis_reg:
                draw_registration_result(pcs[src_id], pcs[0], odometry_src, window_name='current src in world coord(first pc in pcs)')

    option = o3.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=correspondence_distance_min,
        edge_prune_threshold=0.25,
        reference_node=0)
    o3.pipelines.registration.global_optimization(
        pose_graph,
        o3.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)
    for point_id in range(len(pcs)):
        print(pose_graph.nodes[point_id].pose)
        tf_list.append(pose_graph.nodes[point_id].pose)
    return tf_list


def reg_onebyone(pcs_src, voxel_sizes_down_sample_global, voxel_sizes_down_sample_local, flag_vis_reg=True):
    tf_list = [np.eye(4)]
    for tar_id in range(len(pcs_src) - 1):
        src_id = tar_id + 1
        pc_src, pc_tar = pcs_src[src_id], pcs_src[tar_id]
        odometry, *_ = registration_pairwise(pc_src, pc_tar, voxel_sizes_down_sample_global, voxel_sizes_down_sample_local, flag_vis_reg)
        tf_list.append(np.matmul(tf_list[-1], odometry))
    return tf_list


def registration_pairwise(pc_src, pc_tar, voxel_sizes_down_sample_global, voxel_sizes_down_sample_local, flag_vis_reg):
    """

    Args:
        pc_src (): src pc
        pc_tar (): tgt pc
        voxel_sizes_down_sample_global (): voxel sizes for global registration
        voxel_sizes_down_sample_local (): voxel sizes for local registration
        flag_vis_reg (): show registration result?

    Returns:
        tf_icp (): transformation that takes src to tgt
        information_icp (): information matrix
        correspondence_distance_icp (): min correspondence distance for local registration to stop
    """

    tf_init, information_icp, max_correspondence_distance = np.identity(4), None, 0
    '''global reg'''
    for voxel_size_down_sample_global in voxel_sizes_down_sample_global:
        # distance_threshold = voxel_size_down_sample_global
        distance_threshold = voxel_size_down_sample_global * 1.5

        '''downsampling'''
        pc_src_down_sample, pc_tar_down_sample = pc_src.voxel_down_sample(
            voxel_size_down_sample_global), pc_tar.voxel_down_sample(
            voxel_size_down_sample_global)

        '''compute features'''
        radius_normal = voxel_size_down_sample_global * 2
        radius_feature = voxel_size_down_sample_global * 5
        pc_src_down_sample.estimate_normals(o3.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        pc_tar_down_sample.estimate_normals(o3.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        pc_src_fpfh = o3.pipelines.registration.compute_fpfh_feature(pc_src_down_sample, o3.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature, max_nn=100))
        pc_tar_fpfh = o3.pipelines.registration.compute_fpfh_feature(pc_tar_down_sample, o3.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature, max_nn=100))

        '''ranasc'''
        result_global = o3.pipelines.registration.registration_ransac_based_on_feature_matching(
            source=pc_src_down_sample, target=pc_tar_down_sample, source_feature=pc_src_fpfh, target_feature=pc_tar_fpfh,
            mutual_filter=True, max_correspondence_distance=distance_threshold,
            estimation_method=o3.pipelines.registration.TransformationEstimationPointToPoint(False), ransac_n=4,
            checkers=[o3.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                      o3.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
            criteria=o3.pipelines.registration.RANSACConvergenceCriteria(10000000, 0.999)
        )
        if flag_vis_reg:
            draw_registration_result(source=pc_src_down_sample, target=pc_tar_down_sample, transformation=tf_init,
                                     window_name='ransac last')
        tf_init = result_global.transformation
        if flag_vis_reg:
            draw_registration_result(source=pc_src_down_sample, target=pc_tar_down_sample, transformation=tf_init,
                                     window_name='ransac current')

    '''local reg'''
    # tf_init = np.identity(4)
    for voxel_size_down_sample_local in voxel_sizes_down_sample_local:
        pc_src_down_sample, pc_tar_down_sample = pc_src.voxel_down_sample(voxel_size_down_sample_local), pc_tar.voxel_down_sample(
            voxel_size_down_sample_local)
        max_correspondence_distance = voxel_size_down_sample_local * 2.5
        radius_normal = voxel_size_down_sample_local * 2
        pc_src_down_sample.estimate_normals(o3.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        pc_tar_down_sample.estimate_normals(o3.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        icp_result = o3.pipelines.registration.registration_icp(
            source=pc_src_down_sample, target=pc_tar_down_sample,
            max_correspondence_distance=max_correspondence_distance,
            init=tf_init,
            estimation_method=o3.pipelines.registration.TransformationEstimationPointToPlane())
        # estimation_method=o3.pipelines.registration.TransformationEstimationPointToPoint())

        information_icp = o3.pipelines.registration.get_information_matrix_from_point_clouds(
            source=pc_src, target=pc_tar, max_correspondence_distance=max_correspondence_distance,
            transformation=icp_result.transformation)

        if flag_vis_reg:
            draw_registration_result(source=pc_src_down_sample, target=pc_tar_down_sample, transformation=tf_init,
                                     window_name='icp last')
        tf_init = icp_result.transformation
        if flag_vis_reg:
            draw_registration_result(source=pc_src_down_sample, target=pc_tar_down_sample, transformation=tf_init,
                                     window_name='icp current')
    tf_icp = tf_init
    correspondence_distance_icp = max_correspondence_distance
    return tf_icp, information_icp, correspondence_distance_icp


def filter_pc(para, pc):
    voxel_size = para['voxel_size']
    pc_down = pc.voxel_down_sample(0.2)
    points = np.asarray(pc_down.points)
    # points = np.asarray(pc.points)

    # k-means
    # pick a center
    # get rid of offset points
    # rectify orientation
    # cutoff out range points
    x_max_y_max_z_max, x_min_y_min_z_min = para['upper_bound'], para['lower_bound']
    x_max_y_max_z_max, x_min_y_min_z_min = np.asarray(x_max_y_max_z_max), np.asarray(x_min_y_min_z_min)
    assert np.alltrue(x_max_y_max_z_max > x_min_y_min_z_min)
    # x_max_y_max_z_max, x_min_y_min_z_min = x_max_y_max_z_max, x_min_y_min_z_min

    # x_min_y_max_z_max, x_max_y_min_z_min = np.copy(x_max_y_max_z_max), np.copy(x_min_y_min_z_min)
    # x_min_y_max_z_max[0], x_max_y_min_z_min[0] = x_min_y_min_z_min[0], x_max_y_max_z_max[0]
    # x_max_y_min_z_max, x_min_y_max_z_min = np.copy(x_max_y_max_z_max), np.copy(x_min_y_min_z_min)
    # x_max_y_min_z_max[1], x_min_y_max_z_min[1] = x_min_y_min_z_min[1], x_max_y_max_z_max[1]
    # x_max_y_max_z_min, x_min_y_min_z_max = np.copy(x_max_y_max_z_max), np.copy(x_min_y_min_z_min)
    # x_max_y_max_z_min[2], x_min_y_min_z_max[2] = x_min_y_min_z_min[2], x_max_y_max_z_max[2]
    # bounding_polygon = np.array([x_max_y_max_z_max, x_min_y_min_z_min,
    #                              x_min_y_max_z_max, x_max_y_min_z_min,
    #                              x_max_y_min_z_max, x_min_y_max_z_min,
    #                              x_max_y_max_z_min, x_min_y_min_z_max
    #                              ], dtype=np.float64) #* 1.5
    # print(bounding_polygon)
    # vol_pc = o3.geometry.PointCloud()
    # vol_pc.points = o3.utility.Vector3dVector(bounding_polygon)
    #
    # vol = o3.visualization.SelectionPolygonVolume()
    # vol.bounding_polygon = o3.utility.Vector3dVector(bounding_polygon)
    # comp = vol.crop_point_cloud(pc)

    pc_filter = o3.geometry.PointCloud()

    mask = np.all(points < x_max_y_max_z_max, axis=-1)
    mask = np.logical_and(mask, np.all(points > x_min_y_min_z_min, axis=-1))
    points_filtered = points[mask]
    # points_filtered = points

    pc_filter.points = o3.utility.Vector3dVector(points_filtered)

    '''remove outlier'''
    # pc_filter = pc_filter.voxel_down_sample(voxel_size=para['voxel_size']*0.001) if input_mm_output_m else pc_filter.voxel_down_sample(voxel_size=para['voxel_size'])
    pc_filter, _ = pc_filter.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)

    return pc_filter


def main():
    """
    merge point cloud from different view
    """
    '''input file and dataset path'''

    # model_dir_path = 'data/human_models/head_models/model_women/'
    '''dataset dir organized as following
    model_dir_path/ 
        views/
            *.ply
            *.jpg
            *.txt
            dataset.json
        3D_model.pcd
    refer data/human_models/head_models/model_man for detail'''

    flag_vis = True

    '''input dir'''
    # model_dir_path = 'data/human_models/head_models/Hongxiang_Chen/'
    model_dir_path = 'data/human_models/head_models/model_man/'
    # model_dir_path = 'data/human_models/head_models/model_women/'

    '''output file name and voxel size'''
    model_file_name = '/3D_model_camera.pcd'
    pc_merged_file_path = model_dir_path + model_file_name

    '''reg voxel size'''
    voxel_sizes_down_sample_global = (4,)
    voxel_sizes_down_sample_local = (5, 2, 1, 0.5, 0.2)
    voxel_size_output = 0.08

    '''get dataset'''
    dataset = get_dataset(model_dir_path)

    '''compute odometry from first view to others, odometry * view_other = view_first'''
    # start timer
    time_0 = time.time()
    views_filtered = preprocess(dataset, flag_vis=flag_vis)
    odometries = compute_odometry(views_filtered, voxel_sizes_down_sample_global, voxel_sizes_down_sample_local, flag_vis=flag_vis)

    '''merger accordingly'''
    pc_merged = combine_pcs(views_filtered, odometries, voxel_size_output)
    print('finish', len(dataset), 'pc merging in', time.time() - time_0, 'seconds')

    '''save each view pose relative to merged pc'''
    for i, (data, tf) in enumerate(zip(dataset, odometries)):
        # from now on, source is pc_merge, target is view, the transformation is odometry inverse
        draw_registration_result(source=pc_merged, target=views_filtered[i], transformation=np.linalg.inv(tf), window_name='model with view')
        ply_path = data['path']
        pose_file_path = ply_path[:-4] + '.txt'
        np.savetxt(pose_file_path, np.linalg.inv(tf))

    '''save the merged point cloud'''
    o3.io.write_point_cloud(pc_merged_file_path, pc_merged)

    pc_mergered_show = pc_merged.voxel_down_sample(voxel_size=3)
    pc_merged.paint_uniform_color([0.5, 0.706, 0.5])
    o3.visualization.draw_geometries([pc_mergered_show])


if __name__ == '__main__':
    main()
