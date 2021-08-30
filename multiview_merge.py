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


def combine_pcs(model_dir_path, model_file_name='3D_model.pcd'):
    flag_vis = False
    # voxel_down_samples = (5, 2)
    voxel_down_samples = (5, 2, 1, 0.5, 0.2)
    # voxel_size_output = 0.273
    voxel_size_output = 1

    if not model_dir_path[-1] == '/':
        model_dir_path += '/'
    view_dir_path = model_dir_path + 'views/'
    model_file_path = model_dir_path + model_file_name
    assert os.path.isdir(view_dir_path), 'Dictionary for scan view not found'

    # load ply from different view
    with open(view_dir_path + 'filter_parameters.json') as f:
        paras_filter = json.load(f)
    views_src, views_global_down_sample, views_local_down_sample = [], [], []
    file_paths = sorted(glob.glob(view_dir_path + '*.ply'))
    file_paths = [file_paths[i] for i in [2, 1, 0, 3, 4]]

    # start timer
    time_0 = time.time()
    for para_filter, view_file_path in zip(paras_filter, file_paths):
        pc_src = o3.io.read_point_cloud(view_file_path)
        print('Data scanned at ', view_file_path[-14:-10], view_file_path[-10:-6], 'range from', pc_src.get_max_bound(),
              pc_src.get_min_bound())
        pc_src = filter_pc(para_filter, pc_src)
        views_src.append(pc_src)
        if flag_vis and len(views_src) > 1:
            o3.visualization.draw_geometries([views_src[-2]])
            # o3.visualization.draw_geometries(views_src[-2:])
    # align pc from different views together
    # # global reg: icp
    # # local reg: icp
    # tf_list = reg_onebyone(pcs_src=views_src, voxels_down_sample=voxel_down_samples)
    tf_list = reg_multiway(pcs=views_src, voxels_down_sample=voxel_down_samples)
    # merge those pcs from different views
    pc_combined = o3.geometry.PointCloud()
    for view_id in range(len(views_src)):
        views_src[view_id].transform(tf_list[view_id])
        pc_combined += views_src[view_id]
    pcd_combined_down = pc_combined.voxel_down_sample(voxel_size=voxel_size_output)
    print('finish', len(file_paths), 'in', time.time() - time_0, 'seconds')

    # transform the point cloud if necessary
    tf_flip = np.eye(4)
    if 'head' in model_file_path:
        tf_flip[:3, :3] = t3d.euler.euler2mat(*np.deg2rad([180.0, 0.0, 0.0]))
    pcd_combined_down.transform(tf_flip)

    o3.io.write_point_cloud(model_file_path, pcd_combined_down)
    model_file_path = model_file_path[:-3] + 'ply'
    o3.io.write_point_cloud(model_file_path, pcd_combined_down)
    o3.visualization.draw_geometries([pcd_combined_down])


def reg_multiway(pcs, voxels_down_sample, flag_vis_reg=False):
    pose_graph = o3.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3.pipelines.registration.PoseGraphNode(odometry))  # the really first node, world coordinate in this registration
    tf_list = []
    for tar_id in range(len(pcs)):
        for src_id in range(tar_id+1, len(pcs)):
            pc_src, pc_tar = pcs[src_id], pcs[tar_id]
            tf, info, correspondence_distance_icp = registration_pairwise(pc_src, pc_tar, voxels_down_sample, flag_vis_reg)
            if src_id == tar_id+1:
                odometry = np.dot(odometry, tf)
                pose_graph.nodes.append(o3.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
                pose_graph.edges.append(o3.pipelines.registration.PoseGraphEdge(src_id,
                                                                                tar_id,
                                                                                tf,
                                                                                info,
                                                                                uncertain=False))
            else:
                pose_graph.edges.append(o3.pipelines.registration.PoseGraphEdge(src_id,
                                                                                tar_id,
                                                                                tf,
                                                                                info,
                                                                                uncertain=True))
            option = o3.pipelines.registration.GlobalOptimizationOption(
                max_correspondence_distance=correspondence_distance_icp,
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


def reg_onebyone(pcs_src, voxels_down_sample, flag_vis_reg=False):
    tf_list = [np.eye(4)]
    for tar_id in range(len(pcs_src)-1):
        src_id = tar_id + 1
        pc_src, pc_tar = pcs_src[src_id], pcs_src[tar_id]
        odometry, *_ = registration_pairwise(pc_src, pc_tar, voxels_down_sample, flag_vis_reg)
        tf_list.append(np.matmul(tf_list[-1], odometry))
    return tf_list


def registration_pairwise(pc_src, pc_tar, voxels_down_sample, flag_vis_reg):
    tf_init, information_icp, max_correspondence_distance = np.identity(4), None, 0
    for voxel_down_sample in voxels_down_sample:
        max_correspondence_distance = voxel_down_sample * 2
        pc_src_down_sample, pc_tar_down_sample = pc_src.voxel_down_sample(voxel_down_sample), pc_tar.voxel_down_sample(
            voxel_down_sample)
        radius_normal = voxel_down_sample * 2
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
                                     window_name='last')
        tf_init = icp_result.transformation
        if flag_vis_reg:
            draw_registration_result(source=pc_src_down_sample, target=pc_tar_down_sample, transformation=tf_init,
                                     window_name='current')
    tf_icp = tf_init
    correspondence_distance_icp = max_correspondence_distance
    return tf_icp, information_icp, correspondence_distance_icp


def filter_pc(para, pc):
    # voxel_size = para['voxel_size']
    # k_means = para['k_means']
    # pc_down = pc.voxe_down_sample(voxel_size)
    # points = pc_down.points

    # k-means
    # pick a center
    # get rid of offset points
    # rectify orientation
    # cutoff out range points
    x_max_y_max_z_max, x_min_y_min_z_min = para['upper_bound'], para['lower_bound']
    x_max_y_max_z_max, x_min_y_min_z_min = np.asarray(x_max_y_max_z_max), np.asarray(x_min_y_min_z_min)
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
    comp = o3.geometry.PointCloud()
    points = np.asarray(pc.points)
    points = points[np.all(points < x_max_y_max_z_max, axis=-1)]
    points = points[np.all(points > x_min_y_min_z_min, axis=-1)]
    comp.points = o3.utility.Vector3dVector(points)

    print('After crop', comp)
    # o3.visualization.draw_geometries([vol_pc])
    # o3.visualization.draw_geometries([pc])
    # o3.visualization.draw_geometries([comp])
    return comp


def main():
    # input file path
    model_dir_path = 'data/human_models/head_models/model_women/'
    # model_dir_path = 'data/TUW_TUW_models/TUW_models/bunny/'

    # output file name

    # start seg
    combine_pcs(model_dir_path)


if __name__ == '__main__':
    main()