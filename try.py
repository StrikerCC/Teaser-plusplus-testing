# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 8/27/21 3:27 PM
"""
import copy

import numpy as np
import open3d as o3
import transforms3d as t3d
from vis import draw_registration_result


def icp():
    voxel_size_local = 2
    NOISE_BOUND = 1000

    sample_path = './data/human_models/head_models/model_women/3D_model.pcd'
    tf_init = np.identity(4)
    tf_init[:3, :3] = t3d.euler.euler2mat(*np.deg2rad([10.0, 0.0, 0.0]))
    tf_init[:3, 3] = 10

    pc_src = o3.io.read_point_cloud(sample_path)
    print(pc_src)
    draw_registration_result(pc_src)

    pc_tar = copy.deepcopy(pc_src)
    pc_tar.transform(tf_init)

    draw_registration_result(pc_src, pc_tar)

    pc_src_local, pc_tar_local = pc_src.voxel_down_sample(voxel_size_local), pc_tar.voxel_down_sample(voxel_size_local)
    radius_normal = voxel_size_local * 2
    pc_src_local.estimate_normals(o3.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    pc_tar_local.estimate_normals(o3.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    icp_sol = o3.pipelines.registration.registration_icp(
        pc_src, pc_tar, NOISE_BOUND, np.identity(4),
        o3.pipelines.registration.TransformationEstimationPointToPoint(),
        o3.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))

    # icp_sol = o3.registration.registration_icp(
    #     pc_src, pc_tar, NOISE_BOUND, np.identity(4),
    #     o3.registration.TransformationEstimationPointToPoint(),
    #     o3.registration.ICPConvergenceCriteria(max_iteration=100))

    T_icp = icp_sol.transformation

    print(T_icp)
    draw_registration_result(pc_src, pc_tar, T_icp)


def main():
    icp()

if __name__ == '__main__':
    main()