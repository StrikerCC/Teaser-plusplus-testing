# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 10/14/21 5:21 PM
"""
import open3d as o3
from registration import ransac_icp_helper, fpfh_teaser_icp_helper
from vis import draw_registration_result
from dataset import Reader
from tester import tester
from registration import registrations


# def main():
#
#     # src_path = './'
#     # tgt_path = './'
#
#     # src_pc, tgt_pc = o3.io.read_point_cloud(src_path), o3.io.read_point_cloud(tgt_path)
#     input_path = './data/model_man10/'
#     input_views_json_path = input_path + 'data.json'
#
#     '''plot init pose'''
#     draw_registration_result(source=src_pc, target=tgt_pc, window_name='init')
#
#     '''reg'''
#     tf = ransac_icp()
#
#     '''plot reg result'''
#     draw_registration_result(source=)


def main():
    flag_test = True

    '''input dir, files'''
    input_path = './data/model_man10/'
    input_views_json_path = input_path + 'data.json'

    dataset = Reader()
    dataset.read(input_views_json_path)

    src_path = './data/human_models/head_models/model_man/3D_model_face.pcd'

    '''reg pc from mr with model pc'''
    # VOXEL_SIZE_GLOBAL = (4,)
    # VOXEL_SIZE_LOCAL = (4, 3, 1, 0.5, 0.2, 0.1)
    src_pc = o3.io.read_point_cloud(src_path)
    # result_global_mr, result_local_mr, time_global_mr, time_local_mr = ransac_icp_helper(dataset[0]['pc_model'], src_path, VOXEL_SIZE_GLOBAL, VOXEL_SIZE_LOCAL)
    # tf_mr_model = result_local_mr.transformation

    '''reg pc from mr with target pc'''
    VOXEL_SIZE_GLOBAL = 6
    VOXEL_SIZE_LOCAL = (5, 3, 1, 0.4)

    for i in range(len(dataset)):
        # if i in {0, 1, 2, 5, 6, 8, 9}:
        #     continue
        print(i)
        src_pc, tgt_pc = src_pc, dataset[i]['pc_artificial']
        # tgt_pc, src_pc = src_pc, tgt_pc
        result_global, result_local, time_global, time_local = ransac_icp_helper(src_pc, tgt_pc, VOXEL_SIZE_GLOBAL, VOXEL_SIZE_LOCAL)
        tf_global, tf_final = result_global.transformation, result_local.transformation
        # tf_global, result_local, time_global, time_local = fpfh_teaser_icp_helper(src_pc, tgt_pc, VOXEL_SIZE_GLOBAL, VOXEL_SIZE_LOCAL)
        # tf_global, tf_final = tf_global, result_local.transformation

        # print(time_global + time_local)
        print(src_pc)
        print(tgt_pc)
        print('RANSAC fitness', result_global.fitness)
        print('RANSAC inlier_rmse', result_global.inlier_rmse)

        print('ICP fitness', result_local.fitness)
        print('ICP inlier_rmse', result_local.inlier_rmse)

        src_pc_show, tgt_pc_show = src_pc.voxel_down_sample(VOXEL_SIZE_GLOBAL), tgt_pc.voxel_down_sample(
            VOXEL_SIZE_GLOBAL)
        draw_registration_result(source=src_pc_show, target=tgt_pc_show, transformation=tf_global, window_name='reg')
        src_pc_show, tgt_pc_show = src_pc.voxel_down_sample(VOXEL_SIZE_LOCAL[1]), tgt_pc.voxel_down_sample(
            VOXEL_SIZE_LOCAL[1])
        draw_registration_result(source=src_pc_show, target=tgt_pc_show, transformation=tf_final, window_name='reg')

    '''compute error'''


if __name__ == '__main__':
    main()
