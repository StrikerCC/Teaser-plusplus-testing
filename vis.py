# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 8/18/21 10:33 AM
"""
import copy
import numpy as np
import open3d as o3


def draw_registration_result(source, target=None, transformation=np.eye(4), window_name=''):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    if target:
        target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    if target:
        o3.visualization.draw_geometries([source_temp, target_temp], window_name=window_name)
    else:
        o3.visualization.draw_geometries([source_temp], window_name=window_name)


def draw_correspondence(source, target, correspondence, window_name='correspondence'):
    # make line-set between correspondences
    point_set = np.concatenate((np.asarray(source.points),
                                np.asarray(target.points)))
    line_set = correspondence
    line_set[:, 1] += len(correspondence)
    colors = [[0, 1, 0] for i in range(len(line_set))]  # lines are shown in green
    line_set = o3.geometry.LineSet(
        points=o3.utility.Vector3dVector(point_set),
        lines=o3.utility.Vector2iVector(line_set),
    )
    line_set.colors = o3.utility.Vector3dVector(colors)
    # draw
    o3.visualization.draw_geometries([source, target, line_set], window_name=window_name)
    return 1


def bounding_box_3d(xyz_min, xyz_max):
    pass
