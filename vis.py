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
    if target: target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    if target:
        o3.visualization.draw_geometries([source_temp, target_temp], window_name=window_name)
    else:
        o3.visualization.draw_geometries([source_temp], window_name=window_name)
