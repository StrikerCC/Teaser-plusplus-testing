# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 10/12/21 4:54 PM
"""
import copy
import numpy as np
import open3d as o3d
import vedo
import vedo.applications


def seg_mesh():
    mesh = o3d.io.read_triangle_mesh('./data/human_models/head_models/model_man/skin.stl')

    '''mesh cluster'''
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as mm:
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)

    a = mesh.adjacency_list
    print(a)

    '''filter inner mesh'''
    mesh_0 = copy.deepcopy(mesh)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < 10000
    mesh_0.remove_triangles_by_mask(triangles_to_remove)

    print(mesh_0)
    o3d.visualization.draw_geometries([mesh_0])

    mesh_1 = copy.deepcopy(mesh)
    largest_cluster_idx = cluster_n_triangles.argmax()
    triangles_to_remove = triangle_clusters != largest_cluster_idx
    mesh_1.remove_triangles_by_mask(triangles_to_remove)

    print(mesh_1)
    o3d.visualization.draw_geometries([mesh_1])

    '''sampling to point cloud'''
    pcd = mesh_1.sample_points_uniformly(number_of_points=5000)
    o3d.visualization.draw_geometries([pcd])


def main():
    output_pc_file_path = './data/human_models/head_models/model_man/3D_model_from_mr.ply'
    vol = vedo.load('./data/human_models/head_models/model_man/722brain/')
    spacing = vol.spacing()
    print('dicom Spacing: ', spacing)

    vedo.show(vol, axes=1)

    plt = vedo.applications.IsosurfaceBrowser(vol)
    plt.show()

    threshold = [-196.294]
    isos = vol.isosurface(threshold=threshold)

    splitem = isos.splitByConnectivity(maxdepth=10)[0]
    pc = splitem.points()

    vedo.show(splitem)
    vedo.show(isos, __doc__).addCutterTool(isos)

    # vedo.io.write(splitem, output_pc_file_path)


if __name__ == '__main__':
    main()
