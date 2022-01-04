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
from vedo.mesh import merge


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

    mesh_out = vedo.Mesh()
    pc_out = vedo.Points()
    isosurfaces = []

    '''crop neck'''
    vol.crop(left=0.0, right=0.0, back=0.0, front=0.0,
             bottom=0.0, top=0.35)

    spacing = vol.spacing()
    print('dicom Spacing: ', spacing)
    # vedo.show(vol, axes=1)

    '''build surface layer by layer'''
    slice_step = 0.1
    center_global_x = 0.0
    center_global_y = 0.0
    for frac_slice_bottom in np.arange(0.0, 0.8, slice_step):
        frac_slice_top = 1.0 - frac_slice_bottom - slice_step
        print('frac start at ', frac_slice_bottom)
        slice = vedo.load('./data/human_models/head_models/model_man/722brain/')
        slice.crop(left=0.0, right=0.0, back=0.0, front=0.0,
                         bottom=frac_slice_bottom, top=frac_slice_top)


        # plt = vedo.applications.IsosurfaceBrowser(slice)
        # plt.show()

        threshold = [-196.294]
        isos = slice.isosurface(threshold=threshold)

        splitems = isos.splitByConnectivity(maxdepth=5)

        '''take the most ouuter surface'''
        radius_biggest, splitem_farest = 0, splitems[0]
        for splitem in splitems:
            points = splitem.points()
            radius = np.linalg.norm(points - points.mean(axis=0), axis=-1).mean(axis=0)
            if radius_biggest > radius:
                radius_biggest = radius
                splitem_farest = splitem

        '''trim some inner surface using sphere'''
        # center = splitem_farest.centerOfMass()
        # if center_global_x == 0.0:
        #     center_global_x = center[0]
        #     center_global_y = center[1]
        # else:
        #     center[0] = center_global_x
        #     center[1] = center_global_y
        #
        # radius = np.linalg.norm((np.asarray(splitem_farest.points()) - center), axis=-1)
        # radius = radius.mean() * 0.8
        #
        # s1 = vedo.Sphere(pos=center, r=radius, c="red", alpha=0.5)
        # splitem_farest.boolean("minus", s1)



        isosurfaces.append(splitem_farest)

        # pc = splitem.points()

        # vedo.show(splitem_farest)

    face = merge(isosurfaces)

    vedo.show(face)

    '''filter inner points'''
    face_pc = o3d.geometry.PointCloud()

    face_pc.points = o3d.utility.Vector3dVector(face.points())

    o3d.visualization.draw_geometries([face_pc])

    o3d.io.write_point_cloud('./data/human_models/head_models/model_man/3D_model_face_from_mr.pcd', face_pc)
    # vedo.io.write(splitem, output_pc_file_path)


def testing():
    """Boolean operations with Meshes"""

    vedo.settings.useDepthPeeling = True

    # declare the instance of the class
    plt = vedo.Plotter(shape=(2, 2), interactive=0, axes=3)

    # build to sphere meshes
    s1 = vedo.Sphere(pos=[-0.7, 0, 0], c="red", alpha=0.5)
    s2 = vedo.Sphere(pos=[0.7, 0, 0], c="green", alpha=0.5)
    plt.show(s1, s2, __doc__, at=0)

    # make 3 different possible operations:
    b1 = s1.boolean("intersect", s2).c('magenta')
    plt.show(b1, "intersect", at=1, resetcam=False)

    b2 = s1.boolean("plus", s2).c("blue").wireframe(True)
    plt.show(b2, "plus", at=2, resetcam=False)

    # b3 = s1.boolean("minus", s2).computeNormals().addScalarBar(c='white')
    b3 = s1.boolean("minus", s2)
    plt.show(b3, "minus", at=3, resetcam=False)

    vedo.interactive().close()


if __name__ == '__main__':
    main()
    # testing()
