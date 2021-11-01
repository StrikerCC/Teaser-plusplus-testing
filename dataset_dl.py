# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 8/13/21 5:44 PM
"""

import copy
import glob
import os
# import glob
import shutil
import numpy as np
import json
import threading

import open3d as o3
import transforms3d as t3d
from vis import draw_registration_result


class Dataset:
    def __init__(self):
        self.flag_show = False

        self.voxel_sizes = (3,)     # mm
        self.plane_sizes = (0.3,)
        self.Gaussian_sigma_mm = (0.01, 0.02)   # mm
        self.n_move = 100
        self.translation_rg_factor = (-3.1, 3.1)
        # self.translation_rg_factor = (-0.0, 0.0)
        # self.rotation_reg = (-180.0, 180.0)
        self.rotation_reg = (-0.0, 0.0)

        self.instance_dir_paths = ['human_models/head_models/model_man/']
        self.instance_src_paths = ['3D_model.pcd']
        self.instance_view_dir_paths = ['views/']

        print('Total',
              (
                  len(self.instance_dir_paths) * len(self.voxel_sizes) *
                  len(self.plane_sizes) if len(self.plane_sizes) > 0 else 1 *
                  len(self.Gaussian_sigma_mm) if len(self.Gaussian_sigma_mm) > 0 else 1 *
                                                                                      self.n_move
               ), 'pc will be generated. \n',
              len(self.instance_dir_paths), 'instances in account.\n',
              len(self.voxel_sizes), 'voxel sizes\n',
              len(self.plane_sizes), 'plane sizes\n',
              len(self.Gaussian_sigma_mm), 'noises\n',
              self.n_move, ' moves\n',
              len(self.voxel_sizes) * len(self.plane_sizes) *
              len(self.Gaussian_sigma_mm) * self.n_move, 'combo for each instance.')

        self.data_template = {'pc_src': None,
                              'pc_tgt': None, 'pc_view': None, 'instance': None,
                              'scale': 1, 'unit': 'mm', 'voxel_size': None,
                              'tf_view_2_tgt': [], 'tf_src_2_tgt': None, 'tf_src_2_view': None,
                              'sigma': None,
                              'plane': None}

        # def read_instance(self, dir_path):
    #     self.dir_path_read = dir_path
    #     # instances = os.listdir(dir_path)
    #     instances = self.instances
    #     self.file_paths = [dir_path + instance + '/3D_model.pcd' for instance in instances]


class ExeThread(threading.Thread):
    def __init__(self, thread_id, func, args):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.func = func
        self.args = args

    def run(self):
        print(self.thread_id, '::Creating artificial point cloud start')
        self.func(*self.args)
        print(self.thread_id, '::Creating artificial point cloud done')


class Writer(Dataset):
    def __init__(self):
        Dataset.__init__(self)
        self.src_file_paths = []  #
        self.view_file_paths = []
        self.pose_file_paths = []

        self.filename_len = 6
        self.output_format = 'pcd'
        self.meter_2_mm = True
        self.relative_path = False

    def write(self, sample_dir_path, output_dir_path, json_path, num_thread=4):
        # setup output path and file
        if not os.path.isdir(output_dir_path):
            os.makedirs(output_dir_path)
        else:
            shutil.rmtree(output_dir_path)
            os.makedirs(output_dir_path)

        # self.file_paths = [sample_dir_path + instance + '/3D_model.pcd' for instance in instances]
        '''grape all data'''
        self.src_file_paths, self.view_file_paths, self.pose_file_paths = [], [], []
        for instance_dir, instance_src_path in zip(self.instance_dir_paths, self.instance_src_paths):     # grape source file for reg
            self.src_file_paths.append(sample_dir_path + instance_dir + instance_src_path)
        for instance_dir, instance_view_dir in zip(self.instance_dir_paths, self.instance_view_dir_paths):  # grape sensor pc for reg
            view_paths_per_instance, poses_file_paths_per_instance = [], []
            for view_path in glob.glob(sample_dir_path + instance_dir + instance_view_dir + '*.ply'):
                view_paths_per_instance.append(view_path)
            for view_path in glob.glob(sample_dir_path + instance_dir + instance_view_dir + '*.pcd'):
                view_paths_per_instance.append(view_path)
            # find txt per data file
            for data_file_paths_per_instance in view_paths_per_instance:
                poses_file_paths_per_instance.append(data_file_paths_per_instance[:-3] + 'txt')
            self.view_file_paths.append(view_paths_per_instance)
            self.pose_file_paths.append(poses_file_paths_per_instance)

        """why brother registering parameters to build artificial data layer after layer, 
        why not build the point cloud in place. Think about how many point cloud will be here after 5 different 
        downsampling, 4 different plane added, 8 different noise level, it will blow up memory"""
        # register sample pcs and artificial info
        sources = self.__reg_pc(self.src_file_paths, self.view_file_paths, self.pose_file_paths, self.instance_dir_paths)

        # peek model data
        print('Get model point cloud for')
        for i, source in enumerate(sources):
            print(i, source['instance'])

        # register artificial pc parameters
        print('Setting up artificial point cloud for')
        sources = self.__reg_down_sampling(sources)
        # sources = self.__reg_add_plane(sources)
        sources = self.__reg_add_noise(sources)
        sources = self.__reg_add_pose(sources)
        print('Set up artificial point cloud for')

        # assign slice of sources execution for multithreading
        index_sources_thread_list = [int(id_thread * len(sources) / num_thread) for id_thread in range(num_thread)]

        # create artificial pc and save it
        # # create multithreading for pc maker and saver
        exe_thread_list = []
        for id_thread in range(num_thread):
            index_start_sources = index_sources_thread_list[id_thread]
            index_end_sources = index_sources_thread_list[id_thread+1] if id_thread < num_thread-1 else len(sources)
            exe_thread_list.append(ExeThread(thread_id=id_thread, func=self.__exe_all, args=(sources, output_dir_path,
                                                                                             index_start_sources,
                                                                                             index_end_sources)))

        # # start multithreading for pc maker and saver
        for id_thread in range(num_thread):
            exe_thread_list[id_thread].start()

        # # make sure main thread wait for multithreading for pc maker and saver
        for id_thread in range(num_thread):
            exe_thread_list[id_thread].join()

        # make json file to retrieve data
        sources_record = sources  # self.__format_json(sources)
        with open(json_path, 'w') as f:
            json.dump(sources_record, f)

    def __exe_all(self, sources, output_dir_path, index_start_sources=0, index_end_sources=-1):
        # create artificial pc and save it
        instance_cur = None
        for index_source, source in enumerate(sources[index_start_sources:index_end_sources]):
            id_pc_saving = index_start_sources + index_source
            
            # read pc, down sampling, and so on
            self.__load_pc(source, flag_show=self.flag_show)

            # output to screen
            if not instance_cur or not instance_cur == source['instance']:  # notice if instance change
                instance_cur = source['instance']
                print('     ', id_pc_saving, 'iter: Working on', source['instance'], source['pc_tgt'], ' range from',
                      source['pc_tgt'].get_min_bound(), 'to', source['pc_tgt'].get_max_bound())

            self.__exe_down_sampling(source, flag_show=self.flag_show)
            # self.__exe_add_plane(source, flag_show=self.flag_show)
            self.__exe_add_noise(source, flag_show=self.flag_show)
            self.__exe_add_pose(source, flag_show=self.flag_show)
            self.__save_pc(output_dir_path, index=id_pc_saving, source=source)
        print(sources)

    def __reg_pc(self, src_file_paths, view_file_paths_, pose_file_paths_, instances):
        sources = []
        for src_file_path, view_file_paths, pose_file_paths, instance in zip(src_file_paths, view_file_paths_, pose_file_paths_, instances):
            for view_file_path, pose_file_path in zip(view_file_paths, pose_file_paths):
                tf_view_2_upright = np.eye(4)

                assert os.path.exists(pose_file_path), 'Loading tf between view and src for ' + instance + ': ' + pose_file_path + ' not found'
                tf_src_2_view = np.loadtxt(pose_file_path)
                assert tf_src_2_view.shape == (4, 4), 'Expect pose matrix to be 3x3, but get ' + str(tf_src_2_view.shape) + ' instead'

                # transform the point cloud to upright to add plane
                # if 'human' in view_file_path:
                #     tf_view_2_upright[:3, :3] = t3d.euler.euler2mat(*np.deg2rad([180.0, 0.0, 0.0]))
                #     tf_view_2_upright[:3, 3] = 0


                source = copy.deepcopy(self.data_template)
                source['pc_src'] = src_file_path
                source['pc_tgt'] = None
                source['pc_view'] = view_file_path
                source['instance'] = instance
                # source['scale'] = 1.0
                source['unit'] = 'mm'
                # source['voxel_size'] = 0.02
                source['tf_view_2_tgt'].append(tf_view_2_upright)
                # source['tf_src_2_view'] = np.linalg.inv(tf_view_2_src)
                source['tf_src_2_view'] = tf_src_2_view
                # source['sigma'] = 0.02
                # source['plane'] = 0.2

                # # # vis to confirm
                # pc_src = o3.io.read_point_cloud(source['pc_model'])
                # pc_tar = o3.io.read_point_cloud(source['pc_from'])
                # draw_registration_result(source=pc_src, target=pc_tar, transformation=source['pose'])

                sources.append(source)
                '''
                self.data_template = {'pc_src': None,
                              'pc_tgt': None, 'pc_view': None, 'instance': None,
                              'scale': 1, 'unit': 'mm', 'voxel_size': None,
                              'tf_view_2_tgt': [], 'tf_src_2_tgt': None, 'tf_src_2_view': None,
                              'sigma': None,
                              'plane': None}
                '''
            # print(file_path, "\n    max bounds for geometry coordinates", pc.get_max_bound())
        return sources

    def __load_pc(self, source, flag_show=False):
        tf_view_2_upright = source['tf_view_2_tgt'][-1]
        view_path = source['pc_view']
        pc = o3.io.read_point_cloud(filename=view_path)

        print('Mean and Cov', pc.compute_mean_and_covariance())

        source['pc_tgt'] = pc
        pc.scale(scale=source['scale'], center=pc.get_center())
        pc.transform(tf_view_2_upright)

        if flag_show:
            o3.visualization.draw_geometries([pc], window_name='Read ' + str(source['instance']))

    def __save_pc(self, save_dir, index, source):
        if self.output_format == 'ply':
            filename_saving = str(index) + '.ply'
        elif self.output_format == 'pcd':
            filename_saving = str(index) + '.pcd'
        else:
            raise NotImplementedError('Not a known format')

        filename_saving = '0' * (self.filename_len - len(str(index))) + filename_saving
        o3.io.write_point_cloud(save_dir + filename_saving, source['pc_tgt'])
        del source['pc_tgt']  # get rid of the point cloud to save memory
        source['pc_tgt'] = save_dir + filename_saving  # record the artificial pc saving address

    def __reg_down_sampling(self, sources):
        sources_processed = []
        for source in sources:
            for voxel_size in self.voxel_sizes:
                source_ = copy.deepcopy(source)
                source_['voxel_size'] = voxel_size
                sources_processed.append(source_)
        print('Down sampling', self.voxel_sizes)
        assert len(sources_processed) / len(sources) == len(self.voxel_sizes), str(len(sources_processed)) + ' ' + str(
            len(sources)) + ' ' + str(len(self.voxel_sizes))
        return sources_processed

    def __exe_down_sampling(self, source, flag_show=True):
        pc = source['pc_tgt']
        voxel_size = source['voxel_size']
        source['pc_tgt'] = pc.voxel_down_sample(voxel_size)
        if flag_show:
            o3.visualization.draw_geometries([pc], window_name='Initial Setup down to ' + str(voxel_size))

    def __reg_add_pose(self, sources):
        sources_processed = []
        for source in sources:
            for i in range(self.n_move):
                source_ = copy.deepcopy(source)
                translation = ((self.translation_rg_factor[1] - self.translation_rg_factor[0]) * np.random.random(3) +
                               self.translation_rg_factor[0])
                orientation = (self.rotation_reg[1] - self.rotation_reg[0]) * np.random.random((3, 1)) + \
                              self.rotation_reg[0]
                tf_random = np.identity(4)
                tf_random[:3, :3] = t3d.euler.euler2mat(*np.deg2rad(orientation))
                tf_random[:3, 3] = translation
                source_['tf_view_2_tgt'].append(tf_random)
                sources_processed.append(source_)
        print('# of Pose', self.n_move)
        return sources_processed

    def __exe_add_pose(self, source, flag_show=False):
        pc = source['pc_tgt']
        pc_init = copy.deepcopy(pc) if flag_show else None
        # change the pose according to model size
        tf_random = source['tf_view_2_tgt'][-1]
        tp = np.asarray(pc.points)
        rg = tp.max(axis=0) - tp.min(axis=0)        # range
        tf_random[:3, 3] = rg * tf_random[:3, 3]    # rearrange translation
        source['tf_view_2_tgt'][-1] = tf_random

        del tf_random
        pc.transform(source['tf_view_2_tgt'][-1])

        # accumulate transformations to make final pose
        assert len(source['tf_view_2_tgt']) == 2, 'Expect w transformation, but got ' + str(len(source['tf']))
        tf_view_2_tgt = np.eye(4)
        for tf_ in source['tf_view_2_tgt']:
            tf_view_2_tgt = np.matmul(tf_, tf_view_2_tgt)
        source['tf_src_2_tgt'] = np.matmul(tf_view_2_tgt, source['tf_src_2_view'])

        # # # vis to confirm
        # pc_src = o3.io.read_point_cloud(source['pc_model'])
        # pc_tar = source['pc_artificial']
        # draw_registration_result(source=pc_src, target=pc_tar, transformation=source['pose'])

        source['tf_view_2_tgt'] = [tf_.tolist() for tf_ in source['tf_view_2_tgt']]
        source['tf_src_2_tgt'] = source['tf_src_2_tgt'].tolist()
        # source = {}
        source.pop('tf_src_2_view')

        if flag_show:
            # pc_model = o3.io.read_point_cloud(source['pc_model'])
            o3.visualization.draw_geometries([pc_init, pc], window_name='Move at ' + str(tf_view_2_tgt))

    def __reg_add_plane(self, sources):
        sources_processed = []
        for source in sources:
            for plane_size in self.plane_sizes:
                source_ = copy.deepcopy(source)
                source_['plane'] = plane_size
                sources_processed.append(source_)
        print('Adding planes', self.plane_sizes)
        return sources_processed

    def __exe_add_plane(self, source, flag_show=True):
        # if source['instance'] not in self.instances_plane: return
        plane_axis = (0, 1)
        plane_normal_axis = 2
        dis_nearest_neighbor = source['voxel_size']
        plane_size = source['plane']
        pc = source['pc_tgt']
        tp = np.asarray(pc.points)
        # dis_nearest_neighbor = min(np.linalg.norm(tp - tp[0, :], axis=1)[2:])
        rg = 1.5 * (tp.max(axis=0) - tp.min(axis=0))  # range

        # add a plane underneath the model
        # dis_nearest_neighbor = dis_nearest_neighbor / plane_size
        nx = int(plane_size * rg[plane_axis[0]] / dis_nearest_neighbor)
        ny = int(plane_size * rg[plane_axis[1]] / dis_nearest_neighbor)
        x = np.linspace(-plane_size * rg[plane_axis[0]], rg[plane_axis[0]] * plane_size, nx)
        y = np.linspace(-plane_size * rg[plane_axis[1]], rg[plane_axis[1]] * plane_size, ny)
        x, y = np.meshgrid(x, y)

        # make a empty shadow
        mask = np.logical_or(y < - rg[plane_axis[0]] / 8, np.logical_or(x < - rg[plane_axis[0]] / 4, x > rg[plane_axis[0]] / 4))
        x, y = x[mask], y[mask]
        z = np.zeros(y.shape) + tp.min(axis=0)[plane_normal_axis]
        if 'human' in source['pc_src']:
            z -= 135
        plane = np.stack([x, y, z])
        plane = np.reshape(plane, newshape=(3, -1)).T

        # make a hole at the intersection and behind
        model_center = np.mean(tp, axis=0)
        dis = np.linalg.norm(plane - model_center, axis=1)
        mask = dis > rg[plane_axis[1]] / 2 * 0.75
        if 'human' in source['pc_src']:
            mask = dis > rg[plane_axis[1]] / 2 * 1.12
        pc.points = o3.utility.Vector3dVector(np.r_[tp, plane[mask]])

        if flag_show:
            o3.visualization.draw_geometries([pc], window_name='Initial Setup add ' + str(plane_size) + ' plane')

    def __reg_add_noise(self, sources):
        sources_processed = []
        for i, source in enumerate(sources):
            for j, noise_sigma in enumerate(self.Gaussian_sigma_mm):
                source_ = copy.deepcopy(source)
                source_['sigma'] = noise_sigma
                sources_processed.append(source_)
        print('Adding Gaussian noise', self.Gaussian_sigma_mm)
        return sources_processed

    def __exe_add_noise(self, source, flag_show=True):
        noise_sigma = source['sigma']
        pc = source['pc_tgt']
        tp = np.asarray(pc.points)
        pc.points = o3.utility.Vector3dVector(np.r_[tp + noise_sigma * np.random.randn(*tp.shape)])
        if flag_show:
            o3.visualization.draw_geometries([pc], window_name='Initial Setup add ' + str(noise_sigma) + ' sigma')


class Reader(Dataset):
    def __init__(self):
        Dataset.__init__(self)
        self.meter_2_mm = False
        self.sources = None

    def read(self, json_path):
        with open(json_path, 'r') as f:
            self.sources = json.load(f)

        # check unit
        if self.sources[0]['unit'] == 'mm':
            self.meter_2_mm = False
        elif self.sources[0]['unit'] == 'm':
            self.meter_2_mm = True
        else:
            raise AssertionError('Unknown unit ', self.sources[0]['unit'])

    def __len__(self):
        return len(self.sources) if self.sources else 0

    def __getitem__(self, item):
        if not self.sources: return None
        source = copy.deepcopy(self.sources[item])
        # process data
        if isinstance(source['pc_tgt'], str):
            source['pc_tgt'] = o3.io.read_point_cloud(source['pc_tgt'])  # read artificial pc
        if isinstance(source['pc_src'], str):
            # scale_factor = source['scale']
            pc_model = o3.io.read_point_cloud(source['pc_src'])
            # pc_model.scale(scale=scale_factor, center=pc_model.get_center())
            source['pc_src'] = pc_model
        if not isinstance(source['tf_src_2_tgt'], np.ndarray):
            source['tf_src_2_tgt'] = np.asarray(source['tf_src_2_tgt'])
        return source

    def get(self, item):
        if not self.sources:
            return None
        source = copy.deepcopy(self.sources[item])
        return source


def main():
    write = True

    # sample_path = './data/TUW_TUW_models/TUW_models/'
    # output_path = './data/TUW_TUW_data/'
    # output_path = 'data/TUW_TUW_data_uniform_size/'
    # output_path = 'data/TUW_TUW_data_small/'
    # output_json_path = output_path + 'data.json'

    sample_path = './data/'
    output_path = './data/model_man10/'
    output_json_path = output_path + 'data.json'

    ds = Writer()
    # ds.read_instance(data_path)
    if write:
        ds.write(sample_path, output_path, output_json_path, num_thread=4)

    dl = Reader()
    dl.read(output_json_path)

    i = -1
    for i in range(len(dl)):
        data = dl[i]
        pc_src = data['pc_src']
        pc_tgt = data['pc_tgt']
        tf = data['tf_src_2_tgt']

        draw_registration_result(source=pc_src, target=pc_tgt)
        draw_registration_result(source=pc_src, target=pc_tgt, transformation=tf, window_name='gt')
        print(dl[i])

    # for i in range(len(dl)):
    #     data = dl[i]
    #     pc_model = data['pc_model']
    #     pc_artificial = data['pc_artificial']
    #     tf = data['pose']
    #     pc_model_ = copy.deepcopy(pc_model)
    #     pc_model_.transform(tf)

        # draw_registration_result(source=pc_artificial)
        # draw_registration_result(source=pc_artificial, target=pc_model_)
        # print(dl[0])


if __name__ == '__main__':
    main()
