# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 8/19/21 5:22 PM
"""
import glob
import json
import os
import numpy as np

from dataset import Reader
from registration import registrations, VOXEL_SIZE_GLOBAL, VOXEL_SIZE_LOCAL
from vis import draw_registration_result, draw_correspondence

statistics_testing = {global_registration: {'method': global_registration.__name__,
                                            '#case': 0,
                                            '#failure': 0,
                                            'time_global': 0.0,
                                            'time_local': 0.0,
                                            'error_t': 0.0,
                                            'error_o': 0.0,
                                            'index_failure': [],
                                            'tf_failure': [],
                                            'pose_failure': [],
                                            'error_t_failure': [],
                                            'error_o_failure': [],
                                            'voxel_size_reg_failure': [],
                                            'correspondence_set_failure': [],
                                            '#points': []
                                            } for global_registration in registrations}


class tester:
    def __init__(self):
        self.statistics = statistics_testing
        self.dataloader = None
        # self.registration = None

    def start_reg(self, regs, dataloader, result_path=None, show_flag=False):
        self.statistics = statistics_testing
        self.dataloader = dataloader
        if not isinstance(regs, list):
            regs = [regs]
        for i, reg in enumerate(regs):
            voxel_size_global = VOXEL_SIZE_GLOBAL[i]
            voxel_size_local = VOXEL_SIZE_LOCAL[i]
            statistic = self.statistics[reg]
            reg(self.dataloader, voxel_size_global, voxel_size_local, statistic, show_flag=show_flag)
        self.__report_log(result_path)
        if result_path:
            self.__report_out(self.statistics)

    def __report_out(self, statistics):
        regs = statistics.keys()
        for i, reg in enumerate(regs):
            statistic = statistics[reg]
            print('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            print(reg)
            if statistic['#case'] == 0:
                print('no test case')
                statistic['#case'] = 1
            print('Translation rms', statistic['error_t'])
            print('Orientation rms', statistic['error_o'])
            print('Time average', (statistic['time_global'] + statistic['time_local']))
            print('Failure percent', statistic['#failure'] / statistic['#case'])
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
        print('Done')

    def __report_log(self, output_path):
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        if not output_path[-1] == '/':
            output_path += '/'
        prefix, post_fix = 'result', '.json'
        infix = 0
        result_list_previous = glob.glob(output_path+'*.json')
        if len(result_list_previous):
            find_index = lambda file_path: file_path[len(output_path+prefix):-len(post_fix)]
            index_list_previous = [find_index(result_previous) for result_previous in result_list_previous]
            index_list_previous = sorted(index_list_previous)
            infix = int(index_list_previous[-1]) + 1
        result_json_path = output_path + prefix + str(infix) + post_fix

        # make a shallow copy of self.statistics, since key <function> is not a string, and cannot be writen to json file
        statistics_result = {}
        for reg in self.statistics.keys():
            statistics_result[reg.__name__] = self.statistics[reg]
        with open(result_json_path, 'w') as f:
            json.dump(statistics_result, f)
        return 1

    def peek_failures(self, result_json_path, dataloader, flag_vis=False):
        # if not result_json_path or not dataloader:
        #     assert self.statistics and self.dataloader
        #     statistics_result = self.statistics
        #     dataloader = self.dataloader
        # else:
        with open(result_json_path, 'r') as f:
            statistics_result = json.load(f)
        dataloader = dataloader
        regs = statistics_result.keys()
        for i_reg, reg in enumerate(regs):
            statistic = statistics_result[reg]
            print('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            print(reg)
            if statistic['#failure'] == 0:
                print('No failure case')
                statistic['#case'] = 1
            print('Translation rms', statistic['error_t'])
            print('Orientation rms', statistic['error_o'])
            print('Time average', (statistic['time_global'] + statistic['time_local']))
            print('Failure percent', statistic['#failure'] / statistic['#case'])
            print('Showing failing case')
            for i_failure, index_fail in enumerate(statistic['index_failure']):
                source = dataloader[index_fail]
                voxel_size = statistic['voxel_size_reg_failure'][i_failure]
                pc_src, pc_tar = source['pc_model'], source['pc_artificial']
                pc_src, pc_tar = pc_src.voxel_down_sample(voxel_size=voxel_size), pc_tar.voxel_down_sample(voxel_size=voxel_size)
                num_points = statistic['#points'][i_failure]
                correspondence_set = np.asarray(statistic['correspondence_set_failure'][i_failure])
                tf_failure = np.asarray(statistic['tf_failure'][i_failure])
                # pose_gt = np.asarray(statistic['pose_failure'])[i_failure]
                print('     instance', source['instance'],
                      '\n     # points', num_points,
                      '\n     # correspondence', len(correspondence_set),
                      '\n     voxel_size', source['voxel_size'],
                      '\n     angle', source['angle'],
                      '\n     sigma', source['sigma'],
                      '\n     plane', source['plane'])
                print(statistic['error_t_failure'][i_failure], statistic['error_o_failure'][i_failure])

                # vis
                if flag_vis:
                    draw_registration_result(source=pc_src, target=pc_tar, window_name='Initial pose of #' + str(i_failure) + '/'+str(len(statistic['index_failure']))+' using '+reg)
                    draw_correspondence(pc_src, pc_tar, correspondence_set)     # # make line-set between correspondences
                    draw_registration_result(source=pc_src, target=pc_tar, transformation=tf_failure, window_name='Final reg of #' + str(i_failure) + '/'+str(len(statistic['index_failure']))+' using '+reg)
                    # skip the rest of failure case
                    if i_failure >= 3:
                        break

            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
            print('Failure Enumeration Done')

        for i_reg, reg in enumerate(regs):
            statistic = statistics_result[reg]
            print('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            print(reg)
            if statistic['#failure'] == 0:
                print('No successful test case')
                statistic['#case'] = 1
            print('Translation rms', statistic['error_t'])
            print('Orientation rms', statistic['error_o'])
            print('Time average', (statistic['time_global'] + statistic['time_local']))
            print('Failure percent', statistic['#failure'] / statistic['#case'])
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
            print('Summary Done')
        return 1


def main():
    flag_test = True

    # sample_path = './data/TUW_TUW_models/TUW_models/'
    # # input_path = './data/TUW_TUW_data/'
    # input_path = 'data/TUW_TUW_data_uniform_size/'
    # input_json_path = input_path + 'data.json'
    #
    # result_path = './result/TUW_TUW_test/'

    # sample_path = './data/human_models/head_models/'
    input_path = './data/human_data/'
    input_views_json_path = input_path + 'data.json'
    input_model_path = './data/human_models/head_models/model_women/3D_model.pcd'
    result_path = './result/human_head_test/'
    dl = Reader()
    dl.read(input_views_json_path)

    reg_tester = tester()
    if flag_test:
        reg_tester.start_reg(regs=registrations, dataloader=dl, result_path=result_path, show_flag=False)

    # show latest reg failure
    json_names = sorted(os.listdir(result_path), key=lambda x: int(x[6:][:-5]))
    json_name = json_names[-1]
    result_json_path = result_path + json_name
    reg_tester.peek_failures(result_json_path, dl, flag_vis=True)


if __name__ == '__main__':
    main()
