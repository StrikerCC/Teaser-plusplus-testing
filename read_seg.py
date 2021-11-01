import numpy as np
import open3d as o3


def get_point_with_ins_obj(file_path):
    isnum = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-'}
    f = open(file_path, 'r')
    print('Getting', f.name)

    points_per_ins = {}

    '''take points per instance'''
    # for i in range(100000):
    while True:
        line = f.readline()
        if not line:
            break

        '''ignore non data part'''
        if not line[0] in isnum:
            print(line[:-1])
            continue

        '''get xyz data'''
        xyz_ins_obj = [float(num) for num in line.split(' ')]
        assert len(xyz_ins_obj) == 6, 'from ' + line + ' to ' + str(xyz_ins_obj)

        '''sort points per instance'''
        ins = xyz_ins_obj[-2]
        xyz = xyz_ins_obj[:3]
        if ins in points_per_ins.keys():
            points_per_ins[ins].append(xyz)
        else:
            points_per_ins[ins] = [xyz]
    f.close()

    print(' Get', len(points_per_ins.keys()), 'instances', )
    for key in points_per_ins.keys():
        print('     instance', key, 'has', len(points_per_ins[key]), 'points')
    return points_per_ins


def build_pc_per_ins(points_per_ins):
    pc_list = {}
    for ins in points_per_ins.keys():
        pc = o3.geometry.PointCloud()
        pc.points = o3.utility.Vector3dVector(np.asarray(points_per_ins[ins]))
        pc_list[ins] = pc
    return pc_list


def main():
    file_path = './data/human_models/head_models/model_man/3D_model_from_mr_seg.pcd'
    points_per_ins = get_point_with_ins_obj(file_path)
    pc_per_ins = build_pc_per_ins(points_per_ins)
    for ins in pc_per_ins:
        # print('     instance', ins, 'has', len(pc_per_ins[ins].points), 'points')
        o3.visualization.draw_geometries([pc_per_ins[ins]])
        if int(ins) == 5:
            o3.io.write_point_cloud('./data/human_models/head_models/model_man/3D_model_face.pcd', pc_per_ins[ins])


if __name__ == '__main__':
    main()
