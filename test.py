import torch
import numpy as np
import sys
import os
import h5py
import copy
from pytorch3d.loss import chamfer_distance
from tqdm import tqdm

from data import create_dataset
from models import create_model
from options.test_options import TestOptions
from util import pc_utils


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
EPS = 1e-10

torch.manual_seed(0)
np.random.seed(seed=122222)


def load_transformation(transformation_path, device):
    fn = os.path.join(transformation_path)
    with h5py.File(fn, "r") as f:
        transform = np.asarray(f["transform"])

    return torch.FloatTensor(transform).to(device)

def main(opt):
    # hard-code some parameters for test
    opt.split = 'test'
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.eval()

    if opt.transform_path != '':
        presaved_trans = load_transformation(opt.transform_path)
    else:
        presaved_trans = None

    num_exp = opt.num_stability_exp
    with torch.no_grad():
        consistent_array = []
        rot_mats = []
        for i, data in enumerate(tqdm(dataset, total=round(float(len(dataset)) / opt.batch_size))):
            model.set_input(data, use_rand_trans=False)  # unpack data from data loader

            out = model.test()  # run inference
            pc_at_canonic = out[0]
            rot_mat, t_vec = out[1]

            rot_mats.append(pc_utils.to_rotation_mat(rot_mat.detach()))

        rot_mats = torch.cat(rot_mats, dim=0)
        rot_mat_mean = pc_utils.to_rotation_mat(torch.mean(rot_mats, dim=0, keepdim=True))
        dists = pc_utils.cal_angular_metric(rot_mats, rot_mat_mean)

        consistent_array.append(torch.sqrt(torch.mean(dists ** 2)).item())

        print("Consistency: ", np.mean(consistent_array))

        #Stability:
        stab_arr = []
        for i, data in enumerate(tqdm(dataset, total=round(float(len(dataset))/opt.batch_size))):
            if (i *  opt.batch_size) > opt.num_test:
                break
            data_ = copy.deepcopy(data)
            rot_mats = []
            for j in range(num_exp):
                if not presaved_trans is None:
                    input_rot = presaved_trans[j][:3,:3].unsqueeze(0)
                    data_[0] = torch.bmm(data[0].transpose(1, 2).cuda(), input_rot).transpose(1, 2)
                    in_points, _ = model.set_input(data_, use_rand_trans=True)
                else:
                    in_points, trot = model.set_input(data_, use_rand_trans=True)
                    input_rot = trot.get_matrix()[:,:3,:3]

                out = model.test()  # run inference
                pc_at_canonic = out[0]
                rot_mat, t_vec = out[1]

                rot_mats.append(torch.matmul(pc_utils.to_rotation_mat(rot_mat.detach()), input_rot.transpose(1,2)))

            rot_mats = torch.cat(rot_mats, dim=0)
            rot_mat_mean = pc_utils.to_rotation_mat(torch.mean(rot_mats, dim=0, keepdim=True))
            dists = pc_utils.cal_angular_metric(rot_mats, rot_mat_mean)
            obj_stability = torch.sqrt(torch.mean(dists ** 2)).item()
            stab_arr.append(obj_stability)

        print("Stability: ", np.mean(stab_arr))



if __name__ == '__main__':
    args = TestOptions().parse()  # get training options
    main(args)