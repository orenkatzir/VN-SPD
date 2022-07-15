from __future__ import print_function
import os
import numpy as np

import h5py
from data.base_dataset import BaseDataset


class ShapenetDataset(BaseDataset):
    def __init__(self, opt):
        freeze_data = False
        require_normal = False
        num_pts = opt.npoints
        data_dump_folder = opt.dataroot
        jitter_type = "None"

        if freeze_data:
            # freeze test data
            print("Freezing data. No randomness!")
            transform_fn = os.path.join(
                data_dump_folder, "random_T_100k_uni-180-0.2.h5")
            with h5py.File(transform_fn, "r") as f:
                self.test_transforms = np.asarray(f["transform"])
                # self.test_pose1s = np.asarray(f["pose1"])
            self.fix_idx = True
        else:
            self.test_transforms = None
            self.fix_idx = False

        self.jitter_type = jitter_type
        self.num_pts = num_pts
        self.require_normal = require_normal
        # data_dump_folder = os.path.join(
        #     data_dump_folder, "ShapeNetAtlasNetH5"
        # )

        # get h5path
        opt.phase = 'valid' if opt.phase == 'test' else opt.phase
        fn_cat = os.path.join(data_dump_folder, f"{opt.phase}_cat.txt")
        #fn_cat = os.path.join(data_dump_folder, f"cat.txt")
        h5path = [line.rstrip() for line in list(open(fn_cat, "r"))]
        class_choice = opt.class_choice
        if not 'all' in class_choice:
            selectedh5path = []
            for i, single_class_choice in enumerate(class_choice):
                if single_class_choice == 'airplane':
                    single_class_choice = 'plane'
                id = h5path.index(f'{single_class_choice}.h5')
                selectedh5path.append(h5path[id])
            h5path = selectedh5path
        self.h5path = [
            os.path.join(data_dump_folder, opt.phase, h5path_) for h5path_ in h5path]
        self.dataset = [h5py.File(path, "r") for path in self.h5path]

        # get len and cumsum
        lens = [len(d["pcd"]["point"]) for d in self.dataset]
        self.len = np.sum(lens)
        self.cuts = np.cumsum(lens)

        # close file
        for i, d in enumerate(self.dataset):
            d.close()
            self.dataset[i] = None

    def preprocess(self, xyz, other=None):
        # xyz: Nx3, other(NxC)
        # Sampling
        if self.fix_idx:
            idx = np.arange(self.num_pts)
        else:
            idx = np.random.choice(
                len(xyz), self.num_pts, replace=False)
        xyz = xyz[idx]
        if other is None:
            return xyz
        else:
            other = other[idx]
            return xyz, other

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        # print("Loading", item)
        d_index = np.searchsorted(self.cuts, item + 1)  # Figure out the dataset index
        original_item = item
        item = (item - self.cuts[d_index - 1]) if d_index > 0 else item
        if self.dataset[d_index] is None:
            self.dataset[d_index] = h5py.File(self.h5path[d_index], "r")
        xyz = np.asarray(
            self.dataset[d_index]["pcd"]["point"][str(item)]).astype(np.float32)
        # label = self.dataset[d_index]["pcd"]["point"]

        if self.require_normal:
            normal = np.asarray(
                self.dataset[d_index]["pcd"]["normal"][str(item)]).astype(np.float32)

        # Sampling points
        if self.require_normal:
            xyz_resample, normal_resample = self.preprocess(xyz, normal)
        else:
            xyz_resample = self.preprocess(xyz)

        data = {
            "pc": xyz_resample.astype("float32"),
            # "pc": xz,
            "label": d_index}
        if self.require_normal:
            data["normal"] = normal_resample.astype("float32")
        if self.test_transforms is not None:
            transform = self.test_transforms[original_item]  # 4x4 trans mat
            data["transform"] = transform
            data["item_idx"] = original_item

        return data["pc"].transpose(1,0), data["label"]

    # Dataset preparation code from AtlasNetV2 code


#CHUNK_SIZE = 150
#lenght_line = 60


#def my_get_n_random_lines(path, n=5):
#    MY_CHUNK_SIZE = lenght_line * (n + 2)
#    lenght = os.stat(path).st_size
#    with open(path, 'r') as file:
#        file.seek(random.randint(400, lenght - MY_CHUNK_SIZE))
#        chunk = file.read(MY_CHUNK_SIZE)
#        lines = chunk.split(os.linesep)
#        return lines[1:n + 1]


# This code was used to generate the h5 files
# def extract_h5(mode="train"):
#     data_dump_folder = "data_dump"
#     train = mode == "train"
#     data_dump_folder = os.path.join(
#         data_dump_folder, "ShapeNet"
#     )
#     rootimg = os.path.join(
#         data_dump_folder, "ShapeNet/ShapeNetRendering")
#     rootpc = os.path.join(
#         data_dump_folder, "customShapeNet")
#
#     idx = 0
#     catfile = os.path.join(
#         data_dump_folder, 'synsetoffset2category.txt')
#     cat = {}
#     meta = {}
#     with open(catfile, 'r') as f:
#         for line in f:
#             ls = line.strip().split()
#             cat[ls[0]] = ls[1]
#     empty = []
#     for item in cat:
#         dir_img = os.path.join(rootimg, cat[item])
#         fns_img = sorted(os.listdir(dir_img))
#
#         try:
#             dir_point = os.path.join(rootpc, cat[item], 'ply')
#             fns_pc = sorted(os.listdir(dir_point))
#         except:
#             fns_pc = []
#         fns = [val for val in fns_img if val + '.points.ply' in fns_pc]
#         print('category ', cat[item], 'files ' + str(len(fns)), len(fns) / float(len(fns_img)), "%"),
#         if train:
#             fns = fns[:int(len(fns) * 0.8)]
#         else:
#             fns = fns[int(len(fns) * 0.8):]
#
#         if len(fns) != 0:
#             meta[item] = []
#             for fn in fns:
#                 objpath = "./data/ShapeNetCorev2/" + cat[item] + "/" + fn + "/models/model_normalized.ply"
#                 meta[item].append((os.path.join(dir_img, fn, "rendering"), os.path.join(dir_point, fn + '.points.ply'),
#                                    item, objpath, fn))
#         else:
#             empty.append(item)
#     for item in empty:
#         del cat[item]
#     idx2cat = {}
#     size = {}
#     i = 0
#     data_dump_dir = f"data_dump/ShapeNetAtlasNetH5/{mode}"
#     if not os.path.exists(data_dump_dir):
#         os.makedirs(data_dump_dir)
#
#     for item in cat:
#         datapath = []
#         idx2cat[i] = item
#
#         size[i] = len(meta[item])
#         i = i + 1
#         # for fn in self.meta[item]:
#         l = int(len(meta[item]))
#         for fn in meta[item][0:l]:
#             datapath.append(fn)
#
#         vs = []
#         ns = []
#         for fn in datapath:
#             # read ply file
#             v, _, n, _ = pcu.read_ply(fn[1])  # return xyz and normal
#             if len(v) != 30000:
#                 print(f"len: {len(v)}, fn: {fn}")
#             vs += [v.astype(np.float32)]
#             ns += [n.astype(np.float32)]
#         # vs = np.stack(vs).astype(np.float32)
#         # ns = np.stack(ns).astype(np.float32)
#         h5fn = os.path.join(data_dump_dir, f"{item}.h5")
#         print(f"writing h5 for: {item}_{mode}, {len(vs)}")
#         with h5py.File(h5fn, "w") as f:
#             pcd = f.create_group("pcd")
#             pcd_point = pcd.create_group("point")
#             pcd_normal = pcd.create_group("normal")
#             for i in range(len(vs)):
#                 pcd_point[str(i)] = vs[i]
#                 pcd_normal[str(i)] = ns[i]


if __name__ == "__main__":
    for mode in ["train", "valid"]:
        print("Please uncomment code and run extract_h5")
        #extract_h5(mode)