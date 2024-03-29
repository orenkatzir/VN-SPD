# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import math
import numpy as np
import torch
import transforms3d
from scipy.spatial.transform import Rotation
from scipy.special import erfc, erfcinv

class Compose(object):
    def __init__(self, transforms):
        self.transformers = []
        for tr in transforms:
            transformer = eval(tr["callback"])
            parameters = tr["parameters"] if "parameters" in tr else None
            self.transformers.append(
                {"callback": transformer(parameters), "objects": tr["objects"]}
            )  # yapf: disable

    def __call__(self, data):
        for tr in self.transformers:
            transform = tr["callback"]
            objects = tr["objects"]
            rnd_value = np.random.uniform(0, 1)
            # Only for rotation_translation:
            vp_rand = np.random.randn(1, 3)
            wp_rand = np.random.randint(2, size=(1,))
            t_rand = np.random.uniform(-1, 1, (1, 3, 1))
            inv_rand = np.random.rand(1)
            if transform.__class__ in [NormalizeObjectPose]:
                data = transform(data)
            else:
                for k, v in data.items():
                    if k in objects and k in data:
                        if transform.__class__ in [
                            RandomFlip,
                            RandomRotatePoints,
                            RandomScalePoints,
                            RandomMirrorPoints,
                        ]:
                            data[k] = transform(v, rnd_value)
                        elif transform.__class__ in [
                            RandomRotateAnyAxisPoints
                        ]:
                            data[k] = transform(v, vp_rand, wp_rand, inv_rand, t_rand)
                        else:
                            data[k] = transform(v)

        return data


class ToTensor(object):
    def __init__(self, parameters):
        pass

    def __call__(self, arr):
        shape = arr.shape
        if len(shape) == 3:  # RGB/Depth Images
            arr = arr.transpose(2, 0, 1)

        # Ref: https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663/2
        return torch.from_numpy(arr.copy()).float()


class Normalize(object):
    def __init__(self, parameters):
        self.mean = parameters["mean"]
        self.std = parameters["std"]

    def __call__(self, arr):
        arr = arr.astype(np.float32)
        arr /= self.std
        arr -= self.mean

        return arr

class RandomFlip(object):
    def __init__(self, parameters):
        pass

    def __call__(self, img, rnd_value):
        if rnd_value > 0.5:
            img = np.fliplr(img)

        return img


class RandomPermuteRGB(object):
    def __init__(self, parameters):
        pass

    def __call__(self, img):
        rgb_permutation = np.random.permutation(3)
        return img[..., rgb_permutation]


class RandomBackground(object):
    def __init__(self, parameters):
        self.random_bg_color_range = parameters["bg_color"]

    def __call__(self, img):
        img_h, img_w, img_c = img.shape
        if img_c != 4:
            return img

        r, g, b = [
            np.random.randint(
                self.random_bg_color_range[i][0], self.random_bg_color_range[i][1] + 1
            )
            for i in range(3)
        ]
        alpha = (np.expand_dims(img[:, :, 3], axis=2) == 0).astype(np.float32)
        img = img[:, :, :3]
        bg_color = np.array([[[r, g, b]]]) / 255.0
        img = alpha * bg_color + (1 - alpha) * img

        return img


class RandomSamplePoints(object):
    def __init__(self, parameters):
        self.n_points = parameters["n_points"]

    def __call__(self, ptcloud):
        choice = np.random.permutation(ptcloud.shape[0])
        ptcloud = ptcloud[choice[: self.n_points]]

        if ptcloud.shape[0] < self.n_points:
            zeros = np.zeros((self.n_points - ptcloud.shape[0], 3))
            ptcloud = np.concatenate([ptcloud, zeros])

        return ptcloud


class RandomClipPoints(object):
    def __init__(self, parameters):
        self.sigma = parameters["sigma"] if "sigma" in parameters else 0.01
        self.clip = parameters["clip"] if "clip" in parameters else 0.05

    def __call__(self, ptcloud):
        ptcloud += np.clip(
            self.sigma * np.random.randn(*ptcloud.shape), -self.clip, self.clip
        ).astype(np.float32)
        return ptcloud


class RandomRotatePoints(object):
    def __init__(self, parameters):
        pass

    def __call__(self, ptcloud, rnd_value):
        trfm_mat = transforms3d.zooms.zfdir2mat(1)
        angle = 2 * math.pi * rnd_value
        trfm_mat = np.dot(transforms3d.axangles.axangle2mat([0, 1, 0], angle), trfm_mat)

        ptcloud[:, :3] = np.dot(ptcloud[:, :3], trfm_mat.T)
        return ptcloud

# class RotatePoints(object):
#     def __init__(self, parameters):
#         pass
#
#     def __call__(self, ptcloud, rnd_value):
#         trfm_mat = transforms3d.zooms.zfdir2mat(1)
#         angle = 2 * math.pi * rnd_value
#         trfm_mat = np.dot(transforms3d.axangles.axangle2mat([0, 1, 0], angle), trfm_mat)
#
#         ptcloud[:, :3] = np.dot(ptcloud[:, :3], trfm_mat.T)
#         return ptcloud

class RandomScalePoints(object):
    def __init__(self, parameters):
        self.scale = parameters["scale"]

    def __call__(self, ptcloud, rnd_value):
        trfm_mat = transforms3d.zooms.zfdir2mat(1)
        scale = np.random.uniform(1.0 / self.scale * rnd_value, self.scale * rnd_value)
        trfm_mat = np.dot(transforms3d.zooms.zfdir2mat(scale), trfm_mat)

        ptcloud[:, :3] = np.dot(ptcloud[:, :3], trfm_mat.T)
        return ptcloud


class RandomMirrorPoints(object):
    def __init__(self, parameters):
        pass

    def __call__(self, ptcloud, rnd_value):
        trfm_mat = transforms3d.zooms.zfdir2mat(1)
        trfm_mat_x = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), trfm_mat)
        trfm_mat_z = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), trfm_mat)
        if rnd_value <= 0.25:
            trfm_mat = np.dot(trfm_mat_x, trfm_mat)
            trfm_mat = np.dot(trfm_mat_z, trfm_mat)
        elif rnd_value <= 0.5:  # lgtm [py/redundant-comparison]
            trfm_mat = np.dot(trfm_mat_x, trfm_mat)
        elif rnd_value <= 0.75:
            trfm_mat = np.dot(trfm_mat_z, trfm_mat)

        ptcloud[:, :3] = np.dot(ptcloud[:, :3], trfm_mat.T)
        return ptcloud


class NormalizeObjectPose(object):
    def __init__(self, parameters):
        input_keys = parameters["input_keys"]
        self.ptcloud_key = input_keys["ptcloud"]
        self.bbox_key = input_keys["bbox"]

    def __call__(self, data):
        ptcloud = data[self.ptcloud_key]
        bbox = data[self.bbox_key]

        # Calculate center, rotation and scale
        # References:
        # - https://github.com/wentaoyuan/pcn/blob/master/test_kitti.py#L40-L52
        center = (bbox.min(0) + bbox.max(0)) / 2
        bbox -= center
        yaw = np.arctan2(bbox[3, 1] - bbox[0, 1], bbox[3, 0] - bbox[0, 0])
        rotation = np.array(
            [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
        )
        bbox = np.dot(bbox, rotation)
        scale = bbox[3, 0] - bbox[0, 0]
        bbox /= scale
        ptcloud = np.dot(ptcloud - center, rotation) / scale
        ptcloud = np.dot(ptcloud, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])

        data[self.ptcloud_key] = ptcloud
        return data

class RandomRotateAnyAxisPoints(object):
    def __init__(self, parameters, random_range="uni-180-0.2"):
        self.random_range = random_range

    def __call__(self, ptcloud, vp_rand, wp_rand, inv_rand, t_rand):
        Rs, ts = random_pose(N=1, vp_rand=vp_rand, wp_rand=wp_rand, inv_rand=inv_rand,
                             t_rand=t_rand, range=self.random_range)
        Rs, ts = Rs.astype(np.float32)[0], ts.astype(np.float32)[0]

        ptcloud_rot = np.dot(Rs, ptcloud.transpose()) + ts
        return ptcloud_rot.transpose()


def random_pose(N, vp_rand, wp_rand, inv_rand, t_rand, range="uni-180-0"):
    limit_rot, limit_t = range.split("-")[1:]

    rots = random_rot((N, ), vp_rand, wp_rand, inv_rand, float(limit_rot))
    ts = random_t((N, ), t_rand, float(limit_t))
    return rots, ts

# Credit to: Daniel Rebain
def randn_tail(shape, limits):
    comp_widths = erfc(limits / np.pi)
    widths = 1.0 - comp_widths
    inv_x = comp_widths * np.random.rand(*shape)
    inv_x = np.clip(inv_x, np.finfo(float).tiny, 2.0)
    x = erfcinv(inv_x) * np.pi
    return x

# Credit to: Daniel Rebain
def random_rot(shape, vp_rand, wp_rand, inv_rand, limit=180):
    """
    Modification of the Gaussian method to limit the angle directly.
    by Daniel Rebain
    shape: (N, )
    limit: max_angle
    rot: (N, 3, 3)
    """
    limit = limit / 180.0 * np.pi

    vp = vp_rand
    d2 = np.sum(vp**2, axis=-1)
    c2theta = np.cos(0.5 * limit)**2
    wp_limit = np.sqrt(c2theta * d2 / (1.0 - c2theta))

    comp_widths = erfc(wp_limit / np.pi)
    # widths = 1.0 - comp_widths
    inv_x = comp_widths * inv_rand #np.random.rand(*shape)
    inv_x = np.clip(inv_x, np.finfo(float).tiny, 2.0)
    wp = erfcinv(inv_x) * np.pi

    wp *= 2.0 * wp_rand - 1.0
    q = np.concatenate([vp, wp[:, None]], axis=-1)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    rot = Rotation.from_quat(q).as_matrix()
    return rot

def random_t(shape, random_t, limit=0):
    t = random_t * limit
    return t