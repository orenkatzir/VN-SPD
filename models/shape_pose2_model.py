import torch
import itertools
from .base_model import BaseModel
from . import networks
import util.pc_utils as pc_utils
from pytorch3d import loss
from .shape_pose_model import ShapePoseModel

EPS = 1e-10

class ShapePose2Model(ShapePoseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        #Training arguments:

        parser.set_defaults(weight_sample_T=1.0)
        parser.set_defaults(weight_sample_R=1.0)
        parser.set_defaults(weight_fps_T=1.0)
        parser.set_defaults(weight_fps_R=1.0)
        parser.set_defaults(weight_can_T=1.0)
        parser.set_defaults(weight_can_R=1.0)
        parser.set_defaults(weight_part_T=1.0)
        parser.set_defaults(weight_part_R=1.0)
        parser.set_defaults(weight_noise_T=1.0)
        parser.set_defaults(weight_noise_R=1.0)
        parser.set_defaults(weight_ortho=3.0)
        parser.set_defaults(global_bias=0)

        parser.set_defaults(net_d='patch')

        return parser

    def __init__(self, opt):
        ShapePoseModel.__init__(self, opt)