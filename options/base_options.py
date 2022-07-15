import argparse
import os
from util import util
import torch
import models
import data


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--model', type=str, default='cycle_gan', help='chooses which model to use. [atlas_vnt]')

        # model parameters
        parser.add_argument('--rot', type=str, default='se3', help='Apply transforamtion to input point cloud')
        parser.add_argument('--se3_T', type=float, default=0.1, help='range of translation')

        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')

        #General settings:
        parser.add_argument('--no_input_resample', type=int, default=0, help='Avoid random resampling the input')

        #Encoder params
        parser.add_argument('--nlatent', type=int, default=1020, help='')
        parser.add_argument('--pooling', type=str, default='mean', help='Pooling method [default: mean]',
                            choices=['mean', 'max'])
        parser.add_argument('--n_knn', default=40, type=int,
                            help='Number of nearest neighbors to use [default: 40]')
        parser.add_argument('--which_norm_VNT', type=str, default='norm', help='Normalization method of VNTLinear layer [default: norm]')
        parser.add_argument('--global_bias', type=int, default=1,
                            help='Use global/local bias at feature extraction')

        # Network selection
        parser.add_argument('--net_e', type=str, default='simple', help='which encoder to use')
        parser.add_argument('--base_ch', default=64, type=int, help='base encoder channels')

        parser.add_argument('--net_rot', type=str, default='simple', help='which rotation module to use')
        parser.add_argument('--which_strict_rot', type=str, default='None', choices=['svd', 'gram_schmidt', 'None'],
                            help='Define rotation tansform, [default: None]')

        #Decoder params
        parser.add_argument('--net_d', type=str, default='point', help='which decoder to use')
        parser.add_argument('--patchDim', type=int, default=2, help='Dimension of patch, relevant for atlasNet')
        parser.add_argument('--patchDeformDim', type=int, default=3, help='Output dimension of atlas net decoder, relevant for atlasNet')
        parser.add_argument('--npatch', type=int, default=10, help='number of patches, relevant for atlasNet')

        # dataset parameters
        parser.add_argument('--npoints', default=2500, type=int, help='number of points in the loaded point cloud')
        parser.add_argument('--npoint', default=1020, type=int, help='number of points after sampling the point cloud')

        parser.add_argument('--dataroot', required=True, help='path to dataset (passed to the dataset class)')
        parser.add_argument('--class_choice', type=str, default='all', help='select specific class: e.g. chair  chair,airplane')
        parser.add_argument('--dataset_mode', type=str, default='shapenet', help='chooses the dataset class')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes point clouds in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=0, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=10, help='input batch size')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--no_data_augmentation', action='store_true', help='no rotation')

        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')


        parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')

        #Shared for vis at test time
        # Noise augmentation
        parser.add_argument('--add_noise', action='store_true', help='Apply noise augmentation')
        parser.add_argument('--noise_amp', type=float, default=0.025, help='noise scale')
        parser.add_argument('--weight_noise_T', type=float, default=0.1, help='scale noise for translation branch')
        parser.add_argument('--weight_noise_R', type=float, default=0.1, help='scale noise for rotation branch')

        # FPS augmentation:
        parser.add_argument('--fps', action='store_true', help='Apply fps augmentation')
        parser.add_argument('--min_num_fps_points', type=int, default=300, help='min range for fps')
        parser.add_argument('--max_num_fps_points', type=int, default=500, help='max range for fps')
        parser.add_argument('--weight_fps_T', type=float, default=0.1, help='FPS weight for translation branch')
        parser.add_argument('--weight_fps_R', type=float, default=0.1, help='FPS weight for rotation branch')

        # KNN removal augmentation:
        parser.add_argument('--remove_knn', default=100, type=int,
                            help='number of points to remove, default 0 - no part removal')
        parser.add_argument('--weight_part_T', type=float, default=0.1,
                            help='KNN removal weight for translation branch')
        parser.add_argument('--weight_part_R', type=float, default=0.1, help='KNN removal weight for rotation branch')

        # Resample augmentation:
        parser.add_argument('--resample', action='store_true', help='Apply resample augmentation')
        parser.add_argument('--weight_sample_T', type=float, default=0.1, help='Reample weight for translation branch')
        parser.add_argument('--weight_sample_R', type=float, default=0.1, help='Reample weight for rotation branch')

        # Can rot loss:
        parser.add_argument('--apply_can_rot_loss', type=int, default=1, help='Use can shape as augmentation')
        parser.add_argument('--weight_can_T', type=float, default=0.1,
                            help='Canonical aug of weight for translation branch')
        parser.add_argument('--weight_can_R', type=float, default=0.1, help='Canonical aug weight for rotation branch')


        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        opt.class_choice = opt.class_choice.replace(' ', '').lower()
        if opt.class_choice == '':
            opt.class_choice = []
        else:
            opt.class_choice = opt.class_choice.split(',')


        self.opt = opt
        return self.opt
