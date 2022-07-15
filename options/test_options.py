from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=2000, help='how many test point cloud to run')

        parser.add_argument('--num_stability_exp', type=int, default=10, help='Number of orientation per instance')
        parser.add_argument('--transform_path', type=str, default='', help='Pre-saved transformation, if not given, apply random')

        # rewrite devalue values
        parser.set_defaults(model='test')
        parser.set_defaults(no_input_resample=1)
        parser.set_defaults(which_strict_mode='svd')


        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
