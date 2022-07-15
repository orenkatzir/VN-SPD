from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=0, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=50, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--n_epochs', type=int, default=400, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')

        #TODO: unify the lr of this with atals
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

        parser.add_argument('--lrate', type=float, default=0.001, help='')
        parser.add_argument('--firstdecay', type=int, default=250, help='')
        parser.add_argument('--seconddecay', type=int, default=300, help='')
        parser.add_argument('--alternate', action='store_true',
                            help='Alternate between reconstruction and other losses. default: False')

        parser.add_argument('--which_rot_metric', type=str, default='cosine',
                            help='loss selection for augmented rotation loss')
        parser.add_argument('--which_ortho_metric', type=str, default='MSE', help='loss selection for orthogonal loss')
        parser.add_argument('--weight_ortho', type=float, default=0.5, help='weight of orthogonal loss')
        parser.add_argument('--weight_recon1', type=float, default=1., help='weight of reconstruction loss')
        parser.add_argument('--weight_recon2', type=float, default=0.1, help='weight of reconstruction loss2 (see code)')
        parser.add_argument('--detached_recon_2', action='store_true',
                            help='Detach original rotated point cloud for loss')
        parser.add_argument('--detach_aug_loss', action='store_true',
                            help='Detach non-augmented rotation matrix for aug losses')
        parser.add_argument('--add_ortho_aug', action='store_true', help='Apply orthogonal loss on augmented rotation')


        self.isTrain = True
        return parser
