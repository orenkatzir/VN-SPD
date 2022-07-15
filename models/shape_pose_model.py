import torch
import itertools
from .base_model import BaseModel
from . import networks
import util.pc_utils as pc_utils
from pytorch3d import loss

EPS = 1e-10

class ShapePoseModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        #Training arguments:
        return parser


    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_placeholder = 0
        self.loss_names = ['recon_1', 'recon_2', 'ortho', 'can_rot']

        visual_names = ['pc_show', 'pc_at_inv_show', 'recon_pc_inv_show', 'recon_pc_inv_color_show',
                        'recon_pc_show', 'pc_rotated_show', 'pc_rotated_at_inv_show', 'recon_rotated_inv_show']

        if self.opt.add_noise:
            visual_names += ['noised_pc_show', 'noised_pc_at_inv_show']
            self.loss_names += ["noised_rot", "noised_T"]

        if self.opt.remove_knn > 0:
            self.loss_names += ['part_rot', 'part_T']
            visual_names += ['pc_part_show']

        if self.opt.resample:
            self.loss_names += ['sample']
            visual_names += ['pc_sample_show']

        if self.opt.fps:
            visual_names += ['pc_fps_show', 'pc_fps_at_inv_show']
            self.loss_names += ["fps_rot", "fps_T"]

        self.visual_names += [visual_names]
        self.model_names = ['Encoder', 'Decoder', 'Rotation']

        self.netEncoder = networks.define_encoder(opt)
        self.netDecoder = networks.define_decoder(opt)
        self.netRotation = networks.define_rot_module(opt.net_rot, opt.nlatent, opt.which_strict_rot, opt.gpu_ids)

        # TODO: move this to outside
        # Color for visulization
        self.color = torch.tensor([0., 0., 139.], device=self.device).unsqueeze(0).unsqueeze(2)
        self.decoder_color = self.get_decoder_color()

        if self.isTrain:
            self._initialize_losses()
            if self.opt.alternate:
                self.train_recon = True
            self.criterionPCRecon = loss.chamfer_distance
            self.criterionMSE = torch.nn.MSELoss()
            self.criterionROT = networks.RotationLoss(device=self.device, which_metric=opt.which_rot_metric)
            self.criterionOrtho = networks.OrthogonalLoss(device=self.device, which_metric=opt.which_ortho_metric)

            self.optimizer = torch.optim.Adam(self.get_optim_params(), lr=opt.lrate)
            self.optimizers = [self.optimizer]

    def get_optim_params(self):
        params = itertools.chain(self.netEncoder.parameters(), self.netDecoder.parameters(),
                                 self.netRotation.parameters())

        return params

    def set_input(self, input, use_rand_trans=True):
        # input: 0 is PC, 1 is cls
        #input:
        #   0 - pc BXCXN
        #   1 - cls (not used in this experiment)
        self.pc = input[0].to(self.device)

        if use_rand_trans:
            self.pc, trot = pc_utils.rotate(self.pc, self.opt.rot, self.device, t=self.opt.se3_T, return_trot=True)
        else:
            trot = None

        if self.opt.resample and self.isTrain:
            self.pc_sample = pc_utils.sample(self.pc.clone().detach(), self.opt.npoint, self.device)

        if not self.opt.no_input_resample:
            self.pc = pc_utils.sample(self.pc.clone().detach(), self.opt.npoint, self.device)

        if self.isTrain:
            if self.opt.add_noise:
                self.noised_pc = self.pc.clone() + self.opt.noise_amp * torch.randn(self.pc.size(), device=self.device)

            if self.opt.remove_knn > 0:
                id_to_reomve = torch.randint(0, self.pc.size(2), (self.pc.size(0), 1), device=self.device).contiguous()
                self.pc_part = pc_utils.remove_knn(self.pc.clone().detach().contiguous(), id_to_reomve, k=self.opt.remove_knn, device=self.device)

            if self.opt.fps:
                num_fps_points = torch.randint(self.opt.min_num_fps_points, self.opt.max_num_fps_points, (1,)).item()
                pc = self.pc.clone().detach()
                _, new_pc = pc_utils.farthest_point_sample_xyz(pc.transpose(1,2), num_fps_points)
                self.pc_fps = new_pc.transpose(1,2).detach()

        return self.pc, trot

    #TODO: move to geom_utils
    def to_rotation_mat(self):
        pc_utils.to_rotation_mat(self.rot_mat, self.opt.which_rot)

    def cal_rot_dist(self, R1, R2, detach_R2=False):
        if detach_R2:
            dist = self.criterionROT(R1, R2.detach())
        else:
            dist = self.criterionROT(R1, R2)
        return dist

    def cal_trans_dist(self, T1, T2, detach_T2=False):
        if detach_T2:
            dist = self.criterionMSE(T1, T2.detach())
        else:
            dist = self.criterionMSE(T1, T2)
        return dist

    def test(self, pc=None):
        if pc is None:
            pc = self.pc
        inv_z, eq_z, t_vec = self.netEncoder(pc)

        rot_mat = self.netRotation(eq_z)

        pc_at_inv = torch.matmul(pc.permute(0, 2, 1) - t_vec, rot_mat.transpose(1, 2)).permute(0, 2, 1)

        return pc_at_inv, (rot_mat.detach(), t_vec.detach())

    def get_latent(self):
        inv_z, self.eq_z, t_vec = self.netEncoder(self.pc)
        rot_mat = self.netRotation(self.eq_z)

        return inv_z, rot_mat.detach(), t_vec.detach()

    def decode(self, inv_z):
        decoded_pc_inv, patches = self.netDecoder(inv_z)
        recon_pc_inv = decoded_pc_inv.permute(0, 2, 1)
        return recon_pc_inv

    def forward(self):
        inv_z, self.eq_z, self.t_vec = self.netEncoder(self.pc)
        decoded_pc_inv, self.patches = self.netDecoder(inv_z)
        self.rot_mat = self.netRotation(self.eq_z)

        self.recon_pc = (torch.matmul(decoded_pc_inv, self.rot_mat) + self.t_vec).permute(0,2,1)
        self.pc_at_inv = torch.matmul(self.pc.permute(0, 2, 1) -  self.t_vec, self.rot_mat.transpose(1,2)).permute(0,2,1)
        self.recon_pc_inv = decoded_pc_inv.permute(0, 2, 1)

        ## Apply augmentations:
        if self.opt.add_noise:
            aug_inv_z, aug_eq_z, self.t_noised_vec = self.netEncoder(self.noised_pc)
            self.noised_rot_mat = self.netRotation(aug_eq_z)

            # For visualization
            self.noised_pc_at_inv = torch.matmul(self.noised_pc.permute(0, 2, 1) - self.t_noised_vec,
                                                 self.noised_rot_mat.transpose(1,2)).permute(0,2,1)

        if self.opt.fps:
            fps_inv_z, fps_eq_z, self.t_fps_vec = self.netEncoder(self.pc_fps)
            self.fps_rot_mat = self.netRotation(fps_eq_z)

            #For visualization
            self.pc_fps_at_inv = torch.matmul(self.pc_fps.permute(0, 2, 1) - self.t_fps_vec,
                                              self.fps_rot_mat.transpose(1, 2)).permute(0, 2, 1)

        if self.opt.remove_knn > 0:
            part_inv_z, part_eq_z, self.t_part_vec = self.netEncoder(self.pc_part)
            self.part_rot_mat = self.netRotation(part_eq_z)

        if self.opt.resample:
            sample_inv_z, sample_eq_z, self.t_sample_vec = self.netEncoder(self.pc_sample)
            self.sample_rot_mat = self.netRotation(sample_eq_z)

        #  Visualization - validating equivariance by definition
        with torch.no_grad():
            self.pc_rotated, trot_ = pc_utils.rotate(self.pc, self.opt.rot, self.device, t=self.opt.se3_T, return_trot=True)
            rotated_inv_z, rotated_eq_z, t_vec = self.netEncoder(self.pc_rotated)
            rot_mat_ = self.netRotation(rotated_eq_z)

            self.decoded_rotated_pc_inv, patches = self.netDecoder(rotated_inv_z)
            self.pc_rotated_at_inv = torch.matmul(self.pc_rotated.permute(0, 2, 1) - t_vec, rot_mat_.transpose(1, 2)).permute(0, 2, 1)

    def cal_recon_loss(self):
        self.loss_recon_1 = self.criterionPCRecon(self.recon_pc.permute(0, 2, 1), self.pc.permute(0, 2, 1))[0] * self.opt.weight_recon1

        if self.opt.detached_recon_2:
            self.loss_recon_2 = \
            self.criterionPCRecon(self.pc_at_inv.permute(0, 2, 1).detach(), self.recon_pc_inv.permute(0, 2, 1))[
                0] * self.opt.weight_recon2
        else:
            self.loss_recon_2 = \
            self.criterionPCRecon(self.pc_at_inv.permute(0, 2, 1), self.recon_pc_inv.permute(0, 2, 1))[
                0] * self.opt.weight_recon2

        losses = [self.loss_recon_1, self.loss_recon_2]
        return losses

    def cal_ortho_loss(self):
        self.loss_ortho = self.criterionOrtho(self.rot_mat) * self.opt.weight_ortho
        return [self.loss_ortho]

    def cal_aug_loss(self):
        losses = []
        if self.opt.add_noise:
            self.loss_noised_rot = self.cal_rot_dist(self.noised_rot_mat, self.rot_mat, self.opt.detach_aug_loss) * self.opt.weight_noise_R
            self.loss_noised_T = self.cal_trans_dist(self.t_noised_vec, self.t_vec, self.opt.detach_aug_loss) * self.opt.weight_noise_T
            losses += [self.loss_noised_rot, self.loss_noised_T]
            if self.opt.add_ortho_aug:
                self.loss_ortho += self.criterionOrtho(self.noised_rot_mat)

        if self.opt.fps:
            self.loss_fps_rot = self.cal_rot_dist(self.fps_rot_mat, self.rot_mat, self.opt.detach_aug_loss) * self.opt.weight_fps_R
            self.loss_fps_T = self.cal_trans_dist(self.t_fps_vec, self.t_vec, self.opt.detach_aug_loss) * self.opt.weight_fps_T
            losses += [self.loss_fps_rot, self.loss_fps_T]
            if self.opt.add_ortho_aug:
                self.loss_ortho += self.criterionOrtho(self.fps_rot_mat)

        if self.opt.remove_knn > 0:
            self.loss_part_rot = self.cal_rot_dist(self.part_rot_mat, self.rot_mat, self.opt.detach_aug_loss) * self.opt.weight_part_R
            self.loss_part_T = self.cal_trans_dist(self.t_part_vec, self.t_vec, self.opt.detach_aug_loss) * self.opt.weight_part_T
            losses += [self.loss_part_rot, self.loss_part_T]
            if self.opt.add_ortho_aug:
                self.loss_ortho += self.criterionOrtho(self.part_rot_mat)

        if self.opt.resample:
            self.loss_sample_rot = self.cal_rot_dist(self.sample_rot_mat, self.rot_mat, self.opt.detach_aug_loss) * self.opt.weight_sample_R
            self.loss_sample_T = self.cal_trans_dist(self.t_sample_vec, self.t_vec, self.opt.detach_aug_loss) * self.opt.weight_sample_T
            losses += [self.loss_sample_rot, self.loss_sample_T]
            if self.opt.add_ortho_aug:
                self.loss_ortho += self.criterionOrtho(self.sample_rot_mat)

        if self.opt.apply_can_rot_loss:
            rotated_recon, trot = pc_utils.rotate(self.recon_pc_inv, "so3", self.device, return_trot=True)
            expected_rot = trot.get_matrix()[:, :3, :3].detach()
            _, rotated_recon_eq_z, t_rot_can_vec = self.netEncoder(rotated_recon.detach())
            rotated_recon_rot_mat = self.netRotation(rotated_recon_eq_z).squeeze(-1)

            self.loss_can_rot = self.criterionMSE(t_rot_can_vec, torch.zeros_like(t_rot_can_vec, device=self.device)) * self.opt.weight_can_T
            self.loss_can_rot += self.cal_rot_dist(rotated_recon_rot_mat, expected_rot) * self.opt.weight_can_R
            losses += [self.loss_can_rot]

        return losses

    def backward(self):
        if self.opt.alternate:
            # Alternate between reconstruction loss and rotation losses
            if self.train_recon:
                losses = self.cal_recon_loss()
                self.train_recon = False
            else:
                losses = self.cal_aug_loss() + self.cal_ortho_loss()
                self.train_recon = True
        else:
            losses = self.cal_recon_loss() + self.cal_aug_loss() + self.cal_ortho_loss()

        loss = sum(losses)
        loss.backward()
        return loss.item()

    def optimize_parameters(self, epoch_id, *args):
        #Taken from AtlasNet:
        if epoch_id == self.opt.firstdecay or epoch_id == self.opt.seconddecay:
            params = self.get_optim_params()

            if epoch_id == self.opt.firstdecay:
                self.optimizer = torch.optim.Adam(params, lr=self.opt.lrate / 10)
            else:
                self.optimizer = torch.optim.Adam(params, lr=self.opt.lrate / 100)

        self.optimizer.zero_grad()

        # forward
        self.forward()
        # G
        self.optimizer.zero_grad()
        loss = self.backward()
        self.optimizer.step()


    def add_color(self, pc, color=None):
        if color == None:
            color = self.color.repeat(pc.size(0), 1, pc.size(2))
        elif color.size(0) != pc.size(0):
            color = color.repeat(pc.size(0), 1, 1)
        return torch.cat((pc, color), dim=1)

    def get_decoder_color(self):
        base_colors = torch.tensor([[0., 0. , 139.], [139., 0 , 0.], [0., 139. , 0.],  [139., 0 , 139.],
                            [153., 153 , 253.], [253., 153. , 0.], [139., 139., 0.], [0., 153. , 253.],
                            [ 153., 0., 253.], [ 0., 0., 0.]],
                            device=self.device)
        colors = []
        for i in range(self.opt.npatch):
            colors.append(base_colors[i].unsqueeze(0).unsqueeze(2).repeat(1, 1, self.opt.npoint // self.opt.npatch))

        return torch.cat(colors, dim=2)


    def compute_visuals(self):
        self.netEncoder.eval()
        self.netRotation.eval()
        self.netDecoder.eval()
        with torch.no_grad():
            self.pc_show = self.add_color(self.pc)
            self.pc_at_inv_show = self.add_color(self.pc_at_inv)

            if self.opt.add_noise:
                self.noised_pc_show = self.add_color(self.noised_pc)
                self.noised_pc_at_inv_show = self.add_color(self.noised_pc_at_inv)

            if self.opt.fps:
                self.pc_fps_show = self.add_color(self.pc_fps)
                self.pc_fps_at_inv_show = self.add_color(self.pc_fps_at_inv)

            if self.opt.remove_knn > 0:
                self.pc_part_show = self.add_color(self.pc_part)

            if self.opt.resample:
                self.pc_sample_show = self.add_color(self.pc_sample)

            self.pc_rotated_show = self.add_color(self.pc_rotated)
            self.pc_rotated_at_inv_show = self.add_color(self.pc_rotated_at_inv)
            self.recon_rotated_inv_show = self.add_color(self.decoded_rotated_pc_inv.permute(0, 2, 1))

            self.recon_pc_inv_color_show = self.add_color(self.recon_pc_inv, self.decoder_color)
            self.recon_pc_show = self.add_color(self.recon_pc, self.decoder_color)
            self.recon_pc_inv_show = self.add_color(self.recon_pc_inv)

        self.netEncoder.train()
        self.netRotation.train()
        self.netDecoder.train()
