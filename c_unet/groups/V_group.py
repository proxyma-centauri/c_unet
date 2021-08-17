import logging

import numpy as np
import torch.nn.functional as F

import torch
from einops import rearrange


class V_group(object):
    def __init__(self):
        self.group_dim = 4
        self.logger = logging.getLogger(__name__)
        self.cayleytable = self.get_cayleytable()

    def get_cayleytable(self):
        """Returns the Cayley table of V group

        Returns:
            4 by 4 numpy array
        """
        cayley = np.asarray([[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1],
                             [3, 2, 1, 0]])
        return cayley

    def get_rot_mat(self, theta, device):
        theta = torch.tensor(theta)
        rot_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                                [torch.sin(theta),
                                 torch.cos(theta), 0]]).to(device)
        return rot_mat

    def rot_img(self, x, theta):
        rot_mat = self.get_rot_mat(theta,
                                   x.device)[None,
                                             ...].repeat(x.shape[0], 1, 1)
        grid = F.affine_grid(rot_mat, x.size(), align_corners=False)
        x_rot = F.grid_sample(x, grid, align_corners=False)
        return x_rot

    def get_Grotations(self, W):
        """Rotate the tensor x with all 4 Klein Vierergruppe rotations

        Args:
            W: [n_channels, h,w,d]
        Returns:
            list of 4 rotations of x [[n_channels,h,w,d],....]
        """
        angles = [0., np.pi]

        Wrots = []

        for z_angle in angles:
            # 2x 180. rotations about the z axis
            perm = [0, 2, 1, 3]

            Wz_1 = W.permute(perm)
            Wz_2 = self.rot_img(Wz_1, z_angle)
            W_rot_around_z = Wz_2.permute(perm)

            # 2x 180. rotations about another axis
            for y_angle in angles:
                perm = [0, 3, 2, 1]

                Wy_1 = W_rot_around_z.permute(perm)
                Wy_2 = self.rot_img(Wy_1, y_angle)
                W_rot_around_both_axis = Wy_2.permute(perm)

                Wrots.append(W_rot_around_both_axis)

        return Wrots

    def G_permutation(self, W):
        """Permute the outputs of the group convolution
        
        Args:
            W: [group_dim, n_channels_out, group_dim, n_channels_in, h, w, d]
        Returns:
            list of 4 permutations of W [[group_dim, n_channels_out, group_dim, n_channels_in, h, w, d],....]
        """
        Wsh = W.shape
        cayley = self.cayleytable
        Wsh = W.shape
        U = []
        for i in range(4):
            perm_mat = self.get_permutation_matrix(cayley, i).to(W.device)
            w = W[i, :, :, :, :, :, :]
            w = w.permute([0, 5, 2, 3, 4, 1])
            w = w.reshape([-1, 4])
            w = torch.matmul(w, perm_mat)
            w = w.view([-1] + list(Wsh[2:]))
            w = w.permute([0, 5, 2, 3, 4, 1])
            U.append(w)
        return U

    def get_permutation_matrix(self, perm, dim):
        """Creates and return the permutation matrix
        
        Args:
            perm: numpy matrix (Cayley matrix of the group)
        Returns:
            float Tensor
        """
        # TODO : make cleaner
        ndim = perm.shape[0]
        mat = np.zeros((ndim, ndim))
        for j in range(ndim):
            mat[j, perm[j, dim]] = 1
        return torch.from_numpy(mat).float()
