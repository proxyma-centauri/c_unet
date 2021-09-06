import logging

import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as torchtransforms


class S4_group(object):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.group_dim = 24
        self.cayleytable = self.get_cayleytable()

    def get_rot_mat(self, theta, device):
        """
        Construct the rotation matrix associated with the given angle theta

        Args:
            theta: rotation angle (rad)
            device: device the tensor should be created on
        Returns:
            4 by 4 torch tensor
        """
        theta = torch.tensor(theta)
        rot_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                                [torch.sin(theta),
                                 torch.cos(theta), 0]]).to(device)
        return rot_mat

    def rot_img(self, x, theta):
        """
        Rotates the image by an angle of theta radians.
        This is autograd compatible.

        Args:
            x : 3D image to rotate
            theta: rotation angle (rad)
        Returns:
            Rotated image
        """
        rot_mat = self.get_rot_mat(theta,
                                   x.device)[None,
                                             ...].repeat(x.shape[0], 1, 1)
        grid = F.affine_grid(rot_mat, x.size())
        x_rot = F.grid_sample(x, grid)
        return x_rot

    def get_Grotations(self, x):
        """Rotate the tensor x with all 24 S4 rotations

        Args:
            x: [n_channels, h,w,d]
        Returns:
            list of 24 rotations of x [[n_channels,h,w,d],....]
        """
        angles = [0., np.pi / 2., np.pi, 3. * np.pi / 2.]
        rx = []
        for i in range(4):
            # Z4 rotations about the z axis
            perm = [0, 2, 1, 3]
            y = x.permute(perm)
            y = self.rot_img(y, angles[i])
            y = y.permute(perm)
            # Rotations in the quotient space (sphere S^2)
            # i) Z4 rotations about y axis
            for j in range(4):
                perm = [0, 3, 2, 1]
                z = y.permute(perm)
                z = self.rot_img(z, angles[-j])
                z = z.permute(perm)

                rx.append(z)
            # ii) 2 rotations to the poles about the x axis
            perm = [0, 1, 3, 2]
            z = y.permute(perm)
            z = self.rot_img(z, angles[3])
            z = z.permute(perm)
            rx.append(z)

            z = y.permute(perm)
            z = self.rot_img(z, angles[1])
            z = z.permute(perm)
            rx.append(z)

        return rx

    def G_permutation(self, W):
        """Permute the outputs of the group convolution
        
        Args:
            W: [n_channels_in, h, w, d, group_dim, n_channels_out, group_dim]
        Returns:
            list of 24 permutations of W [[n_channels_in, h, w, d, group_dim, n_channels_out, group_dim],....]
        """
        Wsh = W.shape
        cayley = self.cayleytable
        Wsh = W.shape
        U = []
        for i in range(24):
            perm_mat = self.get_permutation_matrix(cayley, i).to(W.device)
            w = W[i, :, :, :, :, :, :]
            w = w.permute([1, 0, 2, 3, 4, 5])
            w = w.reshape([-1, 24])
            w = torch.matmul(w, perm_mat)
            w = w.view([-1] + list(Wsh[2:]))
            w = w.permute([1, 0, 2, 3, 4, 5])
            U.append(w)
        return U

    def get_permutation_matrix(self, perm, dim):
        """Creates and return the permutation matrix
        
        Args:
            perm: numpy matrix (Cayley matrix of the group)
            dim: the number of the group element whose 
                permutation matrix we want
        Returns:
            float Tensor
        """
        ndim = perm.shape[0]
        mat = np.zeros((ndim, ndim))
        for j in range(ndim):
            mat[j, perm[j, dim]] = 1
        return torch.from_numpy(mat).float()

    def get_cayleytable(self):
        """Returns the Cayley table of V group

        Returns:
            4 by 4 numpy array
        """
        Z = self.get_s4mat()
        cayley = []
        for y in Z:
            for z in Z:
                r = z @ y
                for i, el in enumerate(Z):
                    if np.sum(np.square(el - r)) < 1e-6:
                        cayley.append(i)
        cayley = np.stack(cayley)
        return np.reshape(cayley, [24, 24])

    def get_s4mat(self):
        Z = []
        for i in range(4):
            # Z_4 rotation about Y
            # S^2 rotation
            for j in range(4):
                z = self.get_3Drotmat(i, j, 0)
                Z.append(z)
            # Residual pole rotations
            Z.append(self.get_3Drotmat(i, 0, 1))
            Z.append(self.get_3Drotmat(i, 0, 3))
        return Z

    def get_3Drotmat(self, x, y, z):
        c = [1., 0., -1., 0.]
        s = [0., 1., 0., -1]

        Rx = np.asarray([[c[x], -s[x], 0.], [s[x], c[x], 0.], [0., 0., 1.]])
        Ry = np.asarray([[c[y], 0., s[y]], [0., 1., 0.], [-s[y], 0., c[y]]])
        Rz = np.asarray([[1., 0., 0.], [0., c[z], -s[z]], [0., s[z], c[z]]])
        return Rz @ Ry @ Rx
