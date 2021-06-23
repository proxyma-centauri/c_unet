import logging

import numpy as np

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
        cayley = np.asarray([[0,1,2,3],
                             [1,0,3,2],
                             [2,3,0,1],
                             [3,2,1,0]])
        return cayley        


    def get_rot_mat(self,theta, axis):
        theta = torch.tensor(theta)
    
        if axis == 0:
            mat = torch.tensor([[1, 0, 0], [0,
                                            torch.cos(theta), -torch.sin(theta)],
                                [0, torch.sin(theta),
                                torch.cos(theta)]])
        elif axis == 1:
            mat = torch.tensor([[torch.cos(theta), 0,
                                torch.sin(theta)], [0, 1, 0],
                                [-torch.sin(theta), 0,
                                torch.cos(theta)]])
        elif axis == 2:
            mat = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                                [torch.sin(theta),
                                torch.cos(theta), 0], [0, 0, 1]])
        else:
            raise ValueError("This axis does not exist")
    
        return rearrange(mat, "x y -> 1 1 x y")


    def get_Grotations(self, x):
        """Rotate the tensor x with all 4 Klein Vierergruppe rotations

        Args:
            x: [n_channels, h,w,d]
        Returns:
            list of 4 rotations of x [[n_channels,h,w,d],....]
        """
        angles = [0., np.pi]
        z_mats = [self.get_rot_mat(angle, 2) for angle in angles]
        y_mats = [self.get_rot_mat(angle, 1) for angle in angles]
        Wrots = []

        for z_mat in z_mats:
            # 2x 180. rotations about the z axis
            W_rot_around_z = torch.matmul(z_mat, x)

            # 2x 180. rotations about another axis
            for y_mat in y_mats:
                W_rot_around_both_axis = torch.matmul(y_mat, W_rot_around_z)
                Wrots.append(W_rot_around_both_axis)
        return Wrots


    def G_permutation(self, W):
        """Permute the outputs of the group convolution
        
        Args:
            W: [group_dim, n_channels_in, group_dim, n_channels_out, h, w, d]
        Returns:
            list of 4 permutations of W [[group_dim, n_channels_in, group_dim, n_channels_out, h, w, d],....]
        """
        Wsh = W.shape
        cayley = self.cayleytable
        Wsh = W.shape
        U = []

        for i in range(4):
            perm_mat = self.get_permutation_matrix(cayley, i).to(W.device)
            w = W[i,:,:,:,:,:]
            w = w.reshape([-1, 4])
            w = torch.matmul(w, perm_mat)
            w = w.view([4, -1]+list(Wsh[2:])) 
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
            mat[j,perm[j,dim]] = 1
        return torch.from_numpy(mat).float()
