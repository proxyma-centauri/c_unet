import os
import sys
import time
import logging

import numpy as np

import torch
import torchvision.transforms.functional as torchtransforms


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
        self.logger.debug("Computing Cayley table for V group")
        cayley = np.asarray([[0,1,2,3],
                             [1,0,3,2],
                             [2,3,0,1],
                             [3,2,1,0]])
        return cayley        


    def get_Grotations(self, x):
        """Rotate the tensor x with all 4 Klein Vierergruppe rotations

        Args:
            x: [n_channels, h,w,d]
        Returns:
            list of 4 rotations of x [[n_channels,h,w,d],....]
        """
        xsh = x.shape
        angles = [0.,np.pi]
        rx = []
        for i in range(2):
            # 2x 180. rotations about the z axis
            perm = [0,2,1,3]
            y = x.permute(perm)

            y = torchtransforms.rotate(y, angles[i]) # dims

            y = y.permute(perm)

            # 2x 180. rotations about another axis
            for j in range(2):
                perm = [0,3,2,1]
                z = y.permute(perm)
                z = torchtransforms.rotate(z, angles[j])
                z = z.permute(perm)
                rx.append(z)
        return rx


    def G_permutation(self, W):
        """Permute the outputs of the group convolution
        
        Args:
            W: [n_channels_in, h, w, d, group_dim, n_channels_out, group_dim]
        Returns:
            list of 4 permutations of W [[n_channels_in, h, w, d, group_dim, n_channels_out, group_dim],....]
        """
        Wsh = W.shape
        cayley = self.cayleytable
        U = []
        for i in range(4):
            perm_mat = self.get_permutation_matrix(cayley, i).to(W.device)
            w = W[:,:,:,:,:,:,i]
            w = w.permute([0,1,2,3,5,4])
            w = w.reshape([-1, 4])
            w = torch.matmul(w, perm_mat)
            w = w.view(list(Wsh[:4])+[-1,4]) 
            U.append(w.permute([0,1,2,3,5,4]))
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
