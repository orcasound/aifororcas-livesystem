"""
Data augmentation. 
"""
import torch
import numpy as np


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Sourced from: https://github.com/uoguelph-mlrg/Cutout
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (numpy array): image of size (C, H, W).
        Returns:
            numpy array: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.shape[0]
        w = img.shape[1]

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0. 

        # mask = torch.from_numpy(mask)
        # mask = mask.expand_as(img)
        img = img * mask

        return img

class SpecAug(object):
    """
    Randomly mask out time and frequency bands similar to https://arxiv.org/pdf/1904.08779.pdf
    Args:
        m_T (int): number of time masks
        T (int): max width of time mask
        m_F (int): number of frequency masks
        F (int): max width of frequency mask
    """
    def __init__(self,m_T,T,m_F,F):
        self.m_F, self.m_T = m_F, m_T 
        self.F, self.T = F, T
    
    def __call__(self,spec):
        """
        Returns:
            numpy array: spectrogram of size (T,F) with masks applied
        """
        mask = np.ones_like(spec)
        for i in range(self.m_F):
            fmin, fmax = self._get_bin_idxs(spec.shape[1],self.F)
            mask[:,fmin:fmax] = 0.
        for i in range(self.m_T):
            tmin, tmax = self._get_bin_idxs(spec.shape[0],self.T)
            mask[tmin:tmax,:] = 0.
        
        return spec*mask
    
    def _get_bin_idxs(self,max_idx,width):
        # range [width//2, max_idx-width//2 )
        semiwidth = np.random.randint(width)//2
        center = np.random.randint(semiwidth,max_idx-semiwidth)
        return center-semiwidth, center+semiwidth