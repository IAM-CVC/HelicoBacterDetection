# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 16:45:18 2024

@author: debora

The index can be described as:

.. math::

  \text{SSIM}(x, y) = \frac{(2\mu_x\mu_y+c_1)(2\sigma_{xy}+c_2)}
  {(\mu_x^2+\mu_y^2+c_1)(\sigma_x^2+\sigma_y^2+c_2)}

where:
  - :math:`c_1=(k_1 L)^2` and :math:`c_2=(k_2 L)^2` are two variables to
    stabilize the division with weak denominator.
  - :math:`L` is the dynamic range of the pixel-values (typically this is
    :math:`2^{\#\text{bits per pixel}}-1`).

the loss, or the Structural dissimilarity (DSSIM) can be finally described
as:

.. math::

  \text{loss}(x, y) = \frac{1 - \text{SSIM}(x, y)}{2}

Arguments:
    window_size (int): the size of the kernel.
    max_val (float): the dynamic range of the images. Default: 1.
    reduction (str, optional): Specifies the reduction to apply to the
     output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
     'mean': the sum of the output will be divided by the number of elements
     in the output, 'sum': the output will be summed. Default: 'none'.

Returns:
    Tensor: the ssim index.

Shape:
    - Input: :math:`(B, C, H, W)`
    - Target :math:`(B, C, H, W)`
    - Output: scale, if reduction is 'none', then :math:`(B, C, H, W)`

Examples::

    >>> input1 = torch.rand(1, 4, 5, 5)
    >>> input2 = torch.rand(1, 4, 5, 5)
    >>> ssim = kornia.losses.SSIM(5, reduction='none')
    >>> loss = ssim(input1, input2)  # 1x4x5x5
"""
import torch
from kornia.filters import get_gaussian_kernel2d, filter2d
import torch.nn.functional as F
import torch.nn as nn

class SSIM(nn.Module):
    def __init__(self,
            window_size=11,
            reduction = 'mean',
            max_val = 1.0):
        super(SSIM,self).__init__()
        self.window_size: int = window_size
        self.max_val: float = max_val
        self.reduction: str = reduction
    
        self.window = get_gaussian_kernel2d(
            (window_size, window_size), (1.5, 1.5))
        self.padding: int = self.compute_zero_padding(window_size)
    
        self.C1: float = (0.01 * self.max_val) ** 2
        self.C2: float = (0.03 * self.max_val) ** 2
    
    @staticmethod
    def compute_zero_padding(kernel_size: int) -> int:
        """Computes zero padding."""
        return (kernel_size - 1) // 2
    
    def filter2D(
            self,
            input,
            kernel,
            channel):
        return F.conv2d(input, kernel, padding=self.padding, groups=channel)
    
    def forward(  # type: ignore
            self,
            img1,
            img2):


        # prepare kernel
        b, c, h, w = img1.shape
        tmp_kernel = self.window.to(img1.device).to(img1.dtype)
        kernel = tmp_kernel.repeat(c, 1, 1, 1)
    
        # compute local mean per channel
        mu1 = self.filter2D(img1, kernel, c)
        mu2 = self.filter2D(img2, kernel, c)
    
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
    
        # compute local sigma per channel
        sigma1_sq = self.filter2D(img1 * img1, kernel, c) - mu1_sq
        sigma2_sq = self.filter2D(img2 * img2, kernel, c) - mu2_sq
        sigma12 = self.filter2D(img1 * img2, kernel, c) - mu1_mu2
    
        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / \
            ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))
    
        loss = torch.clamp(torch.tensor(1.) - ssim_map, min=0, max=1) / 2.
    
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'none':
            pass
        return loss