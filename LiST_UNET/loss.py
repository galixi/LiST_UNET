
import torch
import torch.nn as nn
import pytorch_ssim
import numpy as np
import scipy.io as io


class Sum:
    def __init__(self, q, x, y, z):
        self.criterion = q
        self.fake_B = torch.squeeze(x)
        self.ground_B = torch.squeeze(y)
        self.var_mask = torch.squeeze(z)

    def remove_nan(self):
        m = ((self.fake_B - self.ground_B).abs() / self.ground_B) * self.var_mask
        m = torch.where(torch.isinf(m), torch.full_like(m, 1), m)
        m = torch.where(torch.isnan(m), torch.full_like(m, 0), m)
        return m.sum()

    def fl(self, loss):
        return -(1 - loss) * torch.log(loss)

    def mse(self):
        self.backloss_comp = (
            lambda fake_B, ground_B, var_mask:
            ((((fake_B - ground_B) / ground_B).abs()) * var_mask).sum() / var_mask.sum()
        )
        self.pixloss = self.backloss_comp(self.fake_B, self.ground_B, self.var_mask)
        return self.pixloss

    def add(self):
        self.dif = ((self.fake_B - self.ground_B) / self.ground_B).abs()

        if self.criterion == 'L2re':
            self.backloss_comp = (
                lambda fake_B, ground_B, var_mask:
                (((((fake_B - ground_B) / ground_B) ** 2) * var_mask).sum() / var_mask.sum())
            )

        elif self.criterion == 'L1retest':
            mask = self.var_mask.bool()
            GT = torch.masked_select(self.ground_B, mask)
            pred = torch.masked_select(self.fake_B, mask)

            loss = nn.L1Loss()
            self.loss = loss(pred, GT)

            ssim_loss = pytorch_ssim.SSIM3D(window_size=11)

            ground_B = self.ground_B * self.var_mask
            ground_B = ground_B.unsqueeze(0).unsqueeze(0)

            pred_B = self.fake_B * self.var_mask
            pred_B = pred_B.unsqueeze(0).unsqueeze(0)

            self.ssimloss = 1 - ssim_loss(ground_B, pred_B)
            self.loss_final = self.loss + self.ssimloss

        elif self.criterion == 'ssim':
            ssim_loss = pytorch_ssim.SSIM3D(window_size=11)

            ground_B = self.ground_B * self.var_mask
            ground_B = ground_B.unsqueeze(0).unsqueeze(0)

            pred_B = self.fake_B * self.var_mask
            pred_B = pred_B.unsqueeze(0).unsqueeze(0)

            self.ssimloss = 1 - ssim_loss(ground_B, pred_B)
            self.loss = self.ssimloss

        elif self.criterion == 'L1.5re':
            self.backloss_comp = (
                lambda fake_B, ground_B, var_mask:
                ((self.dif ** 1.5) * var_mask).sum() / var_mask.sum()
            )

        else:
            raise ValueError('backloss criterion %s not recognized' % self.criterion)

        return self.loss_final


