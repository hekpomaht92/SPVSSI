import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
np.random.seed(42)


class conv(nn.Module):
    '''conv => BN => Scale => ReLU => MaxPool* => Drop'''
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, dilation, drop_rate, use_pool, use_bias):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation, bias=use_bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU())
        if use_pool:
            self.conv.add_module('pool', nn.MaxPool2d(2, stride=2, padding=0))
        self.conv.add_module('drop', nn.Dropout2d(drop_rate))
        
    def forward(self, x):
        x = self.conv(x)
        return x


class deconv(nn.Module):
    '''convTranspose => BN => Scale => ReLU => Drop'''
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, dilation, drop_rate, use_bias):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, dilation, bias=use_bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Dropout2d(drop_rate))
        
    def forward(self, x):
        x = self.conv(x)
        return x


class slicing(nn.Module):
    '''slice W-dimention, conv it'''
    def __init__(self, in_ch, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding),
            nn.ReLU())
    
    def forward(self, x):
        down_convs = []
        up_convs = []
        #downside convolution
        for i in range(59):
            slice_tensor = torch.unsqueeze(x[:,:,:,i], 3)
            slice_tensor = self.conv(slice_tensor)
            slice_tensor += torch.unsqueeze(x[:,:,:,i+1], 3)
            down_convs.append(slice_tensor)
        down_convs.append(self.conv(torch.unsqueeze(x[:,:,:,-1], 3)))
        #upside convolution
        for i in range(59, 0, -1):
            slice_tensor = self.conv(down_convs[i])
            slice_tensor += down_convs[i-1]
            up_convs.append(slice_tensor)
        up_convs = up_convs[::-1]
        up_convs.append(down_convs[-1])
        #concatenation
        cat = torch.cat(up_convs, 3)  
        return cat


class Metrics:
    '''Compute tpr, fpr, fpr, fnr and balanced accuracy'''
    @classmethod
    def compute_tpr(cls, y_true, y_pred):
        
        y_true = y_true.to('cpu').numpy()
        y_pred = y_pred.to('cpu').numpy()

        y_pred_pos = y_pred
        y_pred_neg = 1 - y_pred
        y_true_pos = y_true

        tp = np.sum(y_pred_pos * y_true_pos + 1e-10)
        fn = np.sum(y_pred_neg * y_true_pos + 1e-10)
        return tp / (tp + fn)

    @staticmethod
    def _compute_tpr(y_true, y_pred):
        
        y_true = y_true.to('cpu').numpy()
        y_pred = y_pred.to('cpu').numpy()

        y_pred_pos = y_pred
        y_pred_neg = 1 - y_pred
        y_true_pos = y_true

        tp = np.sum(y_pred_pos * y_true_pos + 1e-10)
        fn = np.sum(y_pred_neg * y_true_pos + 1e-10)
        return tp / (tp + fn)

    @classmethod
    def compute_tnr(cls, y_true, y_pred):
        
        y_true = y_true.to('cpu').numpy()
        y_pred = y_pred.to('cpu').numpy()

        y_pred_pos = y_pred
        y_pred_neg = 1 - y_pred
        y_true_neg = 1 - y_true

        tn = np.sum(y_pred_neg * y_true_neg + 1e-10)
        fp = np.sum(y_pred_pos * y_true_neg + 1e-10)
        return tn / (tn + fp)

    @staticmethod
    def _compute_tnr(y_true, y_pred):
        
        y_true = y_true.to('cpu').numpy()
        y_pred = y_pred.to('cpu').numpy()

        y_pred_pos = y_pred
        y_pred_neg = 1 - y_pred
        y_true_neg = 1 - y_true

        tn = np.sum(y_pred_neg * y_true_neg + 1e-10)
        fp = np.sum(y_pred_pos * y_true_neg + 1e-10)
        return tn / (tn + fp)

    @classmethod
    def compute_ppv(cls, y_true, y_pred):
        
        y_true = y_true.to('cpu').numpy()
        y_pred = y_pred.to('cpu').numpy()

        y_pred_pos = y_pred
        y_true_pos = y_true
        y_true_neg = 1 - y_true

        tp = np.sum(y_pred_pos * y_true_pos + 1e-10)
        fp = np.sum(y_pred_pos * y_true_neg + 1e-10)
        return tp / (tp + fp)

    @classmethod
    def compute_npv(cls, y_true, y_pred):
        
        y_true = y_true.to('cpu').numpy()
        y_pred = y_pred.to('cpu').numpy()

        y_pred_neg = 1 - y_pred
        y_true_pos = y_true
        y_true_neg = 1 - y_true

        tn = np.sum(y_pred_neg * y_true_neg + 1e-10)
        fn = np.sum(y_pred_neg * y_true_pos + 1e-10)
        return tn / (tn + fn)

    @classmethod
    def balanced_accuracy(cls, y_true, y_pred):
        
        tpr = cls._compute_tpr(y_true, y_pred)
        tnr = cls._compute_tnr(y_true, y_pred)
        return (tpr+tnr)/2


if __name__ == '__main__':
    test_value = torch.ones((1, 64, 24, 60), dtype=torch.float)
    layer = slice()
    ans = layer(test_value)
    print(ans.size())

