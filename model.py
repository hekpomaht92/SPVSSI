import torch
import torch.nn as nn
from parts import conv, deconv, slicing
from config import Config
import multiprocessing
from collections import OrderedDict

multiprocessing.set_start_method('spawn', True)
cfg = Config()

def generate_model(pretrained_weights=None):

    model = DarkSCNN(pretrained_weights)
    if pretrained_weights != None:
        model.load_state_dict(torch.load(pretrained_weights))
    return model


class DarkSCNN(nn.Module):

    def __init__(self, pretrained_weights):
        super().__init__()
        self.conv1 = nn.Sequential(OrderedDict([('conv1_1', conv(3, 16, 3, 1, 1, 1, cfg.drop_rate, True, True)),\
                                                ('conv1_2', conv(16, 32, 3, 1, 1, 1, cfg.drop_rate, True, True))]))
        self.conv3 = nn.Sequential(OrderedDict([('conv3_1', conv(32, 64, 3, 1, 1, 1, cfg.drop_rate, False, True)),\
                                                ('conv3_2', conv(64, 32, 1, 1, 0, 1, cfg.drop_rate, False, True)),\
                                                ('conv3_3', conv(32, 64, 3, 1, 1, 1, cfg.drop_rate, True, True))]))
        self.conv4 = nn.Sequential(OrderedDict([('conv4_1', conv(64, 128, 3, 1, 1, 1, cfg.drop_rate, False, True)),\
                                                ('conv4_2', conv(128, 64, 1, 1, 0, 1, cfg.drop_rate, False, True)),\
                                                ('conv4_3', conv(64, 128, 3, 1, 1, 1, cfg.drop_rate, True, True))]))
        self.conv5 = nn.Sequential(OrderedDict([('conv5_1', conv(128, 256, 3, 1, 1, 1, cfg.drop_rate, False, True)),\
                                                ('conv5_2', conv(256, 128, 1, 1, 0, 1, cfg.drop_rate, False, True)),\
                                                ('conv5_3', conv(128, 256, 3, 1, 1, 1, cfg.drop_rate, False, True)),\
                                                ('conv5_4', conv(256, 128, 1, 1, 0, 1, cfg.drop_rate, False, True)),\
                                                ('conv5_5', conv(128, 256, 3, 1, 1, 1, cfg.drop_rate, True, True))]))                                      
        self.conv6 = nn.Sequential(OrderedDict([('conv6_1', conv(256, 512, 3, 1, 2, 2, cfg.drop_rate, False, True)),\
                                                ('conv6_2', conv(512, 256, 1, 1, 0, 1, cfg.drop_rate, False, True)),\
                                                ('conv6_3', conv(256, 512, 3, 1, 2, 2, cfg.drop_rate, False, True)),\
                                                ('conv6_4', conv(512, 256, 1, 1, 0, 1, cfg.drop_rate, False, True)),\
                                                ('conv6_5', conv(256, 512, 3, 1, 1, 1, cfg.drop_rate, False, True))]))
        self.conv7 = nn.Sequential(OrderedDict([('conv7_1', conv(512, 512, 3, 1, 1, 1, cfg.drop_rate, False, True)),\
                                                ('conv7_2', conv(512, 512, 3, 1, 1, 1, cfg.drop_rate, False, True))]))
        self.upsample1 = nn.Sequential(OrderedDict([('reduce1', conv(768, 128, 3, 1, 1, 1, cfg.drop_rate, False, True)),\
                                                    ('deconv1', deconv(128, 64, 3, 2, 1, 1, cfg.drop_rate, True))]))
        self.reorg4 = conv(128, 64, 3, 1, 1, 1, cfg.drop_rate, False, False)
        self.reduse2 = conv(128, 64, 3, 1, 1, 1, cfg.drop_rate, False, False)
        self.slice = slicing(64, (9, 1), 1, (4, 0))
        self.deconv_out = nn.Sequential(nn.Conv2d(64, 64, 1, 1, bias=False),\
                                        nn.ConvTranspose2d(64, 32, 16, 8, 4),\
                                        nn.ConvTranspose2d(32, 16, 6, 2, 2),\
                                        nn.Conv2d(16, cfg.n_classes, 3, 1, 1, bias=False))
        
        if pretrained_weights == None:
            self.conv1.apply(self.init_weights)
            self.conv3.apply(self.init_weights)
            self.conv4.apply(self.init_weights)
            self.conv5.apply(self.init_weights)
            self.conv6.apply(self.init_weights)
            self.conv7.apply(self.init_weights)
            self.upsample1.apply(self.init_weights)
            self.reorg4.apply(self.init_weights)
            self.reduse2.apply(self.init_weights)
            self.slice.apply(self.init_weights)
            self.deconv_out.apply(self.init_weights)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv3(x)
        x_4 = self.conv4(x)
        x_5 = self.conv5(x_4)
        x = self.conv6(x_5)
        x_7 = self.conv7(x)
        cat_8 = torch.cat((x_5, x_7), dim=1)
        del x_5
        del x_7
        up1 = self.upsample1(cat_8)
        del cat_8
        reorg4 = self.reorg4(x_4)
        del x_4
        cat_4 = torch.cat((reorg4, up1), dim=1)
        del reorg4
        del up1
        x = self.reduse2(cat_4)
        del cat_4
        x = self.slice(x)
        x = self.deconv_out(x)

        return x
    
    def init_weights(self, m):
        if (type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d) and m.bias is not None:
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.)
        elif (type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d) and m.bias is None:
            torch.nn.init.xavier_normal_(m.weight)


if __name__ == '__main__':
    test_value = torch.ones((1, 3, 384, 960), dtype=torch.float)
    model = generate_model()
    ans = model(test_value)
    print(ans.size())