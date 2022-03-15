# -*- coding = utf-8 -*-
# @time:2021/4/29 下午8:51
# Author:jy
# @File:test_detransformer.py
# @Software:PyCharm


import torch
from torch import nn
from transformer.position_encoding import PositionEmbeddingSine
from transformer.deformable_transformer import DeformableTransformer


class Transform(nn.Module):
    def __init__(self, num_feature_levels=1, hidden_dim= 128,num_feats=16):
        super(Transform, self).__init__()
        self.position_embed = PositionEmbeddingSine(num_pos_feats=num_feats)
        self.encoder_Detrans = DeformableTransformer(d_model=hidden_dim, dim_feedforward=hidden_dim*4, dropout=0.1, activation='gelu',\
                                                     num_feature_levels=num_feature_levels, nhead=8, num_encoder_layers=12, enc_n_points=6)
        self.conv1 = nn.Conv2d(352, 176, 3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(176)
        self.conv2 = nn.Conv2d(352, 176, 3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(176)
        self.conv3 = nn.Conv2d(352, 176, 3, stride=1, padding=1)
        self.norm3 = nn.BatchNorm2d(176)
        self.relu = nn.ReLU()
    def posi_mask(self, x):

        x_fea = []

        x_posemb = []
        masks = []
        for lvl, fea in enumerate(x):   # 对于每一层
            if lvl >= 0:
                x_fea.append(fea)
                x_posemb.append(self.position_embed(fea))
                masks.append(torch.zeros((fea.shape[0], fea.shape[2], fea.shape[3]), dtype=torch.bool).cuda())    # bs d h w

        return x_fea, masks, x_posemb

    def forward(self, inputs):
        shape0 = inputs[0].shape
        x_fea, masks, x_posemb = self.posi_mask(inputs)
        x_trans, x_trans1, x_trans2= self.encoder_Detrans(x_fea, masks, x_posemb) #,x_trans1
        # Multi-scale
        # x = x_trans[:, 6912::].transpose(-1, -2).view(inputs[-1].shape) # x_trans length: 12*24*24+6*12*12=7776
        x = x_trans[:, 0:shape0[2] * shape0[3]].transpose(-1, -2).view(inputs[0].shape)
        xx = x_trans1[:, 0:shape0[2] * shape0[3]].transpose(-1, -2).view(inputs[0].shape)
        xxx = x_trans2[:, 0:shape0[2] * shape0[3]].transpose(-1, -2).view(inputs[0].shape)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        xx = self.conv2(xx)
        xx = self.norm2(xx)
        xx = self.relu(xx)
        xxx = self.conv3(xxx)
        xxx = self.norm3(xxx)
        xxx = self.relu(xxx)
        outputt = torch.cat((x,xx,xxx,inputs[0]),dim=1)
        # outputt = x + xx + inputs[0]
        return outputt #torch.cat((x,xx,xxx,inputs[0]),dim=1)

if __name__ == '__main__':
    device = torch.device('cuda:0')
    image = torch.randn(1, 32, 128,128)
    image = image.to(device)
    TR = Transform(hidden_dim= 32, num_feats = 16).to(device)
    output = TR([image])
    a = 1