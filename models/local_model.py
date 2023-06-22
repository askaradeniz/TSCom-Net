from pickle import TRUE
from turtle import shape
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import random

def get_models():
    return {'IFNetPlusv2': IFNetPlusv2, 'JIN': JIN}

class IFNetPlusv2(nn.Module):
    def __init__(self, hidden_dim=256):
        super(IFNetPlusv2, self).__init__()

        self.conv_in = nn.Conv3d(3, 16, 3, padding=1)  # out: 256 ->m.p. 128
        self.conv_0 = nn.Conv3d(16, 32, 3, padding=1)  # out: 128
        self.conv_0_1 = nn.Conv3d(32, 32, 3, padding=1)  # out: 128 ->m.p. 64
        self.conv_1 = nn.Conv3d(32, 64, 3, padding=1)  # out: 64
        self.conv_1_1 = nn.Conv3d(64, 64, 3, padding=1)  # out: 64 -> mp 32
        self.conv_2 = nn.Conv3d(64, 128, 3, padding=1)  # out: 32
        self.conv_2_1 = nn.Conv3d(128, 128, 3, padding=1)  # out: 32 -> mp 16
        self.conv_3 = nn.Conv3d(128, 128, 3, padding=1)  # out: 16
        self.conv_3_1 = nn.Conv3d(128, 128, 3, padding=1)  # out: 16 -> mp 8
        self.conv_4 = nn.Conv3d(128, 128, 3, padding=1)  # out: 8
        self.conv_4_1 = nn.Conv3d(128, 128, 3, padding=1)  # out: 8
        self.actvn = nn.ReLU()

        # ## Shape encoder
        self.conv_in_shape = nn.Conv3d(1, 16, 3, padding=1)  # out: 256 ->m.p. 128
        self.conv_0_shape = nn.Conv3d(16, 32, 3, padding=1)  # out: 128
        self.conv_0_1_shape = nn.Conv3d(32, 32, 3, padding=1)  # out: 128 ->m.p. 64
        self.conv_1_shape = nn.Conv3d(32, 64, 3, padding=1)  # out: 64
        self.conv_1_1_shape = nn.Conv3d(64, 64, 3, padding=1)  # out: 64 -> mp 32
        self.conv_2_shape = nn.Conv3d(64, 128, 3, padding=1)  # out: 32
        self.conv_2_1_shape = nn.Conv3d(128, 128, 3, padding=1)  # out: 32 -> mp 16
        self.conv_3_shape = nn.Conv3d(128, 128, 3, padding=1)  # out: 16
        self.conv_3_1_shape = nn.Conv3d(128, 128, 3, padding=1)  # out: 16 -> mp 8
        self.conv_4_shape = nn.Conv3d(128, 128, 3, padding=1)  # out: 8
        self.conv_4_1_shape = nn.Conv3d(128, 128, 3, padding=1)  # out: 8

        ## Texture decoder
        feature_size = (3 +  16 + 32 + 64 + 128 + 128 + 128) * 7 + 3
        feature_size_shape = (1 +  16 + 32 + 64 + 128 + 128 + 128) * 7 + 3
        self.fc_0 = nn.Conv1d(feature_size+feature_size_shape, hidden_dim * 2, 1)
        self.fc_1 = nn.Conv1d(hidden_dim *2, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim , hidden_dim, 1)
        self.fc_out = nn.Conv1d(hidden_dim, 3, 1)


        ## Shape decoder
        self.fc_0_shape = nn.Conv1d(feature_size_shape, hidden_dim*2, 1)
        self.fc_1_shape = nn.Conv1d(hidden_dim*2, hidden_dim, 1)
        self.fc_2_shape = nn.Conv1d(hidden_dim , hidden_dim, 1)
        self.fc_out_shape = nn.Conv1d(hidden_dim, 1, 1)
        self.fusion_layer = nn.Conv1d(feature_size+feature_size_shape, feature_size_shape, 1)

        self.maxpool = nn.MaxPool3d(2)

        self.conv_in_bn = nn.BatchNorm3d(16)
        self.conv0_1_bn = nn.BatchNorm3d(32)
        self.conv1_1_bn = nn.BatchNorm3d(64)
        self.conv2_1_bn = nn.BatchNorm3d(128)
        self.conv3_1_bn = nn.BatchNorm3d(128)
        self.conv4_1_bn = nn.BatchNorm3d(128)

        self.conv_in_bn_shape = nn.BatchNorm3d(16)
        self.conv0_1_bn_shape = nn.BatchNorm3d(32)
        self.conv1_1_bn_shape = nn.BatchNorm3d(64)
        self.conv2_1_bn_shape = nn.BatchNorm3d(128)
        self.conv3_1_bn_shape = nn.BatchNorm3d(128)
        self.conv4_1_bn_shape = nn.BatchNorm3d(128)

        self.bn_shape = nn.BatchNorm1d(feature_size)
        self.bn_texture = nn.BatchNorm1d(feature_size)


        displacment = 0.0722
        displacments = []
        displacments.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacment
                displacments.append(input)

        self.displacments = torch.Tensor(displacments).cuda()

    def encode_shape(self, p, x):
        p_features = p.transpose(1, -1)
        p = p.unsqueeze(1).unsqueeze(1)
        p = torch.cat([p + d for d in self.displacments], dim=2)
        feature_0 = F.grid_sample(x, p, padding_mode='border')

        net = self.actvn(self.conv_in_shape(x))
        net = self.conv_in_bn_shape(net)
        feature_1 = F.grid_sample(net, p, padding_mode='border')
        net = self.maxpool(net) #out 128

        net = self.actvn(self.conv_0_shape(net))
        net = self.actvn(self.conv_0_1_shape(net))
        net = self.conv0_1_bn_shape(net)
        feature_2 = F.grid_sample(net, p, padding_mode='border')
        net = self.maxpool(net) #out 64

        net = self.actvn(self.conv_1_shape(net))
        net = self.actvn(self.conv_1_1_shape(net))
        net = self.conv1_1_bn_shape(net)
        feature_3 = F.grid_sample(net, p, padding_mode='border')
        net = self.maxpool(net)

        net = self.actvn(self.conv_2_shape(net))
        net = self.actvn(self.conv_2_1_shape(net))
        net = self.conv2_1_bn_shape(net)
        feature_4 = F.grid_sample(net, p, padding_mode='border')
        net = self.maxpool(net)

        net = self.actvn(self.conv_3_shape(net))
        net = self.actvn(self.conv_3_1_shape(net))
        net = self.conv3_1_bn_shape(net)
        feature_5 = F.grid_sample(net, p, padding_mode='border')
        net = self.maxpool(net)

        net = self.actvn(self.conv_4_shape(net))
        net = self.actvn(self.conv_4_1_shape(net))
        net = self.conv4_1_bn_shape(net)
        feature_6 = F.grid_sample(net, p, padding_mode='border')

        features = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6),
                             dim=1)  # (B, features, 1,7,sample_num)
        shape = features.shape
        features = torch.reshape(features,(shape[0], shape[1] * shape[3], shape[4]))  # (B, featues_per_sample, samples_num)
        features = torch.cat((features, p_features), dim=1)  # (B, featue_size, samples_num) samples_num 0->0,...,N->N
        return features

    def encode_color(self, p, x):
        p_features = p.transpose(1, -1)
        p = p.unsqueeze(1).unsqueeze(1)
        p = torch.cat([p + d for d in self.displacments], dim=2)
        feature_0 = F.grid_sample(x, p, padding_mode='border')

        net = self.actvn(self.conv_in(x))
        net = self.conv_in_bn(net)
        feature_1 = F.grid_sample(net, p, padding_mode='border')
        net = self.maxpool(net) #out 128

        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        feature_2 = F.grid_sample(net, p, padding_mode='border')
        net = self.maxpool(net) #out 64

        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        feature_3 = F.grid_sample(net, p, padding_mode='border')
        net = self.maxpool(net)

        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        feature_4 = F.grid_sample(net, p, padding_mode='border')
        net = self.maxpool(net)

        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        net = self.conv3_1_bn(net)
        feature_5 = F.grid_sample(net, p, padding_mode='border')
        net = self.maxpool(net)

        net = self.actvn(self.conv_4(net))
        net = self.actvn(self.conv_4_1(net))
        net = self.conv4_1_bn(net)
        feature_6 = F.grid_sample(net, p, padding_mode='border')

        features = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6),
                             dim=1)  # (B, features, 1,7,sample_num)
        shape = features.shape
        features = torch.reshape(features,(shape[0], shape[1] * shape[3], shape[4]))  # (B, featues_per_sample, samples_num)
        features = torch.cat((features, p_features), dim=1)  # (B, featue_size, samples_num) samples_num 0->0,...,N->N

        
        return features

    def decode_shape(self, x):
        net = self.actvn(self.fc_0_shape(x))
        net = self.actvn(self.fc_1_shape(net))
        net = self.actvn(self.fc_2_shape(net))
        out_shape = self.fc_out_shape(net)
        out_shape = out_shape.squeeze(1)

        return out_shape

    def decode_color(self, x):
        net = self.actvn(self.fc_0(x))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        out_rgb = self.fc_out(net)
        out_rgb = out_rgb.squeeze(1)
        out_rgb =  out_rgb.transpose(-1,-2)

        return out_rgb

    def forward(self, p, x, x_shape):
        
        shape_features = self.encode_shape(p, x_shape)
        texture_features = self.encode_texture(p, x)

        out_shape = self.decode_shape(shape_features, texture_features)

        out_rgb = self.decode_texture(shape_features, texture_features)

        return out_rgb, out_shape

    def forward_shape(self, p, x_shape):
        net = self.encode_shape(p, x_shape)
        out_shape = self.decode_shape(net)

        return out_shape

    def forward_color(self, p, x, x_shape):
        net = self.encode_shape(p, x_shape)
        net_tex = self.encode_color(p, x)
        net = torch.cat((net, net_tex),
                             dim=1)

        out_rgb = self.decode_color(net)

        return out_rgb


class JIN(nn.Module):
    def __init__(self, hidden_dim=256):
        super(JIN, self).__init__()
        self.shape_network = ImplicitNetwork(in_channels=1, out_channels=1, conv_dim=16, hidden_dim=256)
        self.color_network = ImplicitNetwork(in_channels=3, out_channels=3, conv_dim=16, hidden_dim=256, multifeat_decoder=True, additional_feat_size=self.shape_network.feature_size)

    def forward_shape(self, p, x_shape):
        shape_out = self.shape_network(p, x_shape)
        return shape_out

    def forward_color(self, p, x, x_shape):
        net = self.shape_network.encode(p, x_shape)
        net_tex = self.color_network.encode(p, x)
        net = torch.cat((net, net_tex),
                             dim=1)
        net = self.color_network.decode(net)
        color_out = net.transpose(-1,-2)
        return color_out


class ImplicitNetwork(nn.Module):
    def __init__(self, hidden_dim=256, conv_dim=16, in_channels=3, out_channels=1, multifeat_decoder=False, additional_feat_size=0, displacment=0.0722):
        super(ImplicitNetwork, self).__init__()

        self.multifeat_decoder = multifeat_decoder

        ##  Encoder
        self.conv_in = nn.Conv3d(in_channels, conv_dim, 3, padding=1)  # out: 256 ->m.p. 128
        self.conv_0 = nn.Conv3d(conv_dim, conv_dim*2, 3, padding=1)  # out: 128
        self.conv_0_1 = nn.Conv3d(conv_dim*2, conv_dim*2, 3, padding=1)  # out: 128 ->m.p. 64
        self.conv_1 = nn.Conv3d(conv_dim*2, conv_dim*4, 3, padding=1)  # out: 64
        self.conv_1_1 = nn.Conv3d(conv_dim*4, conv_dim*4, 3, padding=1)  # out: 64 -> mp 32
        self.conv_2 = nn.Conv3d(conv_dim*4, conv_dim*8, 3, padding=1)  # out: 32
        self.conv_2_1 = nn.Conv3d(conv_dim*8, conv_dim*8, 3, padding=1)  # out: 32 -> mp 16
        self.conv_3 = nn.Conv3d(conv_dim*8, conv_dim*8, 3, padding=1)  # out: 16
        self.conv_3_1 = nn.Conv3d(conv_dim*8, conv_dim*8, 3, padding=1)  # out: 16 -> mp 8
        self.conv_4 = nn.Conv3d(conv_dim*8, conv_dim*8, 3, padding=1)  # out: 8
        self.conv_4_1 = nn.Conv3d(conv_dim*8, conv_dim*8, 3, padding=1)  # out: 8

        ## Decoder
        feature_size = (in_channels +  conv_dim + conv_dim*2 + conv_dim*4 + conv_dim*8 + conv_dim*8 + conv_dim*8) * ((6 * levels)+1) + 3
        self.feature_size = feature_size
        if self.multifeat_decoder == True:
            self.fc_0 = nn.Conv1d(feature_size + additional_feat_size, hidden_dim * 2, 1)
        else:
            
            self.fc_0 = nn.Conv1d(feature_size, hidden_dim * 2, 1)
        self.fc_1 = nn.Conv1d(hidden_dim *2, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim , hidden_dim, 1)
        self.fc_out = nn.Conv1d(hidden_dim, out_channels, 1)
        self.actvn = nn.ReLU()
        self.dropout = nn.Dropout(0.1)


        self.maxpool = nn.MaxPool3d(2)

        self.conv_in_bn = nn.BatchNorm3d(conv_dim)
        self.conv0_1_bn = nn.BatchNorm3d(conv_dim*2)
        self.conv1_1_bn = nn.BatchNorm3d(conv_dim*4)
        self.conv2_1_bn = nn.BatchNorm3d(conv_dim*8)
        self.conv3_1_bn = nn.BatchNorm3d(conv_dim*8)
        self.conv4_1_bn = nn.BatchNorm3d(conv_dim*8)


        displacements = []
        displacements.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacment
                displacements.append(input)
        self.displacements = torch.Tensor(displacements).cuda()
        

    def encode(self, p, x):
        p_features = p.transpose(1, -1)
        p = p.unsqueeze(1).unsqueeze(1)
        p = torch.cat([p + d for d in self.displacements], dim=2)

        feature_0 = F.grid_sample(x, p, padding_mode='border')

        net = self.actvn(self.conv_in(x))
        net = self.conv_in_bn(net)
        feature_1 = F.grid_sample(net, p, padding_mode='border')
        net = self.maxpool(net) #out 128

        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        feature_2 = F.grid_sample(net, p, padding_mode='border')
        net = self.maxpool(net) #out 64

        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        feature_3 = F.grid_sample(net, p, padding_mode='border')
        net = self.maxpool(net)

        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        feature_4 = F.grid_sample(net, p, padding_mode='border')
        net = self.maxpool(net)

        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        net = self.conv3_1_bn(net)
        feature_5 = F.grid_sample(net, p, padding_mode='border')
        net = self.maxpool(net)

        net = self.actvn(self.conv_4(net))
        net = self.actvn(self.conv_4_1(net))
        net = self.conv4_1_bn(net)
        feature_6 = F.grid_sample(net, p, padding_mode='border')

        features = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6),
                             dim=1)  # (B, features, 1,7,sample_num)
        shape = features.shape
        features = torch.reshape(features,(shape[0], shape[1] * shape[3], shape[4]))  # (B, featues_per_sample, samples_num)
        features = torch.cat((features, p_features), dim=1)  # (B, featue_size, samples_num) samples_num 0->0,...,N->N
        return features

    def decode(self, features):
        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        out = self.fc_out(net)
        out = out.squeeze(1)

        return out

    def forward(self, p, x):
        features = self.encode(p, x)
        out = self.decode(features)
        return out