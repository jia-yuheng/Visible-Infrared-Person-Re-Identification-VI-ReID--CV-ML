import torch
import torch.nn as nn
from torch.nn import init
from resnet import resnet50, resnet18
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = in_channels // reduc_ratio

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                      padding=0),
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''
                :param x: (b, c, t, h, w)
                :return:
                '''

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        # f_div_C = torch.nn.functional.softmax(f, dim=-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)


class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class visible_part_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_part_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class thermal_part_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_part_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class base_resnet_gobal(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet_gobal, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x


class base_resnet_part(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet_part, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x


class embed_net(nn.Module):
    def __init__(self, class_num, no_local='on', gm_pool='on', arch='resnet50'):
        super(embed_net, self).__init__()

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = base_resnet_gobal(arch=arch)
        self.non_local = no_local
        if self.non_local == 'on':
            layers = [3, 4, 6, 3]
            non_layers = [0, 2, 3, 0]
            self.NL_1 = nn.ModuleList(
                [Non_local(256) for i in range(non_layers[0])])
            self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
            self.NL_2 = nn.ModuleList(
                [Non_local(512) for i in range(non_layers[1])])
            self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
            self.NL_3 = nn.ModuleList(
                [Non_local(1024) for i in range(non_layers[2])])
            self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
            self.NL_4 = nn.ModuleList(
                [Non_local(2048) for i in range(non_layers[3])])
            self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])

        pool_dim = 2048
        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        self.classifier = nn.Linear(pool_dim, class_num, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gm_pool = gm_pool


    def forward(self, x1, x2, modal=0):
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)

        # shared block
        if self.non_local == 'on':
            NL1_counter = 0
            if len(self.NL_1_idx) == 0: self.NL_1_idx = [-1]
            for i in range(len(self.base_resnet.base.layer1)):
                x = self.base_resnet.base.layer1[i](x)
                if i == self.NL_1_idx[NL1_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_1[NL1_counter](x)
                    NL1_counter += 1
            # Layer 2
            NL2_counter = 0
            if len(self.NL_2_idx) == 0: self.NL_2_idx = [-1]
            for i in range(len(self.base_resnet.base.layer2)):
                x = self.base_resnet.base.layer2[i](x)
                if i == self.NL_2_idx[NL2_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_2[NL2_counter](x)
                    NL2_counter += 1
            # Layer 3
            NL3_counter = 0
            if len(self.NL_3_idx) == 0: self.NL_3_idx = [-1]
            for i in range(len(self.base_resnet.base.layer3)):
                x = self.base_resnet.base.layer3[i](x)
                if i == self.NL_3_idx[NL3_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_3[NL3_counter](x)
                    NL3_counter += 1
            # Layer 4
            NL4_counter = 0
            if len(self.NL_4_idx) == 0: self.NL_4_idx = [-1]
            for i in range(len(self.base_resnet.base.layer4)):
                x = self.base_resnet.base.layer4[i](x)
                if i == self.NL_4_idx[NL4_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_4[NL4_counter](x)
                    NL4_counter += 1
        else:
            x = self.base_resnet(x)


        if self.gm_pool == 'on':
            b, c, h, w = x.shape
            x = x.view(b, c, -1)
            p = 3.0
            x_pool = (torch.mean(x ** p, dim=-1) + 1e-12) ** (1 / p)
        else:
            x_pool = self.avgpool(x)
            x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))

        feat = self.bottleneck(x_pool)

        if self.training:
            return x_pool, self.classifier(feat)
        else:
            return self.l2norm(x_pool), self.l2norm(feat)

def GEMpooling(x):
    b, c, h, w = x.shape
    x = x.view(b, c, -1)
    p = 3.0
    x = (torch.mean(x ** p, dim=-1) + 1e-12) ** (1 / p)
    return x

class GeneralizedMeanPooling(nn.Module):
    def __init__(self, norm=3, output_size=(1, 1), eps=1e-6, *args, **kwargs):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'
GeneralizedMeanPooling = GeneralizedMeanPooling()

class ContrastiveModalAligner(nn.Module):
    def __init__(self, features_dim):
        super(ContrastiveModalAligner, self).__init__()
        # 建两个全连接层（fc1和fc2）。fc1将输入特征维度减半，而fc2恢复到原始特征维度。这种设计有助于学习非线性映射，实现模态间的对齐。
        self.fc1 = nn.Linear(features_dim, features_dim // 2)
        self.conv1 = nn.Conv2d(features_dim, features_dim // 2, 1, bias=True)
        self.fc2 = nn.Linear(features_dim // 2, features_dim)
        self.conv2 = nn.Conv2d(features_dim // 2, features_dim, 1, bias=True)
        # 初始化权重。使用均值为0、标准差为0.01的正态分布进行初始化。
        self.fc1.weight.data.normal_(mean=0.0, std=0.01)
        self.fc2.weight.data.normal_(mean=0.0, std=0.01)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, visible_features, thermal_features):
        visible_features = self.avgpool(visible_features)
        thermal_features = self.avgpool(thermal_features)
        features_mix = torch.cat((visible_features, thermal_features),dim=1)
        features_mix = F.relu(self.conv1(features_mix))
        features_mix = self.conv2(features_mix)
        features_mix = features_mix.view(features_mix.size(0),features_mix.size(1))
        return features_mix


class embed_net_mix(nn.Module):
    def __init__(self, class_num, non_local='off', arch='resnet50'):
        super(embed_net_mix, self).__init__()

        self.thermal_module_gobal = thermal_module(arch=arch)
        self.thermal_module_part = thermal_part_module(arch=arch)
        self.visible_module_gobal = visible_module(arch=arch)
        self.visible_module_part = visible_part_module(arch=arch)
        self.base_resnet_gobal = base_resnet_gobal(arch=arch)
        self.base_resnet_part = base_resnet_part(arch=arch)
        self.non_local = non_local
        self.fusion = ContrastiveModalAligner(4096)

        if self.non_local == 'on':
            layers = [3, 4, 6, 3]
            non_layers = [0, 2, 3, 0]
            self.NL_1 = nn.ModuleList(
                [Non_local(256) for i in range(non_layers[0])])
            self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
            self.NL_2 = nn.ModuleList(
                [Non_local(512) for i in range(non_layers[1])])
            self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
            self.NL_3 = nn.ModuleList(
                [Non_local(1024) for i in range(non_layers[2])])
            self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
            self.NL_4 = nn.ModuleList(
                [Non_local(2048) for i in range(non_layers[3])])
            self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])

        pool_dim = 2048
        self.l2norm = Normalize(2)

        self.bottleneck = nn.BatchNorm1d(2048)
        self.bottleneck_part = nn.BatchNorm1d(2048)
        self.bottleneck_mix = nn.BatchNorm1d(4096)
        #self.bottleneck_inter_mix = nn.BatchNorm1d(4096)

        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck_part.bias.requires_grad_(False)
        self.bottleneck_mix.bias.requires_grad_(False)
        #self.bottleneck_inter_mix.bias.requires_grad_(False)


        # 定义三个分类器
        self.classifier_gobal = nn.Linear(2048, class_num, bias=False)
        self.classifier_part = nn.Linear(2048, class_num, bias=False)
        self.classifier_mix = nn.Linear(4096, class_num, bias=False)
        #self.classifier_mix_downsample = nn.Linear(4096, 2048, bias=True) 好像没有用到，所以先注释掉了。



        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_part.apply(weights_init_kaiming)
        self.bottleneck_mix.apply(weights_init_kaiming)
        #self.bottleneck_inter_mix.apply(weights_init_kaiming)

        self.classifier_gobal.apply(weights_init_classifier)
        self.classifier_part.apply(weights_init_classifier)
        self.classifier_mix.apply(weights_init_classifier)

        #self.classifier_mix_downsample.apply(weights_init_classifier)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(4096, 4096, 1, bias=False)


    def forward(self, x1, x2, upper_part1, upper_part2, modal=0):
        if modal == 0:
            x1 = self.visible_module_gobal(x1)
            x2 = self.thermal_module_gobal(x2)
            x1_part1 = self.visible_module_part(upper_part1)
            x2_part2 = self.thermal_module_part(upper_part2)

            x = torch.cat((x1, x2), 0)
            x_part = torch.cat((x1_part1, x2_part2), 0)
        elif modal == 1:
            x = self.visible_module_gobal(x1)
            x_part = self.visible_module_part(upper_part1)
        elif modal == 2:
            x = self.thermal_module_gobal(x2)
            x_part = self.thermal_module_part(upper_part2)

        if self.non_local == 'on':
            NL1_counter = 0
            if len(self.NL_1_idx) == 0: self.NL_1_idx = [-1]
            for i in range(len(self.base_resnet_gobal.base.layer1)):
                x = self.base_resnet_gobal.base.layer1[i](x)
                if i == self.NL_1_idx[NL1_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_1[NL1_counter](x)
                    NL1_counter += 1

            # Layer 2
            NL2_counter = 0
            if len(self.NL_2_idx) == 0: self.NL_2_idx = [-1]
            for i in range(len(self.base_resnet_gobal.base.layer2)):
                x = self.base_resnet_gobal.base.layer2[i](x)
                if i == self.NL_2_idx[NL2_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_2[NL2_counter](x)
                    NL2_counter += 1

            # Layer 3
            NL3_counter = 0
            if len(self.NL_3_idx) == 0: self.NL_3_idx = [-1]
            for i in range(len(self.base_resnet_gobal.base.layer3)):
                x = self.base_resnet_gobal.base.layer3[i](x)
                if i == self.NL_3_idx[NL3_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_3[NL3_counter](x)
                    NL3_counter += 1

            # Layer 4
            NL4_counter = 0
            if len(self.NL_4_idx) == 0: self.NL_4_idx = [-1]
            for i in range(len(self.base_resnet_gobal.base.layer4)):
                x = self.base_resnet_gobal.base.layer4[i](x)
                if i == self.NL_4_idx[NL4_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_4[NL4_counter](x)
                    NL4_counter += 1

            #local Layer1
            NL1_counter = 0
            if len(self.NL_1_idx) == 0: self.NL_1_idx = [-1]
            for i in range(len(self.base_resnet_part.base.layer1)):
                x_part = self.base_resnet_part.base.layer1[i](x_part)
                if i == self.NL_1_idx[NL1_counter]:
                    _, C, H, W = x_part.shape
                    x_part = self.NL_1[NL1_counter](x_part)
                    NL1_counter += 1

            # local Layer2
            NL2_counter = 0
            if len(self.NL_2_idx) == 0: self.NL_2_idx = [-1]
            for i in range(len(self.base_resnet_part.base.layer2)):
                x_part = self.base_resnet_part.base.layer2[i](x_part)
                if i == self.NL_2_idx[NL2_counter]:
                    _, C, H, W = x_part.shape
                    x_part = self.NL_2[NL2_counter](x_part)
                    NL2_counter += 1


            # local Layer 3
            NL3_counter = 0
            if len(self.NL_3_idx) == 0: self.NL_3_idx = [-1]
            for i in range(len(self.base_resnet_part.base.layer3)):
                x_part = self.base_resnet_part.base.layer3[i](x_part)
                if i == self.NL_3_idx[NL3_counter]:
                    _, C, H, W = x_part.shape
                    x_part = self.NL_3[NL3_counter](x_part)
                    NL3_counter += 1

            # local Layer 4
            NL4_counter = 0
            if len(self.NL_4_idx) == 0: self.NL_4_idx = [-1]
            for i in range(len(self.base_resnet_part.base.layer4)):
                x_part = self.base_resnet_part.base.layer4[i](x_part)
                if i == self.NL_4_idx[NL4_counter]:
                    _, C, H, W = x_part.shape
                    x_part = self.NL_4[NL4_counter](x_part)
                    NL4_counter += 1
        else:
            x = self.base_resnet_gobal(x)
            x_part = self.base_resnet_part(x_part)


        x_pool_gobal = GEMpooling(x)
        x_pool_part = GEMpooling(x_part)
        if self.training:
            x_v_gobal, x_t_gobal = torch.split(x_pool_gobal, [32,32], dim=0)
            x_v_part, x_t_part = torch.split(x_pool_part, [32,32], dim=0)

            x_v_inter = torch.cat((x_v_gobal,x_t_part),dim=0)
            x_t_inter = torch.cat((x_t_gobal,x_v_part),dim=0)

            intra_mix = torch.cat((x_pool_gobal,x_pool_part), dim=1)
            inter_mix = torch.cat((x_v_inter,x_t_inter), dim=1)
        else:
            intra_mix = torch.cat((x_pool_gobal,x_pool_part), dim=1)
            inter_mix = torch.cat((x_pool_gobal,x_pool_part), dim=1)

        # 对池化后的特征进行 Batch Normalization：
        feat_gobal = self.bottleneck(x_pool_gobal)
        feat_part = self.bottleneck_part(x_pool_part)
        feat_intra_mix = self.bottleneck_mix(intra_mix)
        feat_inter_mix = self.bottleneck_mix(inter_mix)

        if self.training:
            return (x_pool_gobal, x_pool_part, intra_mix, inter_mix,
                    self.classifier_gobal(feat_gobal), self.classifier_part(feat_part),
                    self.classifier_mix(feat_intra_mix), self.classifier_mix(feat_inter_mix))
        else:
            return self.l2norm(x_pool_gobal), self.l2norm(x_pool_part), self.l2norm(intra_mix), self.l2norm(
                feat_gobal), self.l2norm(feat_part), self.l2norm(feat_intra_mix)
