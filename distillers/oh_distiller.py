# Original Repo:
# https://github.com/clovaai/overhaul-distillation
# @inproceedings{heo2019overhaul,
#  title={A Comprehensive Overhaul of Feature Distillation},
#  author={Heo, Byeongho and Kim, Jeesoo and Yun, Sangdoo and Park, Hyojin
#  and Kwak, Nojun and Choi, Jin Young},
#  booktitle = {International Conference on Computer Vision (ICCV)},
#  year={2019}
# }

import math
import torch
import torch.nn as nn
from scipy.stats import norm
from trainer import BaseTrainer


def distillation_loss(source, target, margin):
    loss = ((source - margin)**2 * ((source > margin) & (target <= margin)).float() +
            (source - target)**2 * ((source > target) & (target > margin) & (target <= 0)).float() +
            (source - target)**2 * (target > 0).float())
    return torch.abs(loss).sum()


def build_feature_connector(t_channel, s_channel):
    C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
         nn.BatchNorm2d(t_channel)]

    for m in C:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return nn.Sequential(*C)


def get_margin_from_BN(bn):
    margin = []
    std = bn.weight.data
    mean = bn.bias.data
    for (s, m) in zip(std, mean):
        s = abs(s.item())
        m = m.item()
        if norm.cdf(-m / s) > 0.001:
            margin.append(- s * math.exp(- (m / s) ** 2 / 2) /
                          math.sqrt(2 * math.pi) / norm.cdf(-m / s) + m)
        else:
            margin.append(-3 * s)

    return torch.FloatTensor(margin).to(std.device)


class Distiller(nn.Module):
    def __init__(self, t_net, s_net):
        super(Distiller, self).__init__()

        t_channels = t_net.get_channel_num()
        s_channels = s_net.get_channel_num()

        self.Connectors = nn.ModuleList(
            [build_feature_connector(t, s) for t, s in zip(t_channels, s_channels)])

        teacher_bns = t_net.get_bn_before_relu()
        margins = [get_margin_from_BN(bn) for bn in teacher_bns]
        for i, margin in enumerate(margins):
            self.register_buffer('margin%d' % (
                i + 1), margin.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach())

        self.s_net = s_net
        self.t_net = t_net

    def forward(self, x, is_loss=False):

        t_feats, t_pool, t_out = self.t_net(x, is_feat=True)
        s_feats, s_pool, s_out = self.s_net(x, is_feat=True)
        t_feats_num = len(t_feats)
        s_feats_num = len(s_feats)

        loss_distill = 0
        for i in range(s_feats_num):
            s_feats[i] = self.Connectors[i](s_feats[i])
            loss_distill += distillation_loss(s_feats[i], t_feats[i].detach(), getattr(self, 'margin%d' % (i + 1))) \
                / 2 ** (t_feats_num - i - 1)

        if is_loss:
            return s_out, loss_distill
        return s_out


class OHTrainer(BaseTrainer):
    def __init__(self, s_net, config):
        super(OHTrainer, self).__init__(s_net, config)
        # the student net is the base net
        self.s_net = self.net.s_net
        self.d_net = self.net

    def calculate_loss(self, data, target):

        output, loss_distill = self.d_net(data, is_loss=True)
        loss_CE = self.loss_fun(output, target)

        loss = loss_CE + loss_distill.sum() / self.batch_size / 1000

        loss.backward()
        self.optimizer.step()
        return output, loss


def run_oh_distillation(s_net, t_net, **params):

    # Student training
    # Define loss and the optimizer
    print("---------- Training OKD Student -------")
    s_net = Distiller(t_net, s_net).to(params["device"])
    s_trainer = OHTrainer(s_net, config=params)
    best_s_acc = s_trainer.train()

    return best_s_acc
