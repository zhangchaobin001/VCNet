import os
import torch
import yaml
import torch.nn as nn
import numbers
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn import init
import math

def L2Normalization(ff, dim = 1):
     # ff is B*N
     fnorm = torch.norm(ff, p=2, dim=dim, keepdim=True) + 1e-5
     ff = ff.div(fnorm.expand_as(ff))
     return ff

def myphi(x,m):
    x = x * m
    return 1-x**2/math.factorial(2)+x**4/math.factorial(4)-x**6/math.factorial(6) + \
            x**8/math.factorial(8) - x**9/math.factorial(9)

# I largely modified the AngleLinear Loss
class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m = 4, phiflag=True):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))
        init.normal_(self.weight.data, std=0.001)
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]

    def forward(self, input):
        x = input   # size=(B,F)    F is feature len
        w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2,1,1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5) # size=B
        wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum

        cos_theta = x.mm(ww) # size=(B,Classnum)
        cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
        cos_theta = cos_theta.clamp(-1,1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = Variable(cos_theta.data.acos())
            k = (self.m*theta/3.14159265).floor()
            n_one = k*0.0 - 1
            phi_theta = (n_one**k) * cos_m_theta - 2*k
        else:
            theta = cos_theta.acos()
            phi_theta = myphi(theta,self.m)
            phi_theta = phi_theta.clamp(-1*self.m,1)

        cos_theta = cos_theta * xlen.view(-1,1)
        phi_theta = phi_theta * xlen.view(-1,1)
        output = (cos_theta,phi_theta)
        return output # size=(B,Classnum,2)

#https://github.com/auroua/InsightFace_TF/blob/master/losses/face_losses.py#L80
class ArcLinear(nn.Module):
    def __init__(self, in_features, out_features, s=64.0):
        super(ArcLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features,out_features))
        init.normal_(self.weight.data, std=0.001)
        self.loss_s = s

    def forward(self, input):
        embedding = input
        nembedding = L2Normalization(embedding, dim=1)*self.loss_s
        _weight = L2Normalization(self.weight, dim=0)
        fc7 = nembedding.mm(_weight)
        output = (fc7, _weight, nembedding)
        return output

class ArcLoss(nn.Module):
    def __init__(self, m1=1.0, m2=0.5, m3 =0.0, s = 64.0):
        super(ArcLoss, self).__init__()
        self.loss_m1 = m1
        self.loss_m2 = m2
        self.loss_m3 = m3
        self.loss_s = s

    def forward(self, input, target):
        fc7, _weight, nembedding = input

        index = fc7.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,target.data.view(-1,1),1)
        index = index.byte()
        index = Variable(index)

        zy = fc7[index]
        cos_t = zy/self.loss_s
        t = torch.acos(cos_t)
        t = t*self.loss_m1 + self.loss_m2
        body = torch.cos(t) - self.loss_m3

        new_zy = body*self.loss_s
        diff = new_zy - zy
        fc7[index] += diff
        loss = F.cross_entropy(fc7, target)
        return loss

class AngleLoss(nn.Module):
    def __init__(self, gamma=0):
        super(AngleLoss, self).__init__()
        self.gamma   = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        cos_theta,phi_theta = input
        target = target.view(-1,1) #size=(B,1)

        index = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,target.data.view(-1,1),1)
        index = index.byte()
        index = Variable(index)

        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.1*self.it ))
        output = cos_theta * 1.0 #size=(B,Classnum)
        output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

        logpt = F.log_softmax(output, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()

        return loss

def euclidean_dist(x, y, squared=True):
    """
    Compute (Squared) Euclidean distance between two tensors.

    Args:
        x: input tensor with size N x D.
        y: input tensor with size M x D.

        return: distance matrix with size N x M.
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception('Invalid input shape.')

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    dist = torch.pow(x - y, 2).sum(2)

    if squared:
        return dist
    else:
        return torch.sqrt(dist+1e-12)


def cosine_dist(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception('Invalid input shape.')

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    # dist = -torch.mul(x, y).sum(2) / torch.clamp(torch.mul(torch.norm(x, p=2, dim=2), torch.norm(y, p=2, dim=2)), min=1e-6)
    dist = -torch.mul(x, y).sum(2)

    return dist

class DMMLLoss(nn.Module):
    """
    DMML loss with center support distance and hard mining distance.

    Args:
        num_support: the number of support samples per class.
        distance_mode: 'center_support' or 'hard_mining'.
    """
    def __init__(self, num_support, distance_mode='hard_mining', margin=0.4, gid=None):
        super().__init__()

        if not distance_mode in ['center_support', 'hard_mining']:
            raise Exception('Invalid distance mode for DMML loss.')
        if not isinstance(margin, numbers.Real):
            raise Exception('Invalid margin parameter for DMML loss.')

        self.num_support = num_support
        self.distance_mode = distance_mode
        self.margin = margin
        self.gid = gid

    def forward(self, feature, label):
        feature = feature.cpu()
        label = label.cpu()
        classes = torch.unique(label)  # torch.unique() is cpu-only in pytorch 0.4
        if self.gid is not None:
            feature, label, classes = feature.cuda(self.gid), label.cuda(self.gid), classes.cuda(self.gid)
        num_classes = len(classes)
        num_query = label.eq(classes[0]).sum() - self.num_support

        support_inds_list = list(map(
            lambda c: label.eq(c).nonzero()[:self.num_support].squeeze(1), classes))
        query_inds = torch.stack(list(map(
            lambda c: label.eq(c).nonzero()[self.num_support:], classes))).view(-1)
        query_samples = feature[query_inds]

        if self.distance_mode == 'center_support':
            center_points = torch.stack([torch.mean(feature[support_inds], dim=0)
                for support_inds in support_inds_list])
            dists = euclidean_dist(query_samples, center_points)
        elif self.distance_mode == 'hard_mining':
            dists = []
            max_self_dists = []
            for i, support_inds in enumerate(support_inds_list):
                # dist_all = euclidean_dist(query_samples, feature[support_inds])
                dist_all = cosine_dist(query_samples, feature[support_inds])
                max_dist, _ = torch.max(dist_all[i*num_query:(i+1)*num_query], dim=1)
                min_dist, _ = torch.min(dist_all, dim=1)
                dists.append(min_dist)
                max_self_dists.append(max_dist)
            dists = torch.stack(dists).t()
            # dists = torch.clamp(torch.stack(dists).t() - self.margin, min=0.0)
            for i in range(num_classes):
                dists[i*num_query:(i+1)*num_query, i] = max_self_dists[i]

        log_prob = F.log_softmax(-dists, dim=1).view(num_classes, num_query, -1)

        target_inds = torch.arange(0, num_classes)
        if self.gid is not None:
            target_inds = target_inds.cuda(self.gid)
        target_inds = target_inds.view(num_classes, 1, 1).expand(num_classes, num_query, 1).long()

        dmml_loss = -log_prob.gather(2, target_inds).squeeze().view(-1).mean()

        batch_size = feature.size(0)
        l2_loss = torch.sum(feature ** 2) / batch_size
        dmml_loss += 0.002 * 0.25 * l2_loss

        return dmml_loss


