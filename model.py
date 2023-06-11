import argparse
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from losses import AngleLinear, ArcLinear
from efficientnet_pytorch import EfficientNet
import pretrainedmodels
from torch.nn import functional as F
from abc import abstractmethod
from functools import reduce
import pdb
from attention import SequentialPolarizedSelfAttention,MMTM,Attention,MQF,ExternalAttention, MMTMF,MMTMdemo

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

def fix_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return [x, f]
        else:
            x = self.classifier(x)
            return x
#最后一阶段的感知聚合模块
class three_aware(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, M=3, r=16, L=32):
        '''
        :param in_channels:  输入通道维度
        :param out_channels: 输出通道维度   原论文中 输入输出通道维度相同
        :param stride:  步长，默认为1
        :param M:  分支数
        :param r: 特征Z的长度，计算其维度d 时所需的比率（论文中 特征S->Z 是降维，故需要规定 降维的下界）
        :param L:  论文中规定特征Z的下界，默认为32
        采用分组卷积： groups = 32,所以输入channel的数值必须是group的整数倍
        '''
        super().__init__()
        d = max(in_channels // r, L)  # 计算从向量C降维到 向量Z 的长度d
        self.M = M
        self.out_channels = out_channels
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)  # 自适应pool到指定维度    这里指定为1，实现 GAP
        self.fc1 = nn.Sequential(nn.Conv2d(out_channels, d, 1, bias=False),
                                 nn.BatchNorm2d(d),
                                 nn.ReLU(inplace=True))  # 降维
        self.fc2 = nn.Conv2d(d, out_channels * M, 1, 1, bias=False)  # 升维
        self.softmax = nn.Softmax(dim=1)  # 指定dim=1  使得两个全连接层对应位置进行softmax,保证 对应位置a+b+..=1
        self.classifier = nn.Sequential(
            nn.Linear(512, 124, bias = True)
        )
    def forward(self, qian, hou, ce):
        batch_size, chan = qian.size()
        qian = qian.view(batch_size, chan, 1, 1)
        hou = hou.view(batch_size, chan, 1, 1)
        ce = ce.view(batch_size, chan, 1, 1)
        output=[]
        output.append(qian)
        output.append(hou)
        output.append(ce)
        U = reduce(lambda x, y: x + y, output)
        ss = self.global_pool(U)  # [batch_size,channel,1,1]
        z = self.fc1(ss)  # S->Z降维   # [batch_size,d,1,1]
        a_b = self.fc2(z)  # Z->a，b 升维  论文使用conv 1x1表示全连接。结果中前一半通道值为a,后一半为b   [batch_size,out_channels*M,1,1]
        a_b = a_b.reshape(batch_size, self.M, self.out_channels,
                          -1)  # 调整形状，变为 两个全连接层的值[batch_size,M,out_channels,1]
        a_b = self.softmax(a_b)  # 使得两个全连接层对应位置进行softmax [batch_size,M,out_channels,1]
        # the part of selection
        a_b = list(a_b.chunk(self.M,
                             dim=1))  # split to a and b   chunk为pytorch方法，将tensor按照指定维度切分成 几个tensor块 [[batch_size,1,out_channels,1],[batch_size,1,out_channels,1]
        a_b = list(map(lambda x: x.reshape(batch_size, self.out_channels, 1, 1),
                       a_b))  # 将所有分块  调整形状，即扩展两维  [[batch_size,out_channels,1,1],[batch_size,out_channels,1,1]
        V = list(map(lambda x, y: x * y, output,
                     a_b))  # 权重与对应  不同卷积核输出的U 逐元素相乘[batch_size,out_channels,H,W] * [batch_size,out_channels,1,1] = [batch_size,out_channels,H,W]
        V = reduce(lambda x, y: x + y,
                   V)  # 两个加权后的特征 逐元素相加  [batch_size,out_channels,H,W] + [batch_size,out_channels,H,W] = [batch_size,out_channels,H,W]
        x = V.view(V.size(0), V.size(1))
        return x, V  # [batch_size,out_channels,H,W]
#也是感知聚合模块，实验的时候为了验证不同的模型对实验的效果
class three_select(nn.Module):
    def __init__(self, class_num=124):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(512*3, 128, bias=False),
            nn.ReLU(),
            nn.Linear(128, 512 * 3, bias=False),
            nn.Sigmoid()
        )
        self.fc_f = nn.Sequential(
            nn.Linear(512*3, 512, bias=True),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, class_num, bias=True),
        )

    def forward(self, bb, qian, hou, ce):
        z = torch.cat((qian, hou, ce), 1)
        f = self.fc(z)
        f = torch.mul(z, f)
        f = self.fc_f(f)
        f = (bb + f) / 2
        x = self.classifier(f)
        return x, f
#感知车辆前面的特征，与论文中的模块结构不同，后面可自行调整
class qian_aware(nn.Module):
    def __init__(self, channel=512):
        super().__init__()
        self.aware = nn.Sequential(
            nn.Linear(512, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512, bias=False),
            nn.BatchNorm1d(512)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 124, bias=True),
        )
    def forward(self, qian):
        f = self.aware(qian)
        #f = (qian + f) / 2
        x = self.classifier(f)
        return x, f
#感知车辆后面的特征，与论文中的模块结构不同，后面可自行调整
class hou_aware(nn.Module):
    def __init__(self, channel=512):
        super().__init__()
        self.aware = nn.Sequential(
            nn.Linear(512, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512, bias=False),
            nn.BatchNorm1d(512)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 124, bias=True),
        )
    def forward(self, hou):
        f = self.aware(hou)
        #f = (hou + f) / 2
        x = self.classifier(f)
        return x, f
#感知车辆侧面的特征，与论文中的模块结构不同，后面可自行调整
class ce_aware(nn.Module):
    def __init__(self, channel=512):
        super().__init__()
        self.aware = nn.Sequential(
            nn.Linear(512, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512, bias=False),
            nn.BatchNorm1d(512)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 124, bias=True),
        )
    def forward(self, ce):
        f = self.aware(ce)
        #f = (ce + f) / 2
        x = self.classifier(f)
        return x, f

#backbone网络
class ft_net(nn.Module):

    def __init__(self, class_num=751, droprate=0.5, stride=2, circle=False, ibn=False):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        if ibn==True:
            model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.circle = circle
        self.classifier = ClassBlock(2048, class_num, droprate, return_f = circle)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

#到此为止后面的模型可以不用看了，是我之前做实验的模型
# Define the ResNet50-based Model
class ft_net_ours(nn.Module):

    def __init__(self, class_num=751, droprate=0.5, stride=2, circle=False, ibn=False):
        super().__init__()
        model_ft = models.resnet50(pretrained=True)
        if ibn == True:
            model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1, 1)
            model_ft.layer4[0].conv2.stride = (1, 1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.circle = circle
        self.classifier = ClassBlock(2048, class_num, droprate, return_f=circle)
        self.proj0 = nn.ConvTranspose2d(2, 3, 16, 16, bias=False)
        self.proj1 = nn.ConvTranspose2d(2, 64, 4, 4, bias=False)
        self.proj2 = nn.ConvTranspose2d(2, 256, 4, 4, bias=False)
        self.proj3 = nn.ConvTranspose2d(2, 512, 2, 2, bias=False)
        self.proj4 = nn.ConvTranspose2d(2, 1024, 1, 1, bias=False)

    def forward(self, x, vf):
        b, c = vf.size()
        vf = vf.view(b, 2, 16, 16)
        vf0 = self.proj0(vf)  # b, 3, 256, 256
        x = torch.add(x, vf0)

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x) # b, 64, 64, 64

        vf1 = self.proj1(vf)
        x = torch.add(x, vf1)
        x = self.model.layer1(x)  # b, 256, 64, 64

        vf2 = self.proj2(vf)
        x = torch.add(x, vf2)

        x = self.model.layer2(x)  # b, 512, 32, 32
        vf3 = self.proj3(vf)
        x = torch.add(x, vf3)

        x = self.model.layer3(x)  # b, 1024, 16, 16
        vf4 = self.proj4(vf)
        x = torch.add(x, vf4)

        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input):
        raise NotImplementedError

    def decode(self, input):
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs):
        raise RuntimeWarning()

    def generate(self, x, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs) :
        pass

    @abstractmethod
    def loss_function(self, *inputs, **kwargs):
        pass

class WAE_MMD(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims = None,
                 reg_weight: int = 100,
                 kernel_type: str = 'imq',
                 latent_var: float = 2.,
                 **kwargs) -> None:
        super(WAE_MMD, self).__init__()

        self.latent_dim = latent_dim
        self.reg_weight = reg_weight
        self.kernel_type = kernel_type
        self.z_var = latent_var

        modules = []
        if hidden_dims is None:
            hidden_dims = [8, 16, 32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_z = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        z = self.fc_z(result)
        return z

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, input, **kwargs):
        z = self.encode(input)
        return  [self.decode(z), input, z]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        z = args[2]

        batch_size = input.size(0)
        bias_corr = batch_size *  (batch_size - 1)
        reg_weight = self.reg_weight / bias_corr

        recons_loss =F.mse_loss(recons, input)

        mmd_loss = self.compute_mmd(z, reg_weight)

        loss = recons_loss + mmd_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'MMD': mmd_loss}

    def compute_kernel(self,
                       x1,
                       x2):
        # Convert the tensors into row and column vectors
        D = x1.size(1)
        N = x1.size(0)

        x1 = x1.unsqueeze(-2) # Make it into a column tensor
        x2 = x2.unsqueeze(-3) # Make it into a row tensor

        """
        Usually the below lines are not required, especially in our case,
        but this is useful when x1 and x2 have different sizes
        along the 0th dimension.
        """
        x1 = x1.expand(N, N, D)
        x2 = x2.expand(N, N, D)

        if self.kernel_type == 'rbf':
            result = self.compute_rbf(x1, x2)
        elif self.kernel_type == 'imq':
            result = self.compute_inv_mult_quad(x1, x2)
        else:
            raise ValueError('Undefined kernel type.')

        return result


    def compute_rbf(self,
                    x1,
                    x2,
                    eps: float = 1e-7):
        """
        Computes the RBF Kernel between x1 and x2.
        :param x1: (Tensor)
        :param x2: (Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        sigma = 2. * z_dim * self.z_var

        result = torch.exp(-((x1 - x2).pow(2).mean(-1) / sigma))
        return result

    def compute_inv_mult_quad(self,
                               x1,
                               x2,
                               eps: float = 1e-7):
        """
        Computes the Inverse Multi-Quadratics Kernel between x1 and x2,
        given by

                k(x_1, x_2) = \sum \frac{C}{C + \|x_1 - x_2 \|^2}
        :param x1: (Tensor)
        :param x2: (Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        C = 2 * z_dim * self.z_var
        kernel = C / (eps + C + (x1 - x2).pow(2).sum(dim = -1))

        # Exclude diagonal elements
        result = kernel.sum() - kernel.diag().sum()

        return result

    def compute_mmd(self, z, reg_weight: float):
        # Sample from prior (Gaussian) distribution
        prior_z = torch.randn_like(z)

        prior_z__kernel = self.compute_kernel(prior_z, prior_z)
        z__kernel = self.compute_kernel(z, z)
        priorz_z__kernel = self.compute_kernel(prior_z, z)

        mmd = reg_weight * prior_z__kernel.mean() + \
              reg_weight * z__kernel.mean() - \
              2 * reg_weight * priorz_z__kernel.mean()
        return mmd

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

class LogCoshVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims=None,
                 alpha: float = 100.,
                 beta: float = 10.,
                 **kwargs) -> None:
        super(LogCoshVAE, self).__init__()

        self.latent_dim = latent_dim
        self.alpha = alpha
        self.beta = beta

        modules = []
        if hidden_dims is None:
            hidden_dims = [8, 16, 32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        t = recons - input
        # recons_loss = F.mse_loss(recons, input)
        # cosh = torch.cosh(self.alpha * t)
        # recons_loss = (1./self.alpha * torch.log(cosh)).mean()

        recons_loss = self.alpha * t + \
                      torch.log(1. + torch.exp(- 2 * self.alpha * t)) - \
                      torch.log(torch.tensor(2.0))
        # print(self.alpha* t.max(), self.alpha*t.min())
        recons_loss = (1. / self.alpha) * recons_loss.mean()

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss*10 + self.beta * kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

class VanillaVAE(BaseVAE):


    def __init__(self,
                 in_channels,
                 latent_dim,
                 hidden_dims=None,
                 **kwargs):
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [8, 16, 32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)
        #self.classifier = nn.Linear(latent_dim, 75)
        #self.avg = nn.AdaptiveAvgPool2d((1, 1))
        #self.classifier.apply(weights_init_classifier)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

class Autoencoder(nn.Module):
    """AutoEncoder module that projects features to latent space."""

    def __init__(self,
                 encoder_dim=[512,1024,1024,1024,128],
                 activation='relu',
                 batchnorm=True):
        """Constructor.

        Args:
          encoder_dim: Should be a list of ints, hidden sizes of
            encoder network, the last element is the size of the latent representation.
          activation: Including "sigmoid", "tanh", "relu", "leakyrelu". We recommend to
            simply choose relu.
          batchnorm: if provided should be a bool type. It provided whether to use the
            batchnorm in autoencoders.
        """
        super(Autoencoder, self).__init__()

        self._dim = len(encoder_dim) - 1 #encoder_dim = 512,1024,1024,1024,128  _dim=4
        self._activation = activation
        self._batchnorm = batchnorm

        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(
                nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
            if i < self._dim - 1:
                if self._batchnorm:
                    encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1]))
                if self._activation == 'sigmoid':
                    encoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    encoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    encoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        encoder_layers.append(nn.Softmax(dim=1))
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_dim = [i for i in reversed(encoder_dim)]  #decoder_dim = 128,1024,1024,1024,512
        decoder_layers = []
        for i in range(self._dim):
            decoder_layers.append(
                nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            if self._batchnorm:
                decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
            if self._activation == 'sigmoid':
                decoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                decoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                decoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._decoder = nn.Sequential(*decoder_layers)

    def encoder(self, x):
        """Encode sample features.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [n_nodes, latent_dim] float tensor, representation Z.
        """
        latent = self._encoder(x)
        return latent

    def decoder(self, latent):
        """Decode sample features.

            Args:
              latent: [num, latent_dim] float tensor, representation Z.

            Returns:
              x_hat: [n_nodes, feat_dim] float tensor, reconstruction x.
        """
        x_hat = self._decoder(latent)
        return x_hat

    def forward(self, x):
        """Pass through autoencoder.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [num, latent_dim] float tensor, representation Z.
              x_hat:  [num, feat_dim] float tensor, reconstruction x.
        """
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        return x_hat, latent


class Prediction(nn.Module):
    """Dual prediction module that projects features from corresponding latent space."""

    def __init__(self,
                 prediction_dim=[128,128,256,128],
                 activation='relu',
                 batchnorm=True):
        """Constructor.

        Args:
          prediction_dim: Should be a list of ints, hidden sizes of
            prediction network, the last element is the size of the latent representation of autoencoder.
          activation: Including "sigmoid", "tanh", "relu", "leakyrelu". We recommend to
            simply choose relu.
          batchnorm: if provided should be a bool type. It provided whether to use the
            batchnorm in autoencoders.
        """
        super(Prediction, self).__init__()

        self._depth = len(prediction_dim) - 1
        self._activation = activation
        self._prediction_dim = prediction_dim

        encoder_layers = []
        for i in range(self._depth):
            encoder_layers.append(
                nn.Linear(self._prediction_dim[i], self._prediction_dim[i + 1]))
            if batchnorm:
                encoder_layers.append(nn.BatchNorm1d(self._prediction_dim[i + 1]))
            if self._activation == 'sigmoid':
                encoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                encoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                encoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(self._depth, 0, -1):
            decoder_layers.append(
                nn.Linear(self._prediction_dim[i], self._prediction_dim[i - 1]))
            if i > 1:
                if batchnorm:
                    decoder_layers.append(nn.BatchNorm1d(self._prediction_dim[i - 1]))

                if self._activation == 'sigmoid':
                    decoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    decoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    decoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        decoder_layers.append(nn.Softmax(dim=1))
        self._decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        """Data recovery by prediction.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [num, latent_dim] float tensor.
              output:  [num, feat_dim] float tensor, recovered data.
        """
        latent = self._encoder(x)
        output = self._decoder(latent)
        return output, latent

#添加融合的模块
class ft_net_beifen(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg', circle=False):
        super(ft_net_ours, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)

        self.circle = circle
        self.pool = pool
        self.proj1 = nn.ConvTranspose2d(8, 64, 8, 8, bias=True)
        self.proj2 = nn.ConvTranspose2d(8, 256, 8, 8, bias=True)
        self.proj3 = nn.ConvTranspose2d(8, 512, 4, 4, bias=True)
        self.proj4 = nn.ConvTranspose2d(8, 1024, 2, 2, bias=True)
        if pool =='avg+max':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
            self.model = model_ft
            self.classifier = ClassBlock(4096, class_num, droprate, return_f = circle)
        elif pool=='avg':
            model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.model = model_ft
            self.classifier = ClassBlock(2048, class_num, droprate, return_f= circle)

        self.flag = False
        if init_model!=None:
            self.flag = True
            self.model = init_model.model
            self.pool = init_model.pool
            self.classifier.add_block = init_model.classifier.add_block
            self.new_dropout = nn.Sequential(nn.Dropout(p = droprate))
        # hlf
        #self.attention = SequentialPolarizedSelfAttention(channel=2048)
        #self.convr = nn.Conv2d(3072, 2048, kernel_size=(1,1))
        #self.mmtm1 = MMTM(256, 256, 256, 2)
        #self.mmtm2 = MMTM(512, 512, 512, 4)
        #self.mmtm3 = MMTM(1024,1024,1024, 4)
        #self.mmtm4 = MMTM(2048,2048,2048, 8)

    def forward(self, x, vf):
    #def forward(self, x):

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        #将视角最后一层的特征加到第一层
        b, c = vf.size()
        vf = vf.view(b, 8, 8, 8)
        vf1 = self.proj1(vf) #b, 64, 64, 64
        x = x + vf1

        x = self.model.layer1(x) #b, 256, 64, 64
        vf2 = self.proj2(vf)
        x = x + vf2

        x = self.model.layer2(x) #b, 512, 32, 32
        vf3 = self.proj3(vf)
        x = x + vf3

        x = self.model.layer3(x) #b, 1024, 16, 16
        vf4 = self.proj4(vf)
        x = x + vf4

        x = self.model.layer4(x) #b, 2048, 8, 8


        # x = list(torch.split(x, 3, dim=0))  # [3,c,h,w]
        # for index, value in enumerate(x):
        #     x[index] = self.mmtm4(torch.unsqueeze(value[0], dim=0), torch.unsqueeze(value[1], dim=0),torch.unsqueeze(value[2], dim=0))
        # x = torch.cat(x, dim=0)


        if self.pool == 'avg+max':
            x1 = self.model.avgpool2(x)
            x2 = self.model.maxpool2(x)
            x = torch.cat((x1,x2), dim = 1)
            x = x.view(x.size(0), x.size(1))
        elif self.pool == 'avg':
            x = self.model.avgpool(x)
            x = x.view(x.size(0), x.size(1))
        if self.flag:
            x = self.classifier.add_block(x)
            x = self.new_dropout(x)
            x = self.classifier.classifier(x)
        else:
            x = self.classifier(x)
        return x
# Define the ResNet50  Model with angle loss
# The code is borrowed from https://github.com/clcarwin/sphereface_pytorch
class ft_net_angle(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2):
        super(ft_net_angle, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num, droprate)
        #self.classifier.classifier=nn.Sequential()
        self.classifier.classifier = AngleLinear(512, class_num)


    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        #x = self.fc(x)
        return x

class ft_net_arc(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2):
        super(ft_net_arc, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num, droprate)
        #self.classifier.classifier=nn.Sequential()
        self.classifier.classifier = ArcLinear(512, class_num)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        #x = self.fc(x)
        return x

# Define the DenseNet121-based Model
class ft_net_dense(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg', circle=False):
        super().__init__()
        model_ft = models.densenet121(pretrained=True)
        if stride == 1:
            model_ft.features.transition3.pool.stride = 1
        model_ft.fc = nn.Sequential()
        self.pool = pool
        self.circle = circle
        if pool =='avg+max':
            model_ft.features.avgpool = nn.Sequential()
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
            self.model = model_ft
            self.classifier = ClassBlock(2048, class_num, droprate, return_f = circle)
        elif pool=='avg':
            model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.model = model_ft
            self.classifier = ClassBlock(1024, class_num, droprate, return_f = circle)

        self.flag = False
        if init_model!=None:
            self.flag = True
            self.model = init_model.model
            self.pool = init_model.pool
            self.classifier.add_block = init_model.classifier.add_block
            self.new_dropout = nn.Sequential(nn.Dropout(p = droprate))

    def forward(self, x):
        if self.pool == 'avg':
            x = self.model.features(x)
        elif self.pool == 'avg+max':
            x = self.model.features(x)
            x1 = self.model.avgpool2(x)
            x2 = self.model.maxpool2(x)
            x = torch.cat((x1,x2), dim = 1)
        x = x.view(x.size(0), x.size(1))
        if self.flag:
            x = self.classifier.add_block(x)
            x = self.new_dropout(x)
            x = self.classifier.classifier(x)
        else:
            x = self.classifier(x)
        return x

class ft_net_EF4(nn.Module):
    def __init__(self, class_num, droprate=0.2):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b4')
        self.model._fc = nn.Sequential()
        self.classifier = ClassBlock(1792, class_num, droprate)

    def forward(self, x):
        # Convolution layers
        x = self.model.extract_features(x)
        # Pooling and final linear layer
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        x = self.classifier(x)

        return x

class ft_net_EF5(nn.Module):
    def __init__(self, class_num, droprate=0.2):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b5')
        self.model._fc = nn.Sequential()
        self.classifier = ClassBlock(2048, class_num, droprate)

    def forward(self, x):
        # Convolution layers
        x = self.model.extract_features(x)
        # Pooling and final linear layer
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        x = self.classifier(x)

        return x

class ft_net_EF6(nn.Module):
    def __init__(self, class_num, droprate=0.2):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b6')
        self.model._fc = nn.Sequential()
        self.classifier = ClassBlock(2304, class_num, droprate)

    def forward(self, x):
        # Convolution layers
        x = self.model.extract_features(x)
        # Pooling and final linear layer
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        x = self.classifier(x)

        return x
# Define the NAS-based Model
class ft_net_NAS(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2):
        super().__init__()
        model_name = 'nasnetalarge' # could be fbresnet152 or inceptionresnetv2
        model_ft = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        #if stride == 1:
        #    model_ft.layer4[0].downsample[0].stride = (1,1)
        #    model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.dropout = nn.Sequential()
        model_ft.last_linear = nn.Sequential()
        # relu -> inplace
        model_ft.cell_17.apply(fix_relu)
        self.model = model_ft
        # For DenseNet, the feature dim is 4032
        self.classifier = ClassBlock(4032, class_num, droprate)

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avg_pool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

# Define the SE-based Model
class ft_net_SE(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, pool='avg', init_model=None, circle=False):
        super().__init__()
        model_name = 'se_resnext101_32x4d' # could be fbresnet152 or inceptionresnetv2
        model_ft = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        if stride == 1:
            model_ft.layer4[0].conv2.stride = (1,1)
            model_ft.layer4[0].downsample[0].stride = (1,1)
        if pool == 'avg':
            model_ft.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        elif pool == 'max':
            model_ft.avg_pool = nn.AdaptiveMaxPool2d((1,1))
        elif pool == 'avg+max':
            model_ft.avg_pool2 = nn.AdaptiveAvgPool2d((1,1))
            model_ft.max_pool2 = nn.AdaptiveMaxPool2d((1,1))
        else:
           print('UNKNOW POOLING!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #model_ft.dropout = nn.Sequential()
        model_ft.last_linear = nn.Sequential()
        self.circle = circle
        self.model = model_ft
        self.pool  = pool
        # For DenseNet, the feature dim is 2048
        if pool == 'avg+max':
            self.classifier = ClassBlock(4096, class_num, droprate, return_f = circle)
        else:
            self.classifier = ClassBlock(2048, class_num, droprate, return_f = circle)
        self.flag = False
        if init_model!=None:
            self.flag = True
            self.model = init_model.model
            self.classifier.add_block = init_model.classifier.add_block
            self.new_dropout = nn.Sequential(nn.Dropout(p = droprate))

    def forward(self, x):
        x = self.model.features(x)
        if self.pool == 'avg+max':
            x1 = self.model.avg_pool2(x)
            x2 = self.model.max_pool2(x)
            x = torch.cat((x1,x2), dim = 1)
        else:
            x = self.model.avg_pool(x)
        x = x.view(x.size(0), x.size(1))
        # Convolution layers
        # Pooling and final linear layer
        if self.flag:
            x = self.classifier.add_block(x)
            x = self.new_dropout(x)
            x = self.classifier.classifier(x)
        else:
            x = self.classifier(x)
        return x

# Define the SE-based Model
class ft_net_DSE(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, pool='avg'):
        super().__init__()
        model_name = 'senet154' # could be fbresnet152 or inceptionresnetv2
        model_ft = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        if stride == 1:
            model_ft.layer4[0].conv2.stride = (1,1)
            model_ft.layer4[0].downsample[0].stride = (1,1)
        if pool == 'avg':
            model_ft.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        elif pool == 'max':
            model_ft.avg_pool = nn.AdaptiveMaxPool2d((1,1))
        else:
           print('UNKNOW POOLING!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #model_ft.dropout = nn.Sequential()
        model_ft.dropout = nn.Sequential()
        model_ft.last_linear = nn.Sequential()
        self.model = model_ft
        # For DenseNet, the feature dim is 2048
        self.classifier = ClassBlock(2048, class_num, droprate)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

# Define the inceptionresnetv2-based Model
class ft_net_IR(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2):
        super().__init__()
        model_name = 'inceptionresnetv2' # could be fbresnet152 or inceptionresnetv2
        model_ft = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        if stride == 1:
            model_ft.mixed_7a.branch0[1].conv.stride = (1,1)
            model_ft.mixed_7a.branch1[1].conv.stride = (1,1)
            model_ft.mixed_7a.branch2[2].conv.stride = (1,1)
            model_ft.mixed_7a.branch3.stride = 1
        model_ft.avgpool_1a = nn.AdaptiveAvgPool2d((1,1))
        #model_ft.dropout = nn.Sequential()
        model_ft.last_linear = nn.Sequential()
        self.model = model_ft
        # For DenseNet, the feature dim is 2048
        self.classifier = ClassBlock(1536, class_num, droprate)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

# Define the ResNet50-based Model (Middle-Concat)
# In the spirit of "The Devil is in the Middle: Exploiting Mid-level Representations for Cross-Domain Instance Matching." Yu, Qian, et al. arXiv:1711.08106 (2017).
class ft_net_middle(nn.Module):

    def __init__(self, class_num, droprate=0.5):
        super(ft_net_middle, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048+1024, class_num, droprate)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        # x0  n*1024*1*1
        x0 = self.model.avgpool(x)
        x = self.model.layer4(x)
        # x1  n*2048*1*1
        x1 = self.model.avgpool(x)
        x = torch.cat((x0,x1),1)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

# Part Model proposed in Yifan Sun etal. (2018)
class PCB(nn.Module):
    def __init__(self, class_num ):
        super(PCB, self).__init__()

        self.part = 6 # We cut the pool5 to 6 parts
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)
        # define 6 classifiers
        for i in range(self.part):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(2048, class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        part = {}
        predict = {}
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            part[i] = torch.squeeze(x[:,:,i])
            name = 'classifier'+str(i)
            c = getattr(self,name)
            predict[i] = c(part[i])

        y = []
        for i in range(self.part):
            y.append(predict[i])
        return y

class PCB_test(nn.Module):
    def __init__(self,model):
        super(PCB_test,self).__init__()
        self.part = 6
        self.model = model.model
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        y = x.view(x.size(0),x.size(1),x.size(2))
        return y

# Center Part Model proposed in Yifan Sun etal. (2018)
class CPB(nn.Module):
    def __init__(self, class_num ):
        super(CPB, self).__init__()

        self.part = 4 # We cut the pool5 to 4 parts
        #model_ft = models.resnet50(pretrained=True)
        #self.model = EfficientNet.from_pretrained('efficientnet-b5')
        #self.model._fc = nn.Sequential()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_name = 'se_resnext101_32x4d' # could be fbresnet152 or inceptionresnetv2
        model_ft = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.layer4[0].downsample[0].stride = (1,1)
        self.model = model_ft
       #self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        #self.model.layer4[0].downsample[0].stride = (1,1)
        #self.model.layer4[0].conv2.stride = (1,1)
        # define 6 classifiers
        for i in range(self.part):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(2048, class_num, droprate=0.2, relu=False, bnorm=True, num_bottleneck=512))

    def forward(self, x):
        x =  self.model.features(x)
        #x = self.dropout(x)
        #print(x.shape)
        part = {}
        predict = {}
        d = 2+2+2
        for i in range(self.part):
            N,C,H,W = x.shape
            p = 2 #max(2-i,1)
            if i==0:
                part_input = x[:,:,d:W-d,d:H-d]
                part[i] = torch.squeeze(self.avgpool(part_input))
                last_input = torch.nn.functional.pad(part_input, (p,p,p,p), mode='constant', value=0)
                #print(part_input.shape)
            else:
                part_input = x[:,:,d:W-d,d:H-d] - last_input
                #print(part_input.shape)
                part[i] = torch.squeeze(self.avgpool(part_input))
                last_input = torch.nn.functional.pad(part_input, (p,p,p,p), mode='constant', value=0)
            name = 'classifier'+str(i)
            c = getattr(self,name)
            predict[i] = c(part[i])
            d = d - p

        y = []
        for i in range(self.part):
            y.append(predict[i])
        return y

'''
# debug model structure
# Run this code with:
python model.py
'''
if __name__ == '__main__':
# Here I left a simple forward function.
# Test the model, before you train it. 
    net = ft_net_ours(class_num=151, droprate=0.5, stride=2, circle=False, ibn=False)
    #net = ft_net_SE(751)
    print(net)
    qian = Variable(torch.randn(32, 512))
    ce = Variable(torch.randn(32, 512))
    hou = Variable(torch.randn(32, 512))
    output = net(qian,ce,hou)
    print('net output size:', output.shape)
    #print(output.shape)
