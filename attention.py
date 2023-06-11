import numpy as np
import torch
from torch import nn
from torch.nn import init
import pdb

class SequentialPolarizedSelfAttention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.ch_wq=nn.Conv2d(channel,1,kernel_size=(1,1))
        self.softmax_channel=nn.Softmax(1)
        self.softmax_spatial=nn.Softmax(-1)
        self.ch_wz=nn.Conv2d(channel//2,channel,kernel_size=(1,1))
        self.ln=nn.LayerNorm(channel)
        self.sigmoid=nn.Sigmoid()
        self.sp_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.sp_wq=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.agp=nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        b, c, h, w = x.size()

        #Channel-only Self-Attention
        channel_wv=self.ch_wv(x) #bs,c//2,h,w
        channel_wq=self.ch_wq(x) #bs,1,h,w
        channel_wv=channel_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        channel_wq=channel_wq.reshape(b,-1,1) #bs,h*w,1
        channel_wq=self.softmax_channel(channel_wq)
        channel_wz=torch.matmul(channel_wv,channel_wq).unsqueeze(-1) #bs,c//2,1,1
        channel_weight=self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1
        channel_out=channel_weight*x

        #Spatial-only Self-Attention
        spatial_wv=self.sp_wv(channel_out) #bs,c//2,h,w
        spatial_wq=self.sp_wq(channel_out) #bs,c//2,h,w
        spatial_wq=self.agp(spatial_wq) #bs,c//2,1,1
        spatial_wv=spatial_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        spatial_wq=spatial_wq.permute(0,2,3,1).reshape(b,1,c//2) #bs,1,c//2
        spatial_wq=self.softmax_spatial(spatial_wq)
        spatial_wz=torch.matmul(spatial_wq,spatial_wv) #bs,1,h*w
        spatial_weight=self.sigmoid(spatial_wz.reshape(b,1,h,w)) #bs,1,h,w
        spatial_out=spatial_weight*channel_out
        return spatial_out

class MMTM(nn.Module):
    def __init__(self, dim_vis, dim_th, dim_ni, ratio):
        super(MMTM, self).__init__()
        dim = dim_ni+dim_th+dim_vis
        dim_out = int(dim*3/ratio)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.fc_z = nn.Linear(dim, dim_out)
        self.fc_vis = nn.Linear(dim_out, dim_vis)
        self.fc_ni = nn.Linear(dim_out, dim_ni)
        self.fc_th = nn.Linear(dim_out, dim_th)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, vis, th, ni):
        vis_b, vis_c, _, _ = vis.size()
        vis_out = self.squeeze(vis).view(vis_b, vis_c)
        ni_b, ni_c, _, _ = ni.size()
        ni_out = self.squeeze(ni).view(ni_b, ni_c)
        th_b, th_c, _, _ = th.size()
        th_out = self.squeeze(th).view(th_b, th_c)
        dim_z = torch.cat((vis_out, ni_out, th_out), dim=1)
        z = self.fc_z(dim_z)
        z = self.relu(z)
        E_vis = self.fc_vis(z)
        E_ni = self.fc_ni(z)
        E_th = self.fc_th(z)
        E_vis = self.sigmoid(E_vis).view(vis_b, vis_c, 1, 1)
        E_ni = self.sigmoid(E_ni).view(ni_b, ni_c, 1, 1)
        E_th = self.sigmoid(E_th).view(th_b, th_c, 1, 1)
        a, b, c = vis * E_vis.expand_as(vis)*2, th*E_th.expand_as(th)*2, ni*E_ni.expand_as(ni)*2
        output = torch.cat((a, b, c), dim=0)
        return output

class MMTMdemo(nn.Module):
    def __init__(self, dim_vis, dim_th, dim_ni, ratio):
        super(MMTM, self).__init__()
        dim = dim_ni + dim_th + dim_vis
        dim_out = int(dim * 3 / ratio)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.fc_z = nn.Linear(dim, dim_out)
        self.fc_vis = nn.Linear(dim_out, dim_vis)
        self.fc_ni = nn.Linear(dim_out, dim_ni)
        self.fc_th = nn.Linear(dim_out, dim_th)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, vis, th, ni):
        vis_b, vis_c, _, _ = vis.size()
        vis_out = self.squeeze(vis).view(vis_b, vis_c)
        ni_b, ni_c, _, _ = ni.size()
        ni_out = self.squeeze(ni).view(ni_b, ni_c)
        th_b, th_c, _, _ = th.size()
        th_out = self.squeeze(th).view(th_b, th_c)
        dim_z = torch.cat((vis_out, ni_out, th_out), dim=1)
        z = self.fc_z(dim_z)
        z = self.relu(z)
        E_vis = self.fc_vis(z)
        E_ni = self.fc_ni(z)
        E_th = self.fc_th(z)
        E_vis = self.sigmoid(E_vis).view(vis_b, vis_c, 1, 1)
        E_ni = self.sigmoid(E_ni).view(ni_b, ni_c, 1, 1)
        E_th = self.sigmoid(E_th).view(th_b, th_c, 1, 1)
        a, b, c = vis * E_vis.expand_as(vis) * 2, th * E_th.expand_as(th) * 2, ni * E_ni.expand_as(ni) * 2
        output = torch.cat((a, b, c), dim=0)
        ff = torch.cat((E_vis.expand_as(vis),E_th.expand_as(th),E_ni.expand_as(ni)), dim=0)
        return output, ff

class MMTMF(nn.Module):
    def __init__(self, dim_in, ratio):
        super().__init__()
        dim = dim_in * 3
        dim_out = int(dim/ratio)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.fc_z1 = nn.Linear(dim, dim_out)
        self.fc_z2 = nn.Linear(dim_out, dim)
        self.fc_z3 = nn.Linear(dim, dim_in)
        #self.fcvis = nn.Linear(dim, dim_in)
        #self.fcni = nn.Linear(dim, dim_in)
        #self.fcth = nn.Linear(dim, dim_in)
        self.fc_vis = nn.Sequential(
            nn.Linear(dim_in,dim_in//4),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in//4,dim_in),
            nn.Sigmoid()
        )
        self.fc_ni = nn.Sequential(
            nn.Linear(dim_in, dim_in //4),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in // 4, dim_in),
            nn.Sigmoid()
        )
        self.fc_th = nn.Sequential(
            nn.Linear(dim_in, dim_in // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in // 4, dim_in),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, vis, th, ni):
        b, c, _, _ = vis.size()
        vis_out = self.squeeze(vis).view(b, c)
        ni_out = self.squeeze(ni).view(b, c)
        th_out = self.squeeze(th).view(b, c)
        dim_z = torch.cat((vis_out, ni_out, th_out), dim=1)
        z = self.fc_z1(dim_z)
        z = self.relu(z)
        z = self.fc_z2(z)
        z = self.sigmoid(z)
        E_vis = self.fc_vis(vis_out)
        E_ni = self.fc_ni(ni_out)
        E_th = self.fc_th(th_out)
        E_z = torch.cat((E_vis, E_ni, E_th), dim=1)
        z = (z + E_z)/2
        z = self.fc_z3(z)
        E_vis = E_vis.view(b, c, 1, 1)
        E_ni = E_ni.view(b, c, 1, 1)
        E_th = E_th.view(b, c, 1, 1)
        z = z.view(b, c, 1, 1)
        a, b, c = vis * E_vis.expand_as(vis), E_th*z.expand_as(th), E_ni*z.expand_as(ni)
        output = torch.cat((a, b, c), dim=0)
        return output

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x1,x2,x3):
        b,c,h,w = x1.shape
        x1_k = self.qkv(x1)
        x1_v = self.qkv(x1)
        x1_q = self.qkv(x1)
        x2_k = self.qkv(x2)
        x2_v = self.qkv(x2)
        x2_q = self.qkv(x2)
        x3_k = self.qkv(x3)
        x3_v = self.qkv(x3)
        x3_q = self.qkv(x3)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # 在python3.5中@表示矩阵的乘法
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class ExternalAttention(nn.Module):

    def __init__(self, d_model,S=64):
        super().__init__()
        self.mq = nn.Linear(d_model, S, bias=False)
        self.mk = nn.Linear(d_model, S, bias=False)
        self.mv = nn.Linear(d_model, S, bias=False)
        self.line = nn.Linear(S,d_model,bias=False)
        self.softmax=nn.Softmax(dim=1)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, k, v):
        b, c, h, w = queries.size()
        #queries = self.squeeze(queries).view(b, c)
        #k = self.squeeze(k).view(b, c)
        #v = self.squeeze(v).view(b, c)
        queries = queries.permute(2, 3, 0, 1)
        k = k.permute(2, 3, 0, 1)
        v = v.permute(2, 3, 0, 1)

        attn_q = self.mq(queries) #bs,n,S
        attn_k = self.mk(k)  # bs,n,S
        attn_v = self.mv(v)  # bs,n,S
        qk = torch.matmul(attn_q.transpose(-1, -2),attn_k) #bs,n,S
        qk = self.softmax(qk)
        qv = torch.matmul(qk,attn_v.transpose(-1, -2))
        attn_qv = qv/torch.sum(qv,dim=2,keepdim=True) #bs,n,S 相当于layernormal
        out_q=self.line(attn_qv.transpose(-1, -2)).permute(2,3, 0, 1) #bs,n,d_model

        kv = torch.matmul(attn_k.transpose(-1, -2), attn_v)  # bs,n,S
        kv = self.softmax(kv)
        kq = torch.matmul(kv, attn_q.transpose(-1, -2))
        attn_kq = kq / torch.sum(kq, dim=2, keepdim=True)  # bs,n,S 相当于layernormal
        out_k = self.line(attn_kq.transpose(-1, -2)).permute(2,3, 0, 1)  # bs,n,d_model

        vq = torch.matmul(attn_v.transpose(-1, -2), attn_q)  # bs,n,S
        vq = self.softmax(vq)
        vk = torch.matmul(vq, attn_k.transpose(-1, -2))
        attn_vk = vk / torch.sum(vk, dim=2, keepdim=True)  # bs,n,S 相当于layernormal
        out_v = self.line(attn_vk.transpose(-1, -2)).permute(2,3, 0, 1)  # bs,n,d_model
        out = torch.cat((out_q, out_k, out_v), dim=0)

        return out

class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h,dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout=nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        import pdb
        pdb.set_trace()
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        qq = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(qq, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out

class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, height=14, heads=4):
        super().__init__()
        self.heads = heads

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        #self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)
        #self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        content_position = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)

        return out

class MQF(nn.Module):
    def __init__(self, n_dims, width=14, height=14, heads=4):
        super().__init__()
        self.heads = heads

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        #self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)
        #self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q1, q2, q3):
        n_batch, C, width, height = q1.size()
        q = self.query(q1).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(q2).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(q3).view(n_batch, self.heads, C // self.heads, -1)

        content_content_q = torch.matmul(q.permute(0, 1, 3, 2), k)
        content_content_k = torch.matmul(k.permute(0, 1, 3, 2), v)
        content_content_v = torch.matmul(v.permute(0, 1, 3, 2), q)

        content_position_q = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)
        content_position_q = torch.matmul(content_position_q, q)

        content_position_k = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)
        content_position_k = torch.matmul(content_position_k, k)

        content_position_v = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)
        content_position_v = torch.matmul(content_position_v, v)

        energy_q = content_content_q + content_position_q
        attention_q = self.softmax(energy_q)

        energy_k = content_content_k + content_position_k
        attention_k = self.softmax(energy_k)

        energy_v = content_content_v + content_position_v
        attention_v = self.softmax(energy_v)

        out_q = torch.matmul(v, attention_q.permute(0, 1, 3, 2))
        out_q = out_q.view(n_batch, C, width, height)

        out_k = torch.matmul(q, attention_k.permute(0, 1, 3, 2))
        out_k = out_k.view(n_batch, C, width, height)

        out_v = torch.matmul(k, attention_v.permute(0, 1, 3, 2))
        out_v = out_v.view(n_batch, C, width, height)
        output = torch.cat((out_q, out_k, out_v), dim=0)
        return output

if __name__ == '__main__':
    input=torch.randn(1,512,32,32)
    sa = MQF(n_dims=512, width=32, height=32, heads=4)
    output=sa(input,input,input)
    print(output.shape)
