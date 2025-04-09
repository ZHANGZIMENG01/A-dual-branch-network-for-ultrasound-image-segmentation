import torch
from torch import nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_
import math
from models.cc_attention.functions import CrissCrossAttention
import torch
import torch.nn as nn
import torch.nn.functional as F
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3)
        return scale

class CRU(nn.Module):
    def __init__(self, op_channel:int=128, alpha:float = 1/2, squeeze_radio:int = 2, group_size:int = 2, group_kernel_size:int = 3):
        super().__init__()

        self.up_channel     = up_channel = int(alpha * op_channel)
        self.low_channel    = low_channel = op_channel - up_channel
        self.squeeze1       = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2       = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        self.GWC            = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1, padding=group_kernel_size // 2, groups=group_size)  # 创建卷积层
        self.PWC1           = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        self.PWC2           = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1, bias=False)  # 创建卷积层
        self.advavg         = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


class DFAM(nn.Module):
    def __init__(self, inplanes, channel_rate=2, reduction_ratio=16):
        super(DFAM, self).__init__()
        self.in_channels = inplanes
        self.inter_channels = inplanes // channel_rate
        if self.inter_channels == 0:
            self.inter_channels = 1
        self.common_v = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                         padding=0)
        self.Trans_s = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels)
        )
        nn.init.constant_(self.Trans_s[1].weight, 0)
        nn.init.constant_(self.Trans_s[1].bias, 0)
        self.Trans_q = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels)
        )
        nn.init.constant_(self.Trans_q[1].weight, 0)
        nn.init.constant_(self.Trans_q[1].bias, 0)

        self.key = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                             padding=0)
        self.query = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                           padding=0)

        self.dropout = nn.Dropout(0.1)
        self.ChannelGate = ChannelGate(self.in_channels, pool_types=['avg'], reduction_ratio=reduction_ratio)
        self.CRU=CRU(op_channel=128, alpha=0.5, squeeze_radio=2, group_size=2, group_kernel_size=3)

    def forward(self, q, s):
        batch_size, channels, height_q, width_q = q.shape
        batch_size, channels, height_s, width_s = s.shape
        # common feature learning
        v_q = self.common_v(q).view(batch_size, self.inter_channels, -1)
        v_q = v_q.permute(0, 2, 1)

        v_s = self.common_v(s).view(batch_size, self.inter_channels, -1)
        v_s = v_s.permute(0, 2, 1)

        k_x = self.key(s).view(batch_size, self.inter_channels, -1)
        k_x = k_x.permute(0, 2, 1)

        q_x = self.query(q).view(batch_size, self.inter_channels, -1)

        A_s = torch.matmul(k_x, q_x)
        attention_s = F.softmax(A_s, dim=-1)
        p_s = torch.matmul(attention_s, v_s)
        p_s = p_s.permute(0, 2, 1).contiguous()
        p_s = p_s.view(batch_size, self.inter_channels, height_s, width_s)
        p_s = self.Trans_s(p_s)
        E_s = self.ChannelGate(s) * p_s
        E_s = E_s + s
        E_q = E_s + self.CRU(q)

        return E_q

class SKConv(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16, L=32):
        super(SKConv, self).__init__()
        d = max(in_channels // reduction, L)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(out_channels, d, kernel_size=1)
        self.fc2 = nn.Conv2d(d, out_channels * 2, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        conv3 = self.conv3x3(x)
        conv5 = self.conv5x5(x)
        U = conv3 + conv5
        s = self.global_pool(U)
        z = self.fc1(s)
        a_b = self.fc2(F.relu(z))
        a_b = a_b.view(x.size(0), 2, -1, 1, 1)
        a_b = self.softmax(a_b)
        conv3 = conv3 * a_b[:, 0, :, :, :]
        conv5 = conv5 * a_b[:, 1, :, :, :]
        out = conv3 + conv5

        return out

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding,
                      stride=stride, dilation=dilation, groups=dim_in)
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))

class GatedAttentionUnit(nn.Module):
    def __init__(self, in_c, out_c, kernel_size):
        super().__init__()
        self.w1 = nn.Sequential(
            DepthWiseConv2d(in_c, in_c, kernel_size, padding=kernel_size//2),
            nn.Sigmoid()
        )

        self.w2 = nn.Sequential(
            DepthWiseConv2d(in_c, in_c, kernel_size + 2, padding=(kernel_size + 2)//2),
            nn.GELU()
        )
        self.wo = nn.Sequential(
            DepthWiseConv2d(in_c, out_c, kernel_size),
            nn.GELU()
        )

        self.cw = nn.Conv2d(in_c, out_c, 1)
        
    def forward(self, x):
        x1, x2 = self.w1(x), self.w2(x)
        out = self.wo(x1 * x2) + self.cw(x)
        return out


class DilatedGatedAttention(nn.Module):
    def __init__(self, in_c, out_c, k_size=3, dilated_ratio=[7, 5, 2, 1]):
        super().__init__()        
        
        self.mda0 = nn.Conv2d(in_c//4, in_c//4, kernel_size=k_size, stride=1, 
                              padding=(k_size+(k_size-1)*(dilated_ratio[0]-1))//2, 
                             dilation=dilated_ratio[0], groups=in_c//4)
        self.mda1 = nn.Conv2d(in_c//4, in_c//4, kernel_size=k_size, stride=1, 
                              padding=(k_size+(k_size-1)*(dilated_ratio[1]-1))//2, 
                             dilation=dilated_ratio[1], groups=in_c//4)
        self.mda2 = nn.Conv2d(in_c//4, in_c//4, kernel_size=k_size, stride=1, 
                              padding=(k_size+(k_size-1)*(dilated_ratio[2]-1))//2, 
                             dilation=dilated_ratio[2], groups=in_c//4)
        self.mda3 = nn.Conv2d(in_c//4, in_c//4, kernel_size=k_size, stride=1, 
                              padding=(k_size+(k_size-1)*(dilated_ratio[3]-1))//2, 
                             dilation=dilated_ratio[3], groups=in_c//4)
        self.norm_layer = nn.GroupNorm(4, in_c)
        self.conv = nn.Conv2d(in_c, in_c, 1)
        
        self.gau = GatedAttentionUnit(in_c, out_c, 3)
        
    def forward(self, x):
        x = torch.chunk(x, 4, dim=1)
        x0 = self.mda0(x[0])
        x1 = self.mda1(x[1])
        x2 = self.mda2(x[2])
        x3 = self.mda3(x[3])
        x = F.gelu(self.conv(self.norm_layer(torch.cat((x0, x1, x2, x3), dim=1))))
        x = self.gau(x)
        return x
    
    
class EAblock(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_c, in_c, 1)

        self.k = in_c * 4
        self.linear_0 = nn.Conv1d(in_c, self.k, 1, bias=False)

        self.linear_1 = nn.Conv1d(self.k, in_c, 1, bias=False)
        self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0, 2)        
        
        self.conv2 = nn.Conv2d(in_c, in_c, 1, bias=False)
        self.norm_layer = nn.GroupNorm(4, in_c)   

    def forward(self, x):
        idn = x
        x = self.conv1(x)

        b, c, h, w = x.size()
        x = x.view(b, c, h*w)   # b * c * n 

        attn = self.linear_0(x) # b, k, n
        attn = F.softmax(attn, dim=-1) # b, k, n

        attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True)) #  # b, k, n
        x = self.linear_1(attn) # b, c, n

        x = x.view(b, c, h, w)
        x = self.norm_layer(self.conv2(x))
        x = x + idn
        x = F.gelu(x)
        return x
    

class Channel_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        c_plist=[8, 16, 24, 32, 48, 64]
        c_plist_sum = sum(c_plist) - c_plist[-1]
        self.split_att = split_att
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.att1 = nn.Linear(c_plist_sum, c_plist[0]) if split_att == 'fc' else nn.Conv1d(c_plist_sum, c_plist[0], 1)
        self.att2 = nn.Linear(c_plist_sum, c_plist[1]) if split_att == 'fc' else nn.Conv1d(c_plist_sum, c_plist[1], 1)
        self.att3 = nn.Linear(c_plist_sum, c_plist[2]) if split_att == 'fc' else nn.Conv1d(c_plist_sum, c_plist[2], 1)
        self.att4 = nn.Linear(c_plist_sum, c_plist[3]) if split_att == 'fc' else nn.Conv1d(c_plist_sum, c_plist[3], 1)
        self.att5 = nn.Linear(c_plist_sum, c_plist[4]) if split_att == 'fc' else nn.Conv1d(c_plist_sum, c_plist[4], 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, t1, t2, t3, t4, t5):
        att = torch.cat((self.avgpool(t1), 
                         self.avgpool(t2), 
                         self.avgpool(t3), 
                         self.avgpool(t4), 
                         self.avgpool(t5)), dim=1)
        att = self.get_all_att(att.squeeze(-1).transpose(-1, -2))
        if self.split_att != 'fc':
            att = att.transpose(-1, -2)
        att1 = self.sigmoid(self.att1(att))
        att2 = self.sigmoid(self.att2(att))
        att3 = self.sigmoid(self.att3(att))
        att4 = self.sigmoid(self.att4(att))
        att5 = self.sigmoid(self.att5(att))
        if self.split_att == 'fc':
            att1 = att1.transpose(-1, -2).unsqueeze(-1).expand_as(t1)
            att2 = att2.transpose(-1, -2).unsqueeze(-1).expand_as(t2)
            att3 = att3.transpose(-1, -2).unsqueeze(-1).expand_as(t3)
            att4 = att4.transpose(-1, -2).unsqueeze(-1).expand_as(t4)
            att5 = att5.transpose(-1, -2).unsqueeze(-1).expand_as(t5)
        else:
            att1 = att1.unsqueeze(-1).expand_as(t1)
            att2 = att2.unsqueeze(-1).expand_as(t2)
            att3 = att3.unsqueeze(-1).expand_as(t3)
            att4 = att4.unsqueeze(-1).expand_as(t4)
            att5 = att5.unsqueeze(-1).expand_as(t5)
            
        return att1, att2, att3, att4, att5
    
    
class Spatial_Att_Bridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_conv2d = nn.Sequential(nn.Conv2d(2, 1, 7, stride=1, padding=9, dilation=3),
                                          nn.Sigmoid())
    
    def forward(self, t1, t2, t3, t4, t5):
        t_list = [t1, t2, t3, t4, t5]
        att_list = []
        for t in t_list:
            avg_out = torch.mean(t, dim=1, keepdim=True)
            max_out, _ = torch.max(t, dim=1, keepdim=True)
            att = torch.cat([avg_out, max_out], dim=1)
            att = self.shared_conv2d(att)
            att_list.append(att)
        return att_list[0], att_list[1], att_list[2], att_list[3], att_list[4]

class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num:int, group_num:int = 16, eps:float = 1e-10):
        super(GroupBatchnorm2d,self).__init__()
        assert c_num >= group_num
        self.group_num  = group_num
        self.gamma      = nn.Parameter(torch.randn(c_num, 1, 1))
        self.beta       = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps        = eps

    def forward(self, x):
        N, C, H, W  = x.size()
        x           = x.view(N, self.group_num, -1)
        mean        = x.mean(dim=2, keepdim=True)
        std         = x.std(dim=2, keepdim=True)
        x           = (x - mean) / (std + self.eps)
        x           = x.view(N, C, H, W)
        return x * self.gamma + self.beta


class SRU(nn.Module):
    def __init__(self,
                 oup_channels:int,  #
                 group_num:int = 16,
                 gate_treshold:float = 0.5,
                 torch_gn:bool = False
                 ):
        super().__init__()

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x  = self.gn(x)
        w_gamma = self.gn.gamma / sum(self.gn.gamma)
        reweights = self.sigomid(gn_x * w_gamma)


        info_mask = reweights >= self.gate_treshold
        noninfo_mask = reweights < self.gate_treshold
        x_1 = info_mask * x
        x_2 = noninfo_mask * x
        x = self.reconstruct(x_1, x_2)
        return x

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRU(nn.Module):
    def __init__(self, op_channel:int, alpha:float = 1/2, squeeze_radio:int = 2, group_size:int = 2, group_kernel_size:int = 3):
        super().__init__()

        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)


        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1, padding=group_kernel_size // 2, groups=group_size)  # 创建卷积层
        self.PWC1  = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)


        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1, bias=False)  # 创建卷积层
        self.advavg= nn.AdaptiveAvgPool2d(1)

    def forward(self, x):

        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)


        Y1 = self.GWC(up) + self.PWC1(up)


        Y2 = torch.cat([self.PWC2(low), low], dim=1)


        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


class SPOM(nn.Module):
    def __init__(self, op_channel:int, group_num:int = 16, gate_treshold:float = 0.5, alpha:float = 1/2, squeeze_radio:int = 2, group_size:int = 2, group_kernel_size:int = 3):
        super().__init__()

        self.SRU = SRU(op_channel, group_num=group_num, gate_treshold=gate_treshold)
        self.CRU = CRU(op_channel, alpha=alpha, squeeze_radio=squeeze_radio, group_size=group_size, group_kernel_size=group_kernel_size)
        self.cca = CrissCrossAttention(in_dim=op_channel)

    def forward(self, x):
        x = self.SRU(x)
        x=self.cca(self.cca(x))
        return x
class SC_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        
        self.catt = Channel_Att_Bridge(c_list, split_att=split_att)
        self.satt = Spatial_Att_Bridge()
        
    def forward(self, t1, t2, t3, t4, t5):
        r1, r2, r3, r4, r5 = t1, t2, t3, t4, t5

        satt1, satt2, satt3, satt4, satt5 = self.satt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = satt1 * t1, satt2 * t2, satt3 * t3, satt4 * t4, satt5 * t5

        r1_, r2_, r3_, r4_, r5_ = t1, t2, t3, t4, t5
        t1, t2, t3, t4, t5 = t1 + r1, t2 + r2, t3 + r3, t4 + r4, t5 + r5

        catt1, catt2, catt3, catt4, catt5 = self.catt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = catt1 * t1, catt2 * t2, catt3 * t3, catt4 * t4, catt5 * t5

        return t1 + r1_, t2 + r2_, t3 + r3_, t4 + r4_, t5 + r5_
    
    
class DBUNet(nn.Module):
    
    def __init__(self, num_classes=1, input_channels=3, c_list=[8,16,24,32,48,64,128],
                split_att='fc', bridge=True):
        super().__init__()

        self.bridge = bridge
        
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 =nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        ) 
        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
        )
        self.encoder4 = nn.Sequential(
            EAblock(c_list[2]),
            DilatedGatedAttention(c_list[2], c_list[3]),
        )
        self.encoder5 = nn.Sequential(
            EAblock(c_list[3]),
            DilatedGatedAttention(c_list[3], c_list[4]),
        )
        self.encoder6 = nn.Sequential(
            EAblock(c_list[4]),
            DilatedGatedAttention(c_list[4], c_list[5]),
        )
        self.encoder7=nn.Sequential(
            SKConv(c_list[5], c_list[6]),
        )

        self.dfam=DFAM(inplanes=128, channel_rate=2, reduction_ratio=16).cuda()
        self.spom = SPOM(op_channel=24)

        if bridge: 
            self.scab = SC_Att_Bridge(c_list, split_att)
            print('SC_Att_Bridge was used')
        self.decoder0 = nn.Sequential(
            SKConv(c_list[6], c_list[5]),
        )
        self.decoder1 = nn.Sequential(
            DilatedGatedAttention(c_list[5], c_list[4]),
            EAblock(c_list[4]),
        ) 
        self.decoder2 = nn.Sequential(
            DilatedGatedAttention(c_list[4], c_list[3]),
            EAblock(c_list[3]),
        ) 
        self.decoder3 = nn.Sequential(
            DilatedGatedAttention(c_list[3], c_list[2]),
            EAblock(c_list[2]),
        )  
        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
        )  
        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )

        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])
        self.mask_change

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, img, en):
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(img)),2,2))
        t1 = out # b, c0, H/2, W/2
        out_0=F.gelu(F.max_pool2d(self.ebn1(self.encoder1(en)),2,2))

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        t2 = out # b, c1, H/4, W/4
        out_1=F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out_0)),2,2))

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        t3 = out # b, c2, H/8, W/8
        out_2 = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out_1)), 2, 2))

        out_2 =self.spom(out_2)
        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)),2,2))
        t4 = out # b, c3, H/16, W/16
        out_3 = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out_2)), 2, 2))

        
        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)),2,2))
        t5 = out # b, c4, H/32, W/32
        out_4=F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out_3)),2,2))

        if self.bridge: t1, t2, t3, t4, t5 = self.scab(t1, t2, t3, t4, t5)
        
        out = F.gelu(self.encoder6(out)) # b, c5, H/32, W/32
        out = F.gelu(self.encoder7(out)) #b, c6, H/64, W/64
        out_5=F.gelu(self.encoder6(out_4))
        out_5=F.gelu(self.encoder7(out_5))


        out=self.dfam(out,out_5)


        out5 = F.gelu(self.decoder0(out))
        out5 = F.gelu(self.dbn1(self.decoder1(out5))) # b, c4, H/32, W/32
        out5 = torch.add(out5, t5) # b, c4, H/32, W/32
        
        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c3, H/16, W/16
        out4 = torch.add(out4, t4) # b, c3, H/16, W/16
        
        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c2, H/8, W/8
        out3 = torch.add(out3, t3) # b, c2, H/8, W/8
        
        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c1, H/4, W/4
        out2 = torch.add(out2, t2) # b, c1, H/4, W/4 
        
        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c0, H/2, W/2
        out1 = torch.add(out1, t1) # b, c0, H/2, W/2
        
        out0 = F.interpolate(self.final(out1),scale_factor=(2,2),mode ='bilinear',align_corners=True) # b, num_class, H, W
        output=torch.sigmoid(out0)
        return output

