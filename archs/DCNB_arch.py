import torch
import torch.nn as nn
# from torch.nn.modules.utils import _pair
# from torch.nn.modules.module import Module
import math
from dcn import DeformConv
# from deform_conv import DeformConv
# import deform_conv
# print(dir(deform_conv))  # 查看该模块包含的类和函数

from einops import rearrange
from archs.net_basics import Separable3DConv
# from Modulated_DCN.modulated_deform_conv import *
# from mmcv.ops import DeformConv2dPack

class VI_tmp(nn.Module):
    def __init__(self, embed_dim=128):
        super(VI_tmp, self).__init__()
        # self.conv_first = nn.Conv2d(3, 64, 3, padding=1, bias=True)

        self.relu = nn.ReLU(inplace=True)

        # fea_ex = [nn.Conv2d(5 * 3, 64, 3, padding= 1, bias=True),
        #                nn.ReLU()]

        # self.fea_ex = nn.Sequential(*fea_ex)
        self.PS = nn.PixelShuffle(2)
        self.ConvTrans = nn.Conv3d(embed_dim, embed_dim*2, kernel_size=1, stride=1, padding=0, bias=True)
        self.cross_modality = nn.Sequential(nn.Conv2d(embed_dim//2*3, embed_dim//2, 3, padding=1, bias=True), nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.cross_frame = nn.Sequential(nn.Conv2d(embed_dim//2*3, embed_dim//2, 3, padding=1, bias=True), nn.LeakyReLU(negative_slope=0.2, inplace=True))
        # self.cross_frame2 = nn.Sequential(nn.Conv2d(embed_dim//2*3, embed_dim//2, 3, padding=1, bias=True), nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.temporary_cat = nn.Sequential(nn.Conv2d(embed_dim, embed_dim//2, 3, padding=1, bias=True), nn.LeakyReLU(negative_slope=0.2, inplace=True))
        # self.temporary_cat2 = nn.Sequential(nn.Conv2d(embed_dim, embed_dim//2, 3, padding=1, bias=True), nn.LeakyReLU(negative_slope=0.2, inplace=True))
        # self.before_offsets = nn.Conv2d(embed_dim//2, embed_dim//2, 3, padding=1, bias=True)
        # self.before_offsets2 = nn.Conv2d(embed_dim//2, embed_dim//2, 3, padding=1, bias=True)
        # self.off2d_1 = nn.Conv2d(embed_dim//2, 18 * 8, 3, padding=1, bias=True)
        # self.off2d_2 = nn.Conv2d(embed_dim//2, 18 * 8, 3, padding=1, bias=True)
        # self.dconv_1 = DeformConv(embed_dim//2, embed_dim//2, 3, padding=1, deformable_groups=8, groups=embed_dim // (2 * 8))
        # self.dconv_2 = DeformConv(embed_dim//2, embed_dim//2, 3, padding=1, deformable_groups=8, groups=embed_dim // (2 * 8))
        # self.instead_dconv = nn.Conv2d(embed_dim//2, embed_dim//2, 3, padding=1, bias=True)
        # self.fuse_neighb = nn.Sequential(nn.Conv2d(embed_dim, embed_dim//2, 3, padding=1, bias=True), nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.fuse_feature = Separable3DConv(embed_dim, embed_dim//2)

        # xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def align(self, x, vfeat, ifeat, ff=None):
        y = []
        offset= []
        batch_size, num, ch, w, h = x.shape

        for i in range(num):
            ref = x[:, i, :, :, :].clone()
            vref = vfeat[:, i, :, :, :].clone()
            iref = ifeat[:, i, :, :, :].clone()
            
            # if i<20:
            ref_fea = torch.cat([ref, vref, iref], dim=1)
            neighb_fea = torch.cat([ref, vref, iref], dim=1)
            fea = self.cross_modality(ref_fea)
            nfea = self.cross_frame(neighb_fea)
            fea_trans = self.temporary_cat(torch.cat([fea, nfea], 1))
            # cfea = self.before_offsets(fea_trans)
            # fea = self.instead_dconv(cfea)
            y.append(fea_trans)
            # else:
            #     ref_fea = torch.cat([ref, vref, iref], dim=1)
            #     neighb_fea = torch.cat([x[:, i-1, :, :, :], 
            #                             vfeat[:, i-1, :, :, :], 
            #                             ifeat[:, i-1, :, :, :]], dim=1)
            #     fea = self.cross_modality(ref_fea)
            #     nfea = self.cross_frame(neighb_fea)
            #     fea_trans1 = self.temporary_cat(torch.cat([fea, nfea], 1))

            #     neighb_fea = torch.cat([x[:, i-2, :, :, :], 
            #                             vfeat[:, i-2, :, :, :], 
            #                             ifeat[:, i-2, :, :, :]], dim=1)
            #     nfea = self.cross_frame2(neighb_fea)
            #     fea_trans2 = self.temporary_cat2(torch.cat([fea, nfea], 1))

            #     fea = self.fuse_neighb(torch.cat([fea_trans1, fea_trans2], 1))
            #     y.append(fea)
        y = torch.cat([item.unsqueeze(1) for item in y], dim=1)
        return y, fea

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x, vfeat, ifeat, offsets=None):

        x = self.PS(self.ConvTrans(x).permute(0,2,1,3,4).contiguous())
        aligned_f, offsets = self.align(x, vfeat, ifeat) # motion alignments

        enhanced_f = self.fuse_feature(torch.cat([x, aligned_f], 2).permute(0,2,1,3,4).contiguous())+x.permute(0,2,1,3,4).contiguous()

        return enhanced_f, offsets


class VI_TDCN(nn.Module):
    def __init__(self, embed_dim=128):
        super(VI_TDCN, self).__init__()
        self.name = 'VI_TDCN'
        # self.conv_first = nn.Conv2d(3, 64, 3, padding=1, bias=True)

        self.relu = nn.ReLU(inplace=True)
        self.up2 = nn.Upsample(scale_factor=2)

        # fea_ex = [nn.Conv2d(5 * 3, 64, 3, padding= 1, bias=True),
        #                nn.ReLU()]

        # self.fea_ex = nn.Sequential(*fea_ex)
        self.PS = nn.PixelShuffle(2)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.ConvTrans = nn.Conv3d(embed_dim, embed_dim//2, kernel_size=1, stride=1, padding=0, bias=True)
        self.cross_modality = nn.Sequential(nn.Conv2d(embed_dim//2*3, embed_dim//2, 3, padding=1, bias=True), nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.cross_frame = nn.Sequential(nn.Conv2d(embed_dim//2*3, embed_dim//2, 3, padding=1, bias=True), nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.cross_frame2 = nn.Sequential(nn.Conv2d(embed_dim//2*3, embed_dim//2, 3, padding=1, bias=True), nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.temporary_cat = nn.Sequential(nn.Conv2d(embed_dim, embed_dim//2, 3, padding=1, bias=True), nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.temporary_cat2 = nn.Sequential(nn.Conv2d(embed_dim, embed_dim//2, 3, padding=1, bias=True), nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.before_offsets = nn.Conv2d(embed_dim//2, embed_dim//2, 3, padding=1, bias=True)
        self.before_offsets2 = nn.Conv2d(embed_dim//2, embed_dim//2, 3, padding=1, bias=True)
        self.off2d_1 = nn.Conv2d(embed_dim//2, 18 * 8, 3, padding=1, bias=True)
        self.off2d_2 = nn.Conv2d(embed_dim//2, 18 * 8, 3, padding=1, bias=True)
        # self.dconv_1 = ConvOffset2d(64, 64, 3, padding=1, num_deformable_groups=8)
        # self.dconv_1 = DeformConv2d(embed_dim//2, embed_dim//2, 3, padding=1, bias=False, modulation=True)
        self.dconv_1 = DeformConv(embed_dim//2, embed_dim//2, 3, padding=1, deformable_groups=8, groups=embed_dim // (2 * 8))#
        self.dconv_2 = DeformConv(embed_dim//2, embed_dim//2, 3, padding=1, deformable_groups=8, groups=embed_dim // (2 * 8))
        # self.dconv_2 = DeformConv2d(embed_dim//2, embed_dim//2, 3, padding=1, bias=False, modulation=True)
        # self.instead_dconv = nn.Conv2d(embed_dim//2*3, embed_dim//2, 3, padding=1, bias=True)
        self.fuse_neighb = nn.Sequential(nn.Conv2d(embed_dim, embed_dim//2, 3, padding=1, bias=True), nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.fuse_feature = Separable3DConv(embed_dim, embed_dim//2)

        # xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def align(self, x, vfeat, ifeat, offsets=None):
        y = []
        offset= []
        batch_size, num, ch, w, h = x.shape

        for i in range(num):
            ref = x[:, i, :, :, :].clone()
            vref = vfeat[:, i, :, :, :].clone()
            iref = ifeat[:, i, :, :, :].clone()
            ref_fea = torch.cat([ref, vref, iref], dim=1)
            
            if i<2:
                # neighb_fea = torch.cat([x[:, i+1, :, :, :], 
                #                         vfeat[:, i+1, :, :, :], 
                #                         ifeat[:, i+1, :, :, :]], dim=1)
                neighb_fea = ref_fea
                fea = self.cross_modality(ref_fea)
                nfea = self.cross_frame(neighb_fea)
                fea_trans = self.temporary_cat(torch.cat([fea, nfea], 1))
                cfea = self.before_offsets(fea_trans)
                if offsets is None:
                    offset1 = self.off2d_1(cfea)
                else:
                    res = self.up2(offsets[i][0])
                    offset1 = self.off2d_1(cfea)+res
                fea1 = self.dconv_1(nfea, offset1)

                # neighb_fea = torch.cat([x[:, i+2, :, :, :], 
                #                         vfeat[:, i+2, :, :, :], 
                #                         ifeat[:, i+2, :, :, :]], dim=1)
                neighb_fea = ref_fea
                nfea = self.cross_frame2(neighb_fea)
                fea_trans = self.temporary_cat2(torch.cat([fea, nfea], 1))
                cfea = self.before_offsets2(fea_trans)
                if offsets is None:
                    offset2 = self.off2d_2(cfea)
                else:
                    res = self.up2(offsets[i][1])
                    offset2 = self.off2d_2(cfea)+res
                fea2 = self.dconv_2(nfea, offset2)

                fea = self.fuse_neighb(torch.cat([fea1, fea2], 1))
                y.append(fea)
                offset.append([offset1, offset2])
            else:
            # if i>=2:
                neighb_fea = torch.cat([x[:, i-1, :, :, :], 
                                        vfeat[:, i-1, :, :, :], 
                                        ifeat[:, i-1, :, :, :]], dim=1)
                fea = self.cross_modality(ref_fea)
                nfea = self.cross_frame(neighb_fea)
                fea_trans = self.temporary_cat(torch.cat([fea, nfea], 1))
                cfea = self.before_offsets(fea_trans)
                if offsets is None:
                    offset1 = self.off2d_1(cfea)
                else:
                    res = self.up2(offsets[i][0])
                    offset1 = self.off2d_1(cfea)+res
                fea1 = self.dconv_1(nfea, offset1)# + nfea

                # neighb_fea = torch.cat([x[:, 1, :, :, :] if i<7 else x[:, i-5, :, :, :], 
                #                         vfeat[:, 1, :, :, :] if i<7 else vfeat[:, i-5, :, :, :],
                #                         ifeat[:, 1, :, :, :] if i<7 else ifeat[:, i-5, :, :, :]], dim=1)
                neighb_fea = torch.cat([x[:, i-2, :, :, :], 
                                        vfeat[:, i-2, :, :, :], 
                                        ifeat[:, i-2, :, :, :]], dim=1)
                nfea = self.cross_frame2(neighb_fea)
                fea_trans = self.temporary_cat2(torch.cat([fea, nfea], 1))
                cfea = self.before_offsets2(fea_trans)
                if offsets is None:
                    offset2 = self.off2d_2(cfea)
                else:
                    res = self.up2(offsets[i][1])
                    offset2 = self.off2d_2(cfea)+res
                fea2 = self.dconv_2(nfea, offset2)# + nfea
                # fea2, offset2 = self.dconv_2(fea_trans)

                fea = self.fuse_neighb(torch.cat([fea1, fea2], 1))
                y.append(fea)
                offset.append([offset1, offset2])
        y = torch.cat([item.unsqueeze(1) for item in y], dim=1)
        return y, offset

    # def align(self, x, vfeat, ifeat, offsets=None):
    #     y = []
    #     offset= []
    #     batch_size, num, ch, w, h = x.shape

    #     for i in range(num):
    #         ref = x[:, i, :, :, :].clone()
    #         vref = vfeat[:, i, :, :, :].clone()
    #         iref = ifeat[:, i, :, :, :].clone()
    #         ref_fea = torch.cat([ref, vref, iref], dim=1)
            
    #         if i==0:
    #             neighb_fea = ref_fea
    #             fea = self.cross_modality(ref_fea)
    #             nfea = self.cross_frame(neighb_fea)
    #             fea_trans = self.temporary_cat(torch.cat([fea, nfea], 1))
    #             cfea = self.before_offsets(fea_trans)
    #             if offsets is None:
    #                 offset1 = self.off2d_1(cfea)
    #             else:
    #                 res = self.up2(offsets[i][0])
    #                 offset1 = self.off2d_1(cfea)+res
    #             fea1 = self.dconv_1(nfea, offset1) + nfea

    #             neighb_fea = torch.cat([x[:, i+1, :, :, :], 
    #                                     vfeat[:, i+1, :, :, :], 
    #                                     ifeat[:, i+1, :, :, :]], dim=1)
    #             # neighb_fea = ref_fea
    #             nfea = self.cross_frame2(neighb_fea)
    #             fea_trans = self.temporary_cat2(torch.cat([fea, nfea], 1))
    #             cfea = self.before_offsets2(fea_trans)
    #             if offsets is None:
    #                 offset2 = self.off2d_2(cfea)
    #             else:
    #                 res = self.up2(offsets[i][1])
    #                 offset2 = self.off2d_2(cfea)+res
    #             fea2 = self.dconv_2(nfea, offset2) + nfea

    #             fea = self.fuse_neighb(torch.cat([fea1, fea2], 1))
    #             y.append(fea)
    #             offset.append([offset1, offset2])
    #         elif i==num-1:
    #             neighb_fea = torch.cat([x[:, i-1, :, :, :], 
    #                                     vfeat[:, i-1, :, :, :], 
    #                                     ifeat[:, i-1, :, :, :]], dim=1)
    #             fea = self.cross_modality(ref_fea)
    #             nfea = self.cross_frame(neighb_fea)
    #             fea_trans = self.temporary_cat(torch.cat([fea, nfea], 1))
    #             cfea = self.before_offsets(fea_trans)
    #             if offsets is None:
    #                 offset1 = self.off2d_1(cfea)
    #             else:
    #                 res = self.up2(offsets[i][0])
    #                 offset1 = self.off2d_1(cfea)+res
    #             fea1 = self.dconv_1(nfea, offset1) + nfea

    #             neighb_fea = ref_fea
    #             nfea = self.cross_frame2(neighb_fea)
    #             fea_trans = self.temporary_cat2(torch.cat([fea, nfea], 1))
    #             cfea = self.before_offsets2(fea_trans)
    #             if offsets is None:
    #                 offset2 = self.off2d_2(cfea)
    #             else:
    #                 res = self.up2(offsets[i][1])
    #                 offset2 = self.off2d_2(cfea)+res
    #             fea2 = self.dconv_2(nfea, offset2) + nfea

    #             fea = self.fuse_neighb(torch.cat([fea1, fea2], 1))
    #             y.append(fea)
    #             offset.append([offset1, offset2])
    #         else:
    #         # if i>=2:
    #             neighb_fea = torch.cat([x[:, i-1, :, :, :], 
    #                                     vfeat[:, i-1, :, :, :], 
    #                                     ifeat[:, i-1, :, :, :]], dim=1)
    #             fea = self.cross_modality(ref_fea)
    #             nfea = self.cross_frame(neighb_fea)
    #             fea_trans = self.temporary_cat(torch.cat([fea, nfea], 1))
    #             cfea = self.before_offsets(fea_trans)
    #             if offsets is None:
    #                 offset1 = self.off2d_1(cfea)
    #             else:
    #                 res = self.up2(offsets[i][0])
    #                 offset1 = self.off2d_1(cfea)+res
    #             fea1 = self.dconv_1(nfea, offset1) + nfea

    #             neighb_fea = torch.cat([x[:, i+1, :, :, :], 
    #                                     vfeat[:, i+1, :, :, :], 
    #                                     ifeat[:, i+1, :, :, :]], dim=1)
    #             nfea = self.cross_frame2(neighb_fea)
    #             fea_trans = self.temporary_cat2(torch.cat([fea, nfea], 1))
    #             cfea = self.before_offsets2(fea_trans)
    #             if offsets is None:
    #                 offset2 = self.off2d_2(cfea)
    #             else:
    #                 res = self.up2(offsets[i][1])
    #                 offset2 = self.off2d_2(cfea)+res
    #             fea2 = self.dconv_2(nfea, offset2) + nfea
    #             # fea2, offset2 = self.dconv_2(fea_trans)

    #             fea = self.fuse_neighb(torch.cat([fea1, fea2], 1))
    #             y.append(fea)
    #             offset.append([offset1, offset2])
    #     y = torch.cat([item.unsqueeze(1) for item in y], dim=1)
    #     return y, offset
    
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x, vfeat, ifeat, offsets=None):

        # x = self.PS(self.ConvTrans(x).permute(0,2,1,3,4).contiguous())
        b = x.shape[0]
        x = self.up2(rearrange(self.ConvTrans(x), 'b c t h w-> (b t) c h w').contiguous())
        x = rearrange(x, '(b t) c h w -> b t c h w', b=b)
        # _, ch, w, h = x.size() 
        # x = x.reshape(b, -1, ch, w, h)
        # batch_size, t, ch, w, h = x.size() 

        # align supporting frames
        aligned_f, offsets = self.align(x, vfeat, ifeat, offsets) # motion alignments

        enhanced_f = self.fuse_feature(torch.cat([x, aligned_f], 2).permute(0,2,1,3,4).contiguous())+x.permute(0,2,1,3,4).contiguous()

        return enhanced_f, offsets
    

