import torch.nn as nn
from archs.net_basics import *
from einops import rearrange
import torch.nn.functional as F
from archs.psrt_sliding_arch import CombineAttn, CrossAttention, CrossAttn

class Encoder_v3(nn.Module):
    def __init__(self, in_dims, nf):
        super(Encoder_v3, self).__init__()
        self.conv0 = Separable3DConv(in_dims, nf)
        # nn.Sequential(Conv3DModule(in_dims, nf, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
        #                            Conv3DModule(in_dims, nf, kernel_size=(3,1,1), stride=1, padding=(1,0,0)))
        self.conv1 = conv_resblock_two(nf, nf)
        self.conv2 = conv_resblock_two(nf, 2*nf, stride=2)
        self.conv3 = conv_resblock_two(2*nf, 4*nf, stride=2)
    
    def forward(self, x):
        x_ =  self.conv0(x)
        f1 = self.conv1(x_)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        return [f1, f2, f3]

class Branch_Seg(nn.Module):
    def __init__(self, nf):
        super(Branch_Seg, self).__init__()
        self.conv1 = nn.Sequential(conv_resblock_two(nf, nf))
        self.down = nn.Conv2d(nf,1024,kernel_size=3,stride=4)
        self.up = nn.ConvTranspose2d(1024,nf,kernel_size=3,stride=4,output_padding=1)
        # Conv3DModule(nf, 1024, kernel_size=(1,1,1), stride=1, padding=(0,0,0)))
                                #    conv_resblock_two(2*nf, 2*nf),
                                #    conv_resblock_two(2*nf, 2*nf),
                                #    conv_resblock_one(5*nf, 1024))
        # self.resize()
        # self.linear1 = nn.Linear(nf, 1024)
        # self.linear2 = nn.Linear(1024, nf*2)
        self.conv2 = nn.Sequential(conv_resblock_two(nf*2, nf))
    
    def forward(self, x):
        f1 = self.conv1(x)
        seg_output = self.down(rearrange(f1, 'b c t h w -> (b t) c h w'))
        f2 = self.up(seg_output)
        f2 = self.conv2(torch.cat([torch.reshape(f2, f1.shape), f1], 1))
        return f2, seg_output
    # def forward(self, seg_output):
    #     f2 = self.up(seg_output)
    #     f2 = self.conv2(rearrange(f2, '(b t) c h w -> b c t h w', b=1))
    #     return f2, seg_output

class Branch_Visual(nn.Module):
    def __init__(self, nf):
        super(Branch_Visual, self).__init__()
        self.conv1 = conv_resblock_two(nf, 2*nf)
        self.conv2 = conv_resblock_two(2*nf, nf)
        # self.conv2 = conv_resblock_two(2*nf, 2*nf)
    
    def forward(self, x):
        vis_output = self.conv1(x)
        f2 = self.conv2(vis_output)
        return f2, vis_output
    # def forward(self, vis_output):
    #     # vis_output = self.conv1(x)
    #     f2 = self.conv2(vis_output)
    #     return f2, vis_output

class Branch_and_Combine_noattn(nn.Module):
    def __init__(self, nf, img_size, window_size):
        super(Branch_and_Combine_noattn, self).__init__()
        self.Branch_s1 = Branch_Seg(nf)
        self.Branch_v1 = Branch_Visual(nf)
        self.Branch_s2 = Branch_Seg(nf)
        self.Branch_v2 = Branch_Visual(nf)
        self.combine_seg = Conv3DModule(nf*2, nf, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0)) #conv_resblock_two(nf*2, nf)
        self.combine_vis = Conv3DModule(nf*2, nf, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))
        self.SegVisAttn = Conv3DModule(nf*2, nf, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))

        # self.combine = conv_resblock_two(nf*4, nf)
        self.attn_temp = CrossAttention(
            query_dim=nf,
            heads=8,
            dim_head=nf//2
        )

    def forward(self, feat_ir, feat_vi):
        vi_feat_seg, vi_out_seg = self.Branch_s1(feat_vi)
        vi_feat_visual, vi_out_visual = self.Branch_v1(feat_vi)
        ir_feat_seg, ir_out_seg = self.Branch_s2(feat_ir)
        ir_feat_visual, ir_out_visual = self.Branch_v2(feat_ir)
        feat_seg = self.combine_seg(torch.cat([vi_feat_seg, ir_feat_seg], 1))
        # feat_seg = torch.cat([vi_feat_seg, ir_feat_seg], 1)
        feat_visual = self.combine_vis(torch.cat([vi_feat_visual, ir_feat_visual], 1))
        # feat_visual = torch.cat([vi_feat_visual, ir_feat_visual], 1)
        feat = self.SegVisAttn(torch.cat([feat_seg, feat_visual], 1))
        # feat = self.combine(torch.cat([attn_seg, attn_visual], 1))
        feat = self.attn_temp(feat)

        return feat, [ir_out_seg, ir_out_visual], [vi_out_seg, vi_out_visual]

class Branch_and_Combine(nn.Module):
    def __init__(self, nf, img_size, window_size):
        super(Branch_and_Combine, self).__init__()
        self.Branch_s1 = Branch_Seg(nf)
        self.Branch_v1 = Branch_Visual(nf)
        self.Branch_s2 = Branch_Seg(nf)
        self.Branch_v2 = Branch_Visual(nf)
        # self.combine_seg1 = Conv3DModule(nf*2, nf*2, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))
        # self.before_combine2 = Conv3DModule(nf*4, nf*2, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))
        # self.before_combine3 = Conv3DModule(nf*2, nf, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))
        # self.before_combine4 = Conv3DModule(nf*2, nf, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))
        # self.combine_seg = CrossAttn(
        #                    dim=nf,
        #                    num_heads = 8
        #                 )
        # self.combine_vis = CrossAttn(
        #                    dim=nf,
        #                    num_heads = 8
        #                 )
        # self.SegVisAttn = CrossAttn(
        #                    dim=nf*2,
        #                    num_heads = 8
        #                 )
        self.combine_seg = CombineAttn(
                           dim=nf,
                           input_resolution=img_size,
                           depth=(1),
                           num_heads=(4),
                           window_size=window_size
                        )
        self.combine_vis = CombineAttn(
                           dim=nf,
                           input_resolution=img_size,
                           depth=(1),
                           num_heads=(4),
                           window_size=window_size
                        )
        self.SegVisAttn = CombineAttn(
                           dim=nf*2,
                           input_resolution=img_size,
                           depth=(1),
                           num_heads=(4),
                           window_size=window_size
                        )
        self.combine = conv_resblock_two(nf*4, nf)
        self.attn_temp = CrossAttention(
            query_dim=nf,
            heads=8,
            dim_head=nf//2
        )

    def forward(self, feat_ir, feat_vi):
        vi_feat_seg, vi_out_seg = self.Branch_s1(feat_vi)
        vi_feat_visual, vi_out_visual = self.Branch_v1(feat_vi)
        ir_feat_seg, ir_out_seg = self.Branch_s2(feat_ir)
        ir_feat_visual, ir_out_visual = self.Branch_v2(feat_ir)
        vi_feat_seg, ir_feat_seg = self.combine_seg(vi_feat_seg, ir_feat_seg)
        feat_seg = torch.cat([vi_feat_seg, ir_feat_seg], 1)
        vi_feat_visual, ir_feat_visual = self.combine_vis(vi_feat_visual, ir_feat_visual)
        feat_visual = torch.cat([vi_feat_visual, ir_feat_visual], 1)
        attn_seg, attn_visual = self.SegVisAttn(feat_seg, feat_visual)
        feat = self.combine(torch.cat([attn_seg, attn_visual], 1))
        feat = self.attn_temp(feat)

        return feat, [ir_out_seg, ir_out_visual], [vi_out_seg, vi_out_visual]
    
    #gtfeat
    # def forward(self, feat_ir, feat_vi):
    #     vi_feat_seg, vi_out_seg = self.Branch_s1(feat_vi[0])
    #     vi_feat_visual, vi_out_visual = self.Branch_v1(feat_vi[1])
    #     ir_feat_seg, ir_out_seg = self.Branch_s2(feat_ir[0])
    #     ir_feat_visual, ir_out_visual = self.Branch_v2(feat_ir[1])
    #     vi_feat_seg, ir_feat_seg = self.combine_seg(vi_feat_seg, ir_feat_seg)
    #     feat_seg = torch.cat([vi_feat_seg, ir_feat_seg], 1)
    #     vi_feat_visual, ir_feat_visual = self.combine_vis(vi_feat_visual, ir_feat_visual)
    #     feat_visual = torch.cat([vi_feat_visual, ir_feat_visual], 1)
    #     attn_seg, attn_visual = self.SegVisAttn(feat_seg, feat_visual)
    #     feat = self.combine(torch.cat([attn_seg, attn_visual], 1))
    #     feat = self.attn_temp(feat)

    #     return feat, [ir_out_seg, ir_out_visual], [vi_out_seg, vi_out_visual]

    #newattn
    # def forward(self, feat_ir, feat_vi):
    #     vi_feat_seg, vi_out_seg = self.Branch_s1(feat_vi[0])
    #     ir_feat_seg, ir_out_seg = self.Branch_s2(feat_ir[0])
    #     vi_feat_visual, vi_out_visual = self.Branch_v1(feat_vi[1])
    #     ir_feat_visual, ir_out_visual = self.Branch_v2(feat_ir[1])

    #     b, c, t, h, w = vi_feat_seg.shape
    #     vi_feat_seg = rearrange(vi_feat_seg, "b c t h w -> (b t) (h w) c")
    #     ir_feat_seg = rearrange(ir_feat_seg, "b c t h w -> (b t) (h w) c")
    #     vi_feat_seg, ir_feat_seg = self.combine_seg(vi_feat_seg, ir_feat_seg, selfattn=False)
    #     feat_seg = torch.cat([vi_feat_seg, ir_feat_seg], 2)
    #     vi_feat_visual = rearrange(vi_feat_visual, "b c t h w -> (b t) (h w) c")
    #     ir_feat_visual = rearrange(ir_feat_visual, "b c t h w -> (b t) (h w) c")
    #     vi_feat_visual, ir_feat_visual = self.combine_vis(vi_feat_visual, ir_feat_visual, selfattn=False)
    #     feat_visual = torch.cat([vi_feat_visual, ir_feat_visual], 2)
    #     attn_seg, attn_visual = self.SegVisAttn(feat_seg, feat_visual, selfattn=False)
    #     seg_visual = rearrange(torch.cat([attn_seg, attn_visual], 2), "(b t) (h w) c -> b c t h w", h=h, w=w, b=b)
    #     feat = self.combine(seg_visual)
    #     feat = self.attn_temp(feat)

    #     return feat, feat_ir, feat_vi

    #noattn
    # def forward(self, feat_ir, feat_vi):
    #     vi_feat_seg, vi_out_seg = self.Branch_s1(feat_vi[0])
    #     ir_feat_seg, ir_out_seg = self.Branch_s2(feat_ir[0])
    #     vi_feat_visual, vi_out_visual = self.Branch_v1(feat_vi[1])
    #     ir_feat_visual, ir_out_visual = self.Branch_v2(feat_ir[1])
    #     # vi_feat_seg, ir_feat_seg = self.combine_seg(vi_feat_seg, ir_feat_seg)
    #     feat_seg = torch.cat([vi_feat_seg, ir_feat_seg], 1)
    #     # vi_feat_visual, ir_feat_visual = self.combine_vis(vi_feat_visual, ir_feat_visual)
    #     feat_visual = torch.cat([vi_feat_visual, ir_feat_visual], 1)
    #     # attn_seg, attn_visual = self.SegVisAttn(feat_seg, feat_visual)
    #     feat = self.combine(torch.cat([feat_seg, feat_visual], 1))
    #     feat = self.attn_temp(feat)

    #     return feat, feat_ir, feat_vi
    
class Decoder_v3(nn.Module):
    def __init__(self, in_dims, nf):
        super(Decoder_v3, self).__init__()
        self.conv0 = Separable3DConv(in_dims, nf)
        # nn.Sequential(Conv3DModule(in_dims, nf, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
        #                            Conv3DModule(in_dims, nf, kernel_size=(3,1,1), stride=1, padding=(1,0,0)))
        # self.conv1 = conv_resblock_two(nf, nf)
        # self.conv2 = conv_resblock_two(nf, 2*nf, stride=2)
        # self.conv3 = conv_resblock_two(2*nf, 4*nf, stride=2)
    
    def forward(self, x):
        x_ =  self.conv0(x)
        # f1 = self.conv1(x_)
        # f2 = self.conv2(f1)
        # f3 = self.conv3(f2)
        return x_
    
class Decoder(nn.Module):
    def __init__(self, in_dims, nf):
        super(Decoder, self).__init__()
        self.up2 = nn.Upsample(scale_factor=2)
        self.up4 = nn.Upsample(scale_factor=4)
        self.ConvTrans1 = nn.Conv3d(in_dims*4, in_dims, kernel_size=1, stride=1, padding=0, bias=True)
        self.ConvTrans2 = nn.Conv3d(in_dims*2, in_dims, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv0 = Separable3DConv(in_dims*3, nf)
        # nn.Sequential(Conv3DModule(in_dims, nf, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
        #                            Conv3DModule(in_dims, nf, kernel_size=(3,1,1), stride=1, padding=(1,0,0)))
        # self.conv1 = conv_resblock_two(nf, nf)
        # self.conv2 = conv_resblock_two(nf, 2*nf, stride=2)
        # self.conv3 = conv_resblock_two(2*nf, 4*nf, stride=2)
    
    def forward(self, x1, x2, x3):
        b = x1.shape[0]
        x1 = self.up4(rearrange(self.ConvTrans1(x1), 'b c t h w-> (b t) c h w').contiguous())
        x1 = rearrange(x1, '(b t) c h w -> b c t h w', b=b)
        x2 = self.up2(rearrange(self.ConvTrans2(x2), 'b c t h w-> (b t) c h w').contiguous())
        x2 = rearrange(x2, '(b t) c h w -> b c t h w', b=b)
        x_ =  self.conv0(torch.cat([x1,x2,x3], 1))
        # f1 = self.conv1(x_)
        # f2 = self.conv2(f1)
        # f3 = self.conv3(f2)
        return x_