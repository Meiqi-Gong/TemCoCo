import torch.nn as nn
from einops import rearrange
from torchvision.transforms import *
from archs.Blocks import *
from archs.DCNB_arch import VI_TDCN, VI_tmp

class Fusion(nn.Module):
    def __init__(self, 
                 n_feat=32):
        super(Fusion, self).__init__()
        act = nn.PReLU()
        self.Enc_ir = Encoder_v3(3, n_feat)
        self.Enc_vis = Encoder_v3(3, n_feat)
        
        window_size = (2, 8, 8)
        height = (1024 // window_size[1] + 1) * window_size[1]
        width = (1024 // window_size[2] + 1) * window_size[2]
        self.SegBranchCombine = Branch_and_Combine(nf=128, img_size=(height, width), window_size=window_size)
        self.DCN_1 = VI_TDCN(embed_dim=128)
        self.DCN_2 = VI_TDCN(embed_dim=64)
        self.dec = Decoder_v3(n_feat, 3)

    def forward(self, ir, vis):
        b, t, c, h, w = ir.shape
        x_ir = rearrange(ir, 'b t c h w -> b c t h w')
        x_vis = rearrange(vis, 'b t c h w -> b c t h w')
        feat_ir = self.Enc_ir(x_ir)
        feat_vis = self.Enc_vis(x_vis)
        combinefeat, ir_feat, vi_feat = self.SegBranchCombine(feat_ir[2], feat_vis[2])
        dcn_feat1, offsets1 = self.DCN_1(combinefeat, feat_ir[1].permute(0,2,1,3,4), feat_vis[1].permute(0,2,1,3,4))
        ir_input = feat_ir[0].permute(0,2,1,3,4)
        vi_input = feat_vis[0].permute(0,2,1,3,4)
        dcn_feat2, offsets2 = self.DCN_2(dcn_feat1, ir_input, vi_input, offsets=offsets1)
        fused = self.dec(dcn_feat2)
        return fused, ir_feat, vi_feat#, offsets1, offsets2