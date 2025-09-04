import sys
sys.path.append('SEA_RAFT/core')
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms.functional as FF
from PIL import Image
import cv2

from SEA_RAFT.config.parser import parse_args
import datasets
from raft import RAFT
from SEA_RAFT.core.utils.utils import resize_data, load_ckpt
from SEA_RAFT.core.utils.flow_viz import flow_to_image
from tqdm import tqdm

class SEA_RAFT_Evaluator:
    def __init__(self):
        self.args = self.parse_arguments()
        self.model = self.load_model()

    def parse_arguments(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--cfg', default='SEA_RAFT/config/eval/kitti-M.json', help='experiment configure file name', type=str)
        parser.add_argument('--model', default='/home/whu/HDD_16T/timer/gmq/video/SEA-RAFT-main/models/Tartan-C-T-TSKH-kitti432x960-M.pth', help='checkpoint path', type=str)
        args = parse_args(parser)
        return args

    def load_model(self):
        """ Load RAFT model with checkpoint """
        model = RAFT(self.args)
        load_ckpt(model, self.args.model)
        model = model.cuda()
        model.eval()
        print("Model loaded successfully.")
        return model

    @torch.no_grad()
    def forward_flow(self, image1, image2):
        """ Perform forward pass to compute optical flow """
        output = self.model(image1, image2, iters=self.args.iters, test_mode=True)
        flow_final = output['flow'][-1]
        info_final = output['info'][-1]
        return flow_final, info_final

    @torch.no_grad()
    def calc_flow(self, image1, image2):
        """ Calculate optical flow for given image pair """
        img1 = F.interpolate(image1, scale_factor=2 ** self.args.scale, mode='bilinear', align_corners=False)
        img2 = F.interpolate(image2, scale_factor=2 ** self.args.scale, mode='bilinear', align_corners=False)
        flow, info = self.forward_flow(img1, img2)
        
        flow_down = F.interpolate(flow, scale_factor=0.5 ** self.args.scale, mode='bilinear', align_corners=False) * (0.5 ** self.args.scale)
        info_down = F.interpolate(info, scale_factor=0.5 ** self.args.scale, mode='area')
        return flow_down, info_down

    @torch.no_grad()
    def validate(self, image1, image2):
        """ Validate and return the optical flow for input images """
        image1 = FF.resize(image1, (432, 960))
        image2 = FF.resize(image2, (432, 960))
        flow, _ = self.calc_flow(image1.cuda(), image2.cuda())
        return flow

    def eval(self, image1, image2):
        """ Evaluate optical flow on input images """
        flow = self.validate(image1, image2)
        return flow

    def run(self, image1, image2):
        """ Main method to run evaluation """
        flow = self.eval(image1, image2)
        w, h = image1.shape[-2:]
        flow1 = FF.resize(flow, (w, h))
        flow2 = FF.resize(flow, (w//2, h//2))
        return flow1, flow2


# 调用类进行评估
if __name__ == '__main__':
    evaluator = SEA_RAFT_Evaluator()

    image_path = '/home/whu/HDD_16T/timer/gmq/video/Grounded-SAM-2-main/custom_video_frames/vis/1.jpg'
    image = Image.open(image_path)
    image=np.array(image).astype(np.uint8)[..., :3]
    image1 = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)

    image_path = '/home/whu/HDD_16T/timer/gmq/video/Grounded-SAM-2-main/custom_video_frames/vis/7.jpg'
    image = Image.open(image_path)
    image=np.array(image).astype(np.uint8)[..., :3]
    image2 = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)
    # 这里可以传入两个图像张量 image1, image2，假设已经在外部准备好
    # 示例：image1, image2 = torch.rand(1, 3, 432, 960), torch.rand(1, 3, 432, 960)
    flow = evaluator.run(image1, image2)
    flow_vis = flow_to_image(flow[0].squeeze().permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True)
    cv2.imwrite(f"flow_final.jpg", flow_vis)
    ll=1
