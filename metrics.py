
from SEA_RAFT_Evaluator import SEA_RAFT_Evaluator 
# flow calculation method, using code from: https://github.com/princeton-vl/SEA-RAFT
flow_evaluator = SEA_RAFT_Evaluator()
import numpy as np
import torch
from torch.nn import functional as F

def flowD(preoutput_img, output_img, pretarget_img, target_img):
    """
    preoutput_img: \phi_f^(t-1),
    output_img: \phi_f^t,
    pretarget_img: \phi_vis^(t-1),
    target_img: \phi_vis^t,
    """
    flow = flow_evaluator.run(pretarget_img, target_img) #must first resize image to (432, 960) before flow calculation
    warp_image = flow_evaluator.warp_frame_with_flow(preoutput_img, flow)
    OF_diff = np.absolute(output_img - warp_image)
    OF_diff = np.sqrt(np.sum(OF_diff * OF_diff, axis = -1))
    return  OF_diff.mean()

def feaCD(preoutput_img, output_img, pretarget_img, target_img, pretarget_img2, target_img2):
    """
    preoutput_img: \phi_f^(t-1),
    output_img: \phi_f^t,
    pretarget_img: \phi_vis^(t-1),
    target_img: \phi_vis^t,
    pretarget_img2: \phi_ir^(t-1),
    target_img2: \phi_ir^t,
    """
    from torchvision.models import resnet18
    model = resnet18(pretrained=True)
    model.eval()
    output_img = output_img.astype(np.float32)
    preoutput_img = preoutput_img.astype(np.float32)
    feature1 = model(torch.tensor(output_img.copy()).unsqueeze(0).permute(0,3,1,2))
    feature2 = model(torch.tensor(preoutput_img.copy()).unsqueeze(0).permute(0,3,1,2))
    motion_direction = feature2 - feature1
    feature1 = model(torch.tensor(target_img.astype(np.float32).copy()).unsqueeze(0).permute(0,3,1,2))
    feature2 = model(torch.tensor(pretarget_img.astype(np.float32).copy()).unsqueeze(0).permute(0,3,1,2))
    motion_tdirection = feature2 - feature1
    feature1 = model(torch.tensor(target_img2.astype(np.float32).copy()).unsqueeze(0).permute(0,3,1,2))
    feature2 = model(torch.tensor(pretarget_img2.astype(np.float32).copy()).unsqueeze(0).permute(0,3,1,2))
    motion_tdirection2 = feature2 - feature1
    cosine_sim = F.cosine_similarity(motion_direction, motion_tdirection, dim=0)
    cosine_sim2 = F.cosine_similarity(motion_direction, motion_tdirection2, dim=0)
    diff_loss = (1 - cosine_sim).mean().cpu().detach().numpy() + (1 - cosine_sim2).mean().cpu().detach().numpy()
    return  diff_loss