# import sys
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.optim as optim
from collections import OrderedDict
from archs.Net import Fusion
from loss import Fusion_loss
l1_loss = torch.nn.L1Loss()
FusionNet=Fusion().cuda()
Fusion_loss = Fusion_loss().cuda()
from utils import *
from einops import rearrange
from dataset import create_train_val_dataloader, CPUPrefetcher
from options.options import parse_options
from torchvision.utils import make_grid
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.nn.functional as FF
import sys
root_path = osp.abspath(osp.join(__file__, osp.pardir))
opt = parse_options(root_path, is_train=True)

from torchvision.models import vgg19
vgg = vgg19(pretrained=True).features.cuda()#.to(local_rank)

DINOv2_model = torch.hub.load('/home/whu/HDD_16T/timer/gmq/video/dinov2-main/', 'dinov2_vitl14', pretrained=True, source='local').cuda()
DINOv2_model.eval()
vgg.eval()
for param in DINOv2_model.parameters():
    param.requires_grad = False
for param in vgg.parameters():
    param.requires_grad = False



optimizer = optim.Adam(FusionNet.parameters(), lr=opt['train']['optim_g']['lr'], betas=(0.9, 0.999), eps=1e-8)

def calcu_loss(fused, gt_ir, gt_vi, ir_feat, vi_feat):
    loss_dict = OrderedDict()

    loss_fusion = Fusion_loss(fused, gt_ir, gt_vi)
    loss_dict['fusion']=loss_fusion['loss_fusion']
    loss_dict['intensity']=loss_fusion['loss_intensity_max']
    loss_dict['color']=loss_fusion['loss_color']
    loss_dict['grad']=loss_fusion['loss_grad']
    loss_dict['ssim']=loss_fusion['loss_ssim']
    loss_dict['temporary']=loss_fusion['loss_temporary_consist']
    loss_dict['pixel']=loss_fusion['loss_pixel_consist']
    total_loss=loss_fusion['loss_fusion'].clone()

    vi_seg_gt=[]
    ir_seg_gt=[]
    vi_visual_gt = []
    ir_visual_gt = []
    B,C,T,H,W = vi_feat[1].shape
    gt_vi_reshape = gt_vi.reshape(-1, *gt_vi.shape[2:])
    gt_ir_reshape = gt_ir.reshape(-1, *gt_ir.shape[2:])
    vi_input = FF.interpolate(gt_vi_reshape, size=(h//16*14,w//16*14), mode='bilinear', align_corners=False)
    ir_input = FF.interpolate(gt_ir_reshape, size=(h//16*14,w//16*14), mode='bilinear', align_corners=False)
    with torch.no_grad():
        vi_visual_gt = vgg[:12](gt_vi_reshape)
        ir_visual_gt = vgg[:12](gt_ir_reshape)
        vi_seg_gt = DINOv2_model.forward_features(vi_input)
        vi_seg_gt = vi_seg_gt['x_norm_patchtokens'].reshape(B*T, 14, 14, 1024).permute(0,3,1,2)
        ir_seg_gt = DINOv2_model.forward_features(ir_input)
        ir_seg_gt = ir_seg_gt['x_norm_patchtokens'].reshape(B*T, 14, 14, 1024).permute(0,3,1,2)

    loss_visual = FF.mse_loss(rearrange(vi_feat[1], "b c t h w -> (b t) c h w"), vi_visual_gt)
    + FF.mse_loss(rearrange(ir_feat[1], "b c t h w -> (b t) c h w"), ir_visual_gt)
    loss_dict['vgg']=loss_visual 
    total_loss += 1*loss_visual
    loss_seg = FF.mse_loss(vi_feat[0], vi_seg_gt) + FF.mse_loss(ir_feat[0], ir_seg_gt)
    loss_dict['semantic']=loss_seg  
    total_loss += 1*loss_seg
    return total_loss, loss_fusion, loss_visual, loss_seg, loss_dict


start_epoch = 1
iter = 0
resume_state = load_resume_state(opt)
opt['world_size'] = 1
log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}.log")
os.makedirs(opt['path']['log'], exist_ok=True)
logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
logger.info(dict2str(opt))
tb_logger = init_tb_loggers(opt)
msg_logger = MessageLogger(opt, iter, tb_logger)
data = create_train_val_dataloader(opt)#, logger)
# start_epoch = 0
train_loader, train_sampler, val_loader, total_epochs, total_iters = data

model = FusionNet
model_name = 'experiments/VideoFusion_IRnoise2/models/VideoFusionNet_495000.pth'
state_dict = torch.load(model_name, map_location=lambda storage, loc: storage)
model.load_state_dict(state_dict, strict=True)

prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
if prefetch_mode is None or prefetch_mode == 'cpu':
    prefetcher = CPUPrefetcher(train_loader)

print(f'Start training from epoch: {start_epoch}, iter: {iter}')
data_time, iter_time = time.time(), time.time()
start_time = time.time()
# iter=0

FusionNet = FusionNet.train()
for epoch in range(start_epoch, total_epochs + 1):
    train_sampler.set_epoch(epoch)
    prefetcher.reset()
    train_data = prefetcher.next()
    while train_data is not None:
        iter+=1
        if iter>0:
            lq_ir = train_data['lq_ir'].cuda()
            lq_vi = train_data['lq_vi'].cuda()
            gt_ir = train_data['gt_ir'].cuda()
            gt_vi = train_data['gt_vi'].cuda()
            b,t,c,h,w = gt_vi.shape 

            fused, ir_feat, vi_feat = model(lq_ir, lq_vi)

            total_loss, loss_fusion, loss_visual, loss_seg, loss_dict = calcu_loss(fused, gt_ir, gt_vi, ir_feat, vi_feat)
            fused = rearrange(fused, 'b c t h w -> b t c h w')

            optimizer.zero_grad()
            total_loss.backward()
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is None:
                    print(f"Parameter {name} did not receive gradients.")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # if rank==1:
            log_dict = reduce_loss_dict(opt, loss_dict)
            iter_time = time.time() - iter_time

            if iter % 200 == 0:
                for param in optimizer.param_groups:
                    param['lr'] = param['lr']*0.95

            if iter % opt['logger']['print_freq'] == 0:                
                print("===> Epoch/iter[{}/{}]:Loss_all: {:.4f}  Loss_fusion: {:.4f}  loss_seg: {:.4f}  loss_vgg: {:.4f}".format(epoch, iter, total_loss, loss_fusion['loss_fusion'].data, torch.tensor(loss_seg).data, loss_visual.data))
                log_vars = {'epoch': epoch, 'iter': iter}
                log_vars.update({'lrs': [param_group['lr'] for param_group in optimizer.param_groups]})
                log_vars.update({'time': iter_time, 'data_time': data_time})
                log_vars.update(log_dict)
                msg_logger(log_vars)
                b, n, c, h, w = gt_ir.size()
                t = n // 2
                tb_img_first = [
                    gt_ir[0, t, ::].detach().float().cpu(), 
                    gt_vi[0, t, ::].detach().float().cpu(),
                    fused[0, t, ::].detach().float().cpu()
                ]
                tb_img_first_degrade = [
                    lq_ir[0, t, ::].detach().float().cpu(), 
                    lq_vi[0, t, ::].detach().float().cpu(),
                    fused[0, t, ::].detach().float().cpu()
                ]
                tb_img_last = [
                    gt_ir[-1, t, ::].detach().float().cpu(), 
                    gt_vi[-1, t, ::].detach().float().cpu(),
                    fused[-1, t, ::].detach().float().cpu()
                ]
                tb_img_last_degrade = [
                    lq_ir[-1, t, ::].detach().float().cpu(), 
                    lq_vi[-1, t, ::].detach().float().cpu(),
                    fused[-1, t, ::].detach().float().cpu()
                ]
                tb_img = tb_img_first_degrade + tb_img_first + tb_img_last_degrade + tb_img_last
                tb_img = make_grid(tb_img, nrow=3, padding=2)
                tb_logger.add_image('images', tb_img, iter)

                if torch.isnan(total_loss):
                    sys.exit(0)

            if iter % opt['logger']['save_checkpoint_freq'] == 0:
                save_filename = f'VideoFusionNet_{iter}.pth'
                save_path = os.path.join(opt['path']['models'], save_filename)
                torch.save(model.state_dict(), save_path)

            
            data_time = time.time()
            iter_time = time.time()
        # dist.barrier()
        train_data = prefetcher.next()
