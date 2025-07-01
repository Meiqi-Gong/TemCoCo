
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import shutil
from natsort import natsorted
from archs.Net import Fusion
FusionNet=Fusion().cuda()
from utils import *
from einops import rearrange
from options.options import parse_options
from dataset import create_test_dataloader, CPUPrefetcher
import warnings
warnings.filterwarnings("ignore")

root_path = osp.abspath(osp.join(__file__, osp.pardir))
opt = parse_options(root_path, is_train=False)

weights = torch.load("experiments/VideoFusion_IRnoise2/models/VideoFusionNet_495000.pth", map_location='cuda')
savepath = 'results/temp'
new_weights = {}
# for key, value in weights.items():
#     new_key = key.replace('module.', '')  # Remove 'module.' prefix
#     new_weights[new_key] = value
new_weights = weights if new_weights=={} else new_weights
FusionNet.load_state_dict(new_weights, strict=True)

test_loader = create_test_dataloader(opt)
prefetcher = CPUPrefetcher(test_loader)
min_max=(0, 1)
with torch.no_grad():
        for iteration, batch in enumerate(test_loader, 1):
            lq_ir, gt_ir, lq_vi, gt_vi, folder_name=batch['lq_ir'], batch['gt_ir'], batch['lq_vi'], batch['gt_vi'], batch['folder']
            # if folder_name[0].startswith("1204_1140"):
                # lq_ir, lq_vi = lq_ir.cuda(), lq_vi.cuda()
                # fused, _, _, = FusionNet(lq_ir,lq_vi)
            # gt_ir, gt_vi = gt_ir.cuda(), gt_vi.cuda()
            # fused, _, _, = FusionNet(gt_ir,gt_vi)
            lq_ir, lq_vi = lq_ir.cuda(), lq_vi.cuda()
            fused, _, _, = FusionNet(lq_ir,lq_vi)
            fused = rearrange(fused, 'b c t h w -> b t c h w')
            for i in range(fused.shape[1]):
                img = fused[0,i].float().detach().cpu().clamp_(*min_max).numpy().transpose(1, 2, 0)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = (img * 255.0).round()
                img = img.astype(np.uint8)
                os.makedirs(os.path.join(savepath, folder_name[0]), exist_ok=True)
                cv2.imwrite(os.path.join(savepath, folder_name[0], str(i)+'.jpg'), img)


root_dir = savepath

# 获取所有子文件夹
subdirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

# 按前缀分组
prefix_dict = {}
for subdir in subdirs:
    prefix, suffix = "_".join(subdir.split("/")[-1].split("-")[:-1]), subdir.split("/")[-1].split("-")[-1]
    if prefix not in prefix_dict:
        prefix_dict[prefix] = []
    prefix_dict[prefix].append((subdir, suffix))

# 对每组前缀进行操作
for prefix, dirs in prefix_dict.items():
    # 创建目标文件夹
    # target_dir = os.path.join(root_dir, prefix)
    sorted_dirs = sorted(dirs, key=lambda x: int(x[1]))
    target_dir = os.path.join('results/folder/', prefix)
    os.makedirs(target_dir, exist_ok=True)
    
    file_counter = 1  # 文件编号起始值
    
    # 遍历这些子文件夹
    for d, _ in sorted_dirs:
        for file in natsorted(os.listdir(d)):
            old_file_path = os.path.join(d, file)
            if os.path.isfile(old_file_path):
                # 构造新文件路径
                new_file_name = f"{file_counter}.jpg"
                new_file_path = os.path.join(target_dir, new_file_name)
                
                # 移动并重命名文件
                shutil.copy(old_file_path, new_file_path)
                file_counter += 1
        file_counter -= 5

print("文件移动与重命名完成！")
