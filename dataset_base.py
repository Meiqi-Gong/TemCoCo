from pathlib import Path
import torch
from torch.utils import data as data
from utils import get_root_logger, FileClient, imfrombytes, paired_random_crop_multi, img2tensor, scandir, read_img_seq, generate_frame_indices
import random
from os import path as osp
import glob

class VideoFusionDataset(data.Dataset):

    def __init__(self, opt):
        super(VideoFusionDataset, self).__init__()
        self.opt = opt
        # 分别是高清图像和低清图像的路径
        self.ir_gt_root, self.ir_lq_root = Path(opt["dataroot_gt_ir"]), Path(
            opt["dataroot_lq_ir"]
        )
        self.vi_gt_root, self.vi_lq_root = Path(opt["dataroot_gt_vi"]), Path(
            opt["dataroot_lq_vi"]
        )
        self.num_frame = opt["num_frame"]
        
        self.keys = []
        if opt["test_mode"]==True:
            with open(opt["meta_info_file_test"], "r") as fin:
                for line in fin:
                    folder, frame_num, _ = line.split(" ")
                    self.keys.extend(
                        [f"{folder}/{i}/{frame_num}" for i in range(int(frame_num))]
                    )
        elif opt["test_mode"]==False:
            with open(opt["meta_info_file_train"], "r") as fin:
                for line in fin:
                    folder, frame_num, _ = line.split(" ")
                    self.keys.extend(
                        [
                            f"{folder}/{i}/{frame_num}"
                            for i in range(1, int(frame_num) + 1)
                        ]
                    )
        else:
            with open(opt["meta_info_file_val"], "r") as fin:
                for line in fin:
                    folder, frame_num, _ = line.split(" ")
                    self.keys.extend(
                        [
                            f"{folder}/{i}/{frame_num}"
                            for i in range(1, int(frame_num) + 1)
                        ]
                    )            

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt["io_backend"]
        self.is_lmdb = False
        # if self.io_backend_opt["type"] == "lmdb":
        #     self.is_lmdb = True
        #     self.io_backend_opt["db_paths"] = [self.lq_root, self.gt_root]
        #     self.io_backend_opt["client_keys"] = ["lq", "gt"]

        # temporal augmentation configs
        self.interval_list = opt.get("interval_list", [1])
        self.random_reverse = opt.get("random_reverse", False)
        interval_str = ",".join(str(x) for x in self.interval_list)
        logger = get_root_logger()
        logger.info(
            f"Temporal augmentation interval list: [{interval_str}]; "
            f"random reverse is {self.random_reverse}."
        )

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop("type"), **self.io_backend_opt
            )

        scale = self.opt["scale"]
        gt_size = self.opt["gt_size"]
        key = self.keys[index]
        clip_name, frame_name, frame_num = key.split("/")  # key example: 000/000000

        # determine the neighboring frames
        interval = random.choice(self.interval_list)
        # print(f"frame_name: {int(frame_name)}")
        # ensure not exceeding the borders
        start_frame_idx = int(frame_name) - 1
        if start_frame_idx > int(frame_num) - self.num_frame:
            start_frame_idx = random.randint(0, int(frame_num) - self.num_frame)
        end_frame_idx = start_frame_idx + self.num_frame

        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        # get the neighboring LQ and GT frames
        img_ir_lqs, img_vi_lqs = [], []
        img_ir_gts, img_vi_gts = [], []
        for neighbor in neighbor_list:
            if self.is_lmdb:
                img_ir_lq_path = self.ir_lq_root / clip_name / f"{neighbor+1}.jpg"
                img_vi_lq_path = self.vi_lq_root / clip_name / f"{neighbor+1}.jpg"
                img_ir_gt_path = self.ir_gt_root / clip_name / f"{neighbor+1}.jpg"
                img_vi_gt_path = self.vi_gt_root / clip_name / f"{neighbor+1}.jpg"
            else:
                img_ir_lq_path = self.ir_lq_root / clip_name / f"{neighbor+1}.jpg"
                img_vi_lq_path = self.vi_lq_root / clip_name / f"{neighbor+1}.jpg"
                img_ir_gt_path = self.ir_gt_root / clip_name / f"{neighbor+1}.jpg"
                img_vi_gt_path = self.vi_gt_root / clip_name / f"{neighbor+1}.jpg"

            # get LQ
            img_ir_bytes = self.file_client.get(img_ir_lq_path, "lq")
            img_vi_bytes = self.file_client.get(img_vi_lq_path, "lq")
            
            img_ir_lq = imfrombytes(img_ir_bytes, float32=True)
            img_ir_lqs.append(img_ir_lq)
            
            img_vi_lq = imfrombytes(img_vi_bytes, float32=True)
            img_vi_lqs.append(img_vi_lq)

            # get GT
            img_ir_bytes = self.file_client.get(img_ir_gt_path, "gt")
            img_vi_bytes = self.file_client.get(img_vi_gt_path, "gt")
            img_ir_gt = imfrombytes(img_ir_bytes, float32=True)
            img_vi_gt = imfrombytes(img_vi_bytes, float32=True)
            img_ir_gts.append(img_ir_gt)
            img_vi_gts.append(img_vi_gt)

        # randomly crop
        img_ir_gts, img_ir_lqs, img_vi_gts, img_vi_lqs = paired_random_crop_multi(
            img_ir_gts, img_ir_lqs, img_vi_gts, img_vi_lqs, gt_size, scale, img_ir_gt_path
        )
        
        # augmentation - flip, rotate
        img_ir_lqs.extend(img_ir_gts)
        img_vi_lqs.extend(img_vi_gts)
        img_ir_results = img_ir_lqs
        img_vi_results = img_vi_lqs
        
        # img_ir_results = augment(img_ir_results, self.opt["use_flip"], self.opt["use_rot"])
        # img_vi_results = augment(img_vi_results, self.opt["use_flip"], self.opt["use_rot"])

        img_ir_results = img2tensor(img_ir_results)
        img_vi_results = img2tensor(img_vi_results)
        
        img_ir_gts = torch.stack(img_ir_results[len(img_ir_lqs) // 2 :], dim=0)
        img_ir_lqs = torch.stack(img_ir_results[: len(img_ir_lqs) // 2], dim=0)

        img_vi_gts = torch.stack(img_vi_results[len(img_vi_lqs) // 2 :], dim=0)
        img_vi_lqs = torch.stack(img_vi_results[: len(img_vi_lqs) // 2], dim=0)
        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str
        return {"lq_ir": img_ir_lqs, "lq_vi":img_vi_lqs, "gt_ir": img_ir_gts, "gt_vi": img_vi_gts, "key": key}

    def __len__(self):
        return len(self.keys)

class VideoTestDataset(data.Dataset):
    """Video test dataset.

    Supported datasets: Vid4, REDS4, REDSofficial.
    More generally, it supports testing dataset with following structures:

    dataroot
    ├── subfolder1
        ├── frame000
        ├── frame001
        ├── ...
    ├── subfolder1
        ├── frame000
        ├── frame001
        ├── ...
    ├── ...

    For testing datasets, there is no need to prepare LMDB files.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            io_backend (dict): IO backend type and other kwarg.
            cache_data (bool): Whether to cache testing datasets.
            name (str): Dataset name.
            meta_info_file (str): The path to the file storing the list of test
                folders. If not provided, all the folders in the dataroot will
                be used.
            num_frame (int): Window size for input frames.
            padding (str): Padding mode.
    """

    def __init__(self, opt):
        super(VideoTestDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.ir_gt_root, self.ir_lq_root = opt['dataroot_gt_ir'], opt['dataroot_lq_ir']
        self.vi_gt_root, self.vi_lq_root = opt['dataroot_gt_vi'], opt['dataroot_lq_vi']
        
        self.data_info = {'ir_lq_path': [], 'ir_gt_path': [], 'vi_lq_path': [], 'vi_gt_path': [],'folder': [], 'idx': [], 'border': []}
        
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        assert self.io_backend_opt['type'] != 'lmdb', 'No need to use lmdb during validation/test.'

        logger = get_root_logger()
        logger.info(f'Generate data info for VideoTestDataset - {opt["name"]}')
        self.imgs_ir_lq, self.imgs_ir_gt, self.imgs_vi_lq, self.imgs_vi_gt = {}, {}, {}, {}
        if 'meta_info_file' in opt:
            with open(opt['meta_info_file_test'], 'r') as fin:
                subfolders = [line.split(' ')[0] for line in fin]
                subfolders_lq = [osp.join(self.lq_root, key) for key in subfolders]
                subfolders_gt = [osp.join(self.gt_root, key) for key in subfolders]
        else:
            subfolders_ir_lq = sorted(glob.glob(osp.join(self.ir_lq_root, '*'))) ## 遍历self.lq_root下所有子文件夹 并排序 结果是一个路径列表 而不是文件夹名 列表
            subfolders_ir_gt = sorted(glob.glob(osp.join(self.ir_gt_root, '*'))) ## 遍历self.gt_root下所有子文件夹 并排序
            subfolders_vi_lq = sorted(glob.glob(osp.join(self.vi_lq_root, '*'))) ## 遍历self.lq_root下所有子文件夹 并排序 结果是一个路径列表 而不是文件夹名 列表
            subfolders_vi_gt = sorted(glob.glob(osp.join(self.vi_gt_root, '*'))) ## 遍历self.gt_root下所有子文件夹 并排序

        # name=opt['name'].lower()
        if opt['name'].lower() in ['ms3v', 'hdo']:
            for subfolder_ir_lq, subfolder_ir_gt, subfolder_vi_lq, subfolder_vi_gt in zip(subfolders_ir_lq, subfolders_ir_gt, subfolders_vi_lq, subfolders_vi_gt):
                # get frame list for lq and gt
                subfolder_name = osp.basename(subfolder_ir_lq)
                from natsort import natsorted
                ir_img_paths_lq = natsorted(list(scandir(subfolder_ir_lq, full_path=True)))[:205] ## 遍历subfolder_lq下所有文件或文件夹路径，返回一个路径列表
                ir_img_paths_gt = natsorted(list(scandir(subfolder_ir_gt, full_path=True)))[:205]

                vi_img_paths_lq = natsorted(list(scandir(subfolder_vi_lq, full_path=True)))[:205] ## 遍历subfolder_lq下所有文件或文件夹路径，返回一个路径列表
                vi_img_paths_gt = natsorted(list(scandir(subfolder_vi_gt, full_path=True)))[:205]

                max_idx = len(ir_img_paths_lq)
                assert max_idx == len(ir_img_paths_gt), (f'Different number of images in lq ({max_idx})'f' and gt folders ({len(ir_img_paths_gt)})') ## 确保lq和gt文件夹下图片数量相同

                self.data_info['ir_lq_path'].extend(ir_img_paths_lq) ## 将img_paths_lq中的路径添加到self.data_info['lq_path']中
                self.data_info['ir_gt_path'].extend(ir_img_paths_gt) ## 将img_paths_gt中的路径添加到self.data_info['gt_path']中
                self.data_info['vi_lq_path'].extend(vi_img_paths_lq) ## 将img_paths_lq中的路径添加到self.data_info['lq_path']中
                self.data_info['vi_gt_path'].extend(vi_img_paths_gt) ## 将img_paths_gt中的路径添加到self.data_info['gt_path']中
                
                self.data_info['folder'].extend([subfolder_name] * max_idx) ## 将subfolder_name添加到self.data_info['folder']中 每个lq_path 有对应的folder记录其所在文件夹的信息
                for i in range(max_idx):
                    self.data_info['idx'].append(f'{i}/{max_idx}') ## 将i/max_idx添加到self.data_info['idx']中 每个lq_path 有对应的idx记录其所在文件夹的信息
                border_l = [0] * max_idx ## 创建一个长度为max_idx的列表，所有元素都为0
                for i in range(self.opt['num_frame'] // 2): ## 记录是否是边界的几帧 如果是则需要 在训练或测试过程中跳过
                    border_l[i] = 1
                    border_l[max_idx - i - 1] = 1 ##  并将列表的前 num_frame // 2 和后 num_frame // 2 的元素设为 1
                self.data_info['border'].extend(border_l) ## 这个信息是用来干嘛的呢？

                # cache data or save the frame list
                # 如果启用了缓存机制，则将数据缓存；否则，仅保存帧的路径列表
                if self.cache_data:
                    # 记录日志，表示正在缓存该视频序列
                    logger.info(f'Cache {subfolder_name} for VideoTestDataset...')
                    # 将低质量图像序列缓存到字典 `self.imgs_lq` 中，键为 `subfolder_name`，值为通过 `read_img_seq` 读取的图像序列
                    self.imgs_ir_lq[subfolder_name] = read_img_seq(ir_img_paths_lq)
                    self.imgs_vi_lq[subfolder_name] = read_img_seq(vi_img_paths_lq)
                    # 将高质量图像序列缓存到字典 `self.imgs_gt` 中，键为 `subfolder_name`，值为通过 `read_img_seq` 读取的图像序列
                    self.imgs_ir_gt[subfolder_name] = read_img_seq(ir_img_paths_gt)
                    self.imgs_vi_gt[subfolder_name] = read_img_seq(vi_img_paths_gt)
                else:
                    self.imgs_ir_lq[subfolder_name] = ir_img_paths_lq
                    self.imgs_ir_gt[subfolder_name] = ir_img_paths_gt
                    self.imgs_vi_lq[subfolder_name] = vi_img_paths_lq
                    self.imgs_vi_gt[subfolder_name] = vi_img_paths_gt

        else:
            raise ValueError(f'Non-supported video test dataset: {type(opt["name"])}')
        	

    def __getitem__(self, index):
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info['border'][index]
        ir_lq_path = self.data_info['ir_lq_path'][index]
        vi_lq_path = self.data_info['vi_lq_path'][index]

        select_idx = generate_frame_indices(idx, max_idx, self.opt['num_frame'], padding=self.opt['padding'])

        if self.cache_data:
            ir_imgs_lq = self.imgs_ir_lq[folder].index_select(0, torch.LongTensor(select_idx))
            ir_img_gt = self.imgs_ir_gt[folder][idx]
            vi_imgs_lq = self.imgs_vi_lq[folder].index_select(0, torch.LongTensor(select_idx))
            vi_img_gt = self.imgs_vi_gt[folder][idx]
        else:
            ir_img_paths_lq = [self.imgs_ir_lq[folder][i] for i in select_idx]
            ir_imgs_lq = read_img_seq(ir_img_paths_lq)
            ir_img_gt = read_img_seq([self.imgs_ir_gt[folder][idx]]) ## GT还是一幅图像？
            ir_img_gt.squeeze_(0)
            vi_img_paths_lq = [self.imgs_vi_lq[folder][i] for i in select_idx]
            vi_imgs_lq = read_img_seq(vi_img_paths_lq)
            vi_mg_gt = read_img_seq([self.imgs_vi_gt[folder][idx]]) ## GT还是一幅图像？
            vi_img_gt.squeeze_(0)

        return {
            'lq_ir': ir_imgs_lq,  # (t, c, h, w)
            'gt_ir': ir_img_gt,  # (c, h, w)
            'lq_vi': vi_imgs_lq,  # (t, c, h, w)
            'gt_vi': vi_mg_gt,  # (c, h, w)
            'folder': folder,  # folder name
            'idx': self.data_info['idx'][index],  # e.g., 0/99
            'border': border,  # 1 for border, 0 for non-border
            'lq_ir_path': ir_lq_path,  # center frame
            'lq_vi_path': vi_lq_path  # center frame
        }

    def __len__(self):
        return len(self.data_info['ir_gt_path'])
    
class VideoFusionTestDataset(VideoTestDataset):
    
    def __init__(self, opt):
        super(VideoFusionTestDataset, self).__init__(opt)

        ori_folders = sorted(list(self.imgs_ir_lq.keys()))
        ori_num_frames_per_folder = {}
        ir_ori_imgs_lq_paths = {}
        ir_ori_imgs_gt_paths = {}
        vi_ori_imgs_lq_paths = {}
        vi_ori_imgs_gt_paths = {}
        now_idx = 0
        for folder in ori_folders:
            if self.cache_data:
                nf = self.imgs_ir_lq[folder].size()[0]
            else:
                nf = len(self.imgs_ir_lq[folder])
                # nf =  nf if nf<200 else 200
            ori_num_frames_per_folder[folder] = nf
            ir_ori_imgs_lq_paths[folder] = self.data_info['ir_lq_path'][now_idx:now_idx + nf]
            ir_ori_imgs_gt_paths[folder] = self.data_info['ir_gt_path'][now_idx:now_idx + nf]
            vi_ori_imgs_lq_paths[folder] = self.data_info['vi_lq_path'][now_idx:now_idx + nf]
            vi_ori_imgs_gt_paths[folder] = self.data_info['vi_gt_path'][now_idx:now_idx + nf]
            now_idx = now_idx + nf

        # Split Clips
        num_frame = self.opt['num_frame']
        num_overlap = self.opt['num_overlap']
        clip_data_info = {'ir_lq_path': [], 'ir_gt_path': [], 'vi_lq_path': [], 'vi_gt_path': [], 'folder': [], 'idx': [], 'border': []}
        clip_folders = []
        ir_clip_imgs_lq = {}
        ir_clip_imgs_gt = {}
        vi_clip_imgs_lq = {}
        vi_clip_imgs_gt = {}
        def natural_sort_key(s):
            import re
            return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

        for folder in ori_folders:
            num_all = ori_num_frames_per_folder[folder]
            self.imgs_ir_lq[folder] = sorted(self.imgs_ir_lq[folder], key=natural_sort_key)
            self.imgs_ir_gt[folder] = sorted(self.imgs_ir_gt[folder], key=natural_sort_key)
            self.imgs_vi_lq[folder] = sorted(self.imgs_vi_lq[folder], key=natural_sort_key)
            self.imgs_vi_gt[folder] = sorted(self.imgs_vi_gt[folder], key=natural_sort_key)
            ir_ori_imgs_lq_paths[folder] = sorted(ir_ori_imgs_lq_paths[folder], key=natural_sort_key)
            ir_ori_imgs_gt_paths[folder] = sorted(ir_ori_imgs_gt_paths[folder], key=natural_sort_key)
            vi_ori_imgs_lq_paths[folder] = sorted(vi_ori_imgs_lq_paths[folder], key=natural_sort_key)
            vi_ori_imgs_gt_paths[folder] = sorted(vi_ori_imgs_gt_paths[folder], key=natural_sort_key)

            for i in range((num_all - num_overlap) // (num_frame - num_overlap)):
                clip_folder = f'{folder}-{i:03d}'
                clip_folders.append(clip_folder)
                ir_clip_imgs_lq[clip_folder] = \
                    self.imgs_ir_lq[folder][i * (num_frame - num_overlap):i * (num_frame - num_overlap) + num_frame]
                ir_clip_imgs_gt[clip_folder] = \
                    self.imgs_ir_gt[folder][i * (num_frame - num_overlap):i * (num_frame - num_overlap) + num_frame]
                vi_clip_imgs_lq[clip_folder] = \
                    self.imgs_vi_lq[folder][i * (num_frame - num_overlap):i * (num_frame - num_overlap) + num_frame]
                vi_clip_imgs_gt[clip_folder] = \
                    self.imgs_vi_gt[folder][i * (num_frame - num_overlap):i * (num_frame - num_overlap) + num_frame]
                    
                clip_data_info['ir_lq_path'].extend(
                    ir_ori_imgs_lq_paths[folder][i * (num_frame - num_overlap):i * (num_frame - num_overlap) + num_frame])
                clip_data_info['ir_gt_path'].extend(
                    ir_ori_imgs_gt_paths[folder][i * (num_frame - num_overlap):i * (num_frame - num_overlap) + num_frame])
                clip_data_info['vi_lq_path'].extend(
                    vi_ori_imgs_lq_paths[folder][i * (num_frame - num_overlap):i * (num_frame - num_overlap) + num_frame])
                clip_data_info['vi_gt_path'].extend(
                    vi_ori_imgs_gt_paths[folder][i * (num_frame - num_overlap):i * (num_frame - num_overlap) + num_frame])
                
                clip_data_info['folder'].extend([clip_folder] * num_frame)
                for i in range(num_frame):
                    clip_data_info['idx'].append(f'{i}/{num_frame}')
                border_l = [0] * num_frame
                for i in range(num_frame // 2):
                    border_l[i] = 1
                    border_l[num_frame - i - 1] = 1
                clip_data_info['border'].extend(border_l)

            if (num_all - num_overlap) % (num_frame - num_overlap) != 0:
                clip_folder = f'{folder}-{((num_all - num_overlap) // (num_frame - num_overlap)):03d}'
                clip_folders.append(clip_folder)
                ir_clip_imgs_lq[clip_folder] = self.imgs_ir_lq[folder][-((num_all - num_overlap) % (num_frame - num_overlap)):]
                ir_clip_imgs_gt[clip_folder] = self.imgs_ir_gt[folder][-((num_all - num_overlap) % (num_frame - num_overlap)):]
                vi_clip_imgs_lq[clip_folder] = self.imgs_vi_lq[folder][-((num_all - num_overlap) % (num_frame - num_overlap)):]
                vi_clip_imgs_gt[clip_folder] = self.imgs_vi_gt[folder][-((num_all - num_overlap) % (num_frame - num_overlap)):]
                
                clip_data_info['ir_lq_path'].extend(ir_ori_imgs_lq_paths[folder][-num_frame:])
                clip_data_info['ir_gt_path'].extend(ir_ori_imgs_gt_paths[folder][-num_frame:])
                clip_data_info['vi_lq_path'].extend(vi_ori_imgs_lq_paths[folder][-num_frame:])
                clip_data_info['vi_gt_path'].extend(vi_ori_imgs_gt_paths[folder][-num_frame:])
                
                clip_data_info['folder'].extend([clip_folder] * num_frame)
                for i in range(num_frame):
                    clip_data_info['idx'].append(f'{i}/{num_frame}')
                border_l = [0] * num_frame
                for i in range(num_frame // 2):
                    border_l[i] = 1
                    border_l[num_frame - i - 1] = 1
                clip_data_info['border'].extend(border_l)

        self.folders = clip_folders
        self.imgs_ir_lq = ir_clip_imgs_lq
        self.imgs_ir_gt = ir_clip_imgs_gt
        self.imgs_vi_lq = vi_clip_imgs_lq
        self.imgs_vi_gt = vi_clip_imgs_gt
        self.data_info = clip_data_info

    def __getitem__(self, index):
        folder = self.folders[index]

        if self.cache_data:
            ir_imgs_lq = self.imgs_ir_lq[folder]
            ir_imgs_gt = self.imgs_ir_gt[folder]
            vi_imgs_lq = self.imgs_vi_lq[folder]
            vi_imgs_gt = self.imgs_vi_gt[folder]
        else:
            ir_img_paths_lq = self.imgs_ir_lq[folder]
            ir_img_paths_gt = self.imgs_ir_gt[folder]
            ir_imgs_lq = read_img_seq(ir_img_paths_lq)
            ir_imgs_gt = read_img_seq(ir_img_paths_gt)
            vi_img_paths_lq = self.imgs_vi_lq[folder]
            vi_img_paths_gt = self.imgs_vi_gt[folder]
            vi_imgs_lq = read_img_seq(vi_img_paths_lq)
            vi_imgs_gt = read_img_seq(vi_img_paths_gt)

        return {
            'lq_ir': ir_imgs_lq,
            'gt_ir': ir_imgs_gt,
            'lq_vi': vi_imgs_lq,
            'gt_vi': vi_imgs_gt,
            'folder': folder,
        }

    def __len__(self):
        return len(self.folders)
