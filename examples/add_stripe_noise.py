import cv2
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
# from moviepy.editor import VideoFileClip
from PIL import Image
from torch.utils.data.dataset import Dataset
from natsort import natsorted
# from utils import open_sequence
from tqdm import tqdm
import os
from random import random
import glob
import argparse
IMAGETYPES = ('*.bmp', '*.png', '*.jpg', '*.jpeg', '*.tif')

# 假设 genBiasField_ori 函数已定义
# 模拟 genBiasField_ori 生成偏置场的函数
def genBiasField_ori(image_size):
    return np.random.randint(0, 255, (image_size[0], image_size[1]), dtype=np.uint8)

def get_imagenames(seq_dir, pattern=None):
	files = []
	for typ in IMAGETYPES:
		files.extend(glob.glob(os.path.join(seq_dir, typ)))
	if not pattern is None:
		ffiltered = [f for f in files if pattern in os.path.split(f)[-1]]
		files = ffiltered
		del ffiltered

	files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
	return files

def normalize(data):
	data = np.float32(data)
	data = (data-data.min())/(data.max()-data.min())
	return data

def open_image(fpath, gray_mode, expand_if_needed=False, expand_axis0=True, normalize_data=True):
	if not gray_mode:
		img = cv2.imread(fpath)
		img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
	else:
		img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)

	if expand_axis0:
		img = np.expand_dims(img, 0)

	expanded_h = False
	expanded_w = False
	sh_im = img.shape
	if expand_if_needed:
		if sh_im[-2]%2 == 1:
			expanded_h = True
			if expand_axis0:
				img = np.concatenate((img, \
									  img[:, :, -1, :][:, :, np.newaxis, :]), axis=2)
			else:
				img = np.concatenate((img, \
									  img[:, -1, :][:, np.newaxis, :]), axis=1)


		if sh_im[-1]%2 == 1:
			expanded_w = True
			if expand_axis0:
				img = np.concatenate((img, \
									  img[:, :, :, -1][:, :, :, np.newaxis]), axis=3)
			else:
				img = np.concatenate((img, \
									  img[:, :, -1][:, :, np.newaxis]), axis=2)

	if normalize_data:
		img = normalize(img)
	return img, expanded_h, expanded_w

def open_sequence(seq_dir, gray_mode, expand_if_needed=False, max_num_fr=100, disp_status=True):
	files = natsorted(get_imagenames(seq_dir))
	# print(files)
	files_num = files.__len__()
	seq_list = []
	if disp_status == True:
		print("\tOpen sequence in folder: ", seq_dir)
	for fpath in files:
		img, expanded_h, expanded_w = open_image(fpath,
												 gray_mode=gray_mode,
												 expand_if_needed=expand_if_needed,
												 expand_axis0=False)
		seq_list.append(img)
	seq = np.stack(seq_list, axis=0)
	return seq, expanded_h, expanded_w, files

class ADDNOISE_class():
	def __init__(self, image_size):
		super(ADDNOISE_class, self).__init__()
		self.bias_field_ori = genBiasField_ori(image_size)
		self.SIZE = image_size

	def addnoise(self, x):

		device = x.device
		t = x		
		t = t.unsqueeze(dim=0)
		# print(t.shape)
		N, F, C, H, W = t.shape
		# stdN_G = torch.FloatTensor(1).uniform_(9, 15)
		# beta = torch.FloatTensor(1).uniform_(5, 15)
	
		stdn = torch.randint(9, 15, (N, 1, 1, 1, 1), dtype=torch.float32).to(device) / 255. # Gaussian
		stdncol_spatial = torch.randint(5, 16, (N, 1, 1, 1, 1), dtype=torch.float32).to(device) / 255.0 # 条纹噪声

		noise_gaussian = torch.zeros(N, F, 1, H, W).to(device)
		noise_gaussian = torch.normal(mean=noise_gaussian, std=stdn.expand_as(noise_gaussian))
		noise_gaussian = noise_gaussian.repeat(1, 1, C, 1, 1)
		
		col_noise_spatial = torch.zeros([N, 1, 1, 1, W]).to(device)
		col_noise_spatial = torch.normal(mean=col_noise_spatial, std=stdncol_spatial.expand_as(col_noise_spatial))
		col_noise_spatial = col_noise_spatial.repeat(1, F, C, H, 1)

		# imgn = t + row_noise_spatial + col_noise_spatial + noise_bias_field + noise_gaussian + line_noise_time
		imgn = t +  col_noise_spatial +  noise_gaussian
		imgn = imgn.clamp(0., 1.)
		imgn = imgn.reshape(F, C, H, W).to(device)
		return imgn

	def add_biasfield(self, patchsize=96, intensity=1):
		device = intensity.device
		N = intensity.shape[0]
		if patchsize != None:
			H = patchsize
			W = patchsize
		else:
			H = self.SIZE[0]
			W = self.SIZE[1]
		Bias_field = torch.zeros(N, 1, 1, H, W)

		for i in range(N):
			B_boosted = Image.fromarray(self.bias_field_ori)
			if patchsize != None:
				nw = random.randint(0, B_boosted.size[1] - patchsize)  ##裁剪图像在原图像中的坐标
				nh = random.randint(0, B_boosted.size[0] - patchsize)
				B_boosted = B_boosted.crop((nh, nw, nh + patchsize, nw + patchsize))

			to_tensor = transforms.ToTensor()
			B_boosted = to_tensor(B_boosted).unsqueeze(dim=0).unsqueeze(dim=0)
			Bias_field[i, 0, 0, :, :] = B_boosted

		Bias_field = Bias_field.to(device)
		Bias_field = Bias_field * intensity - intensity / 2

		return Bias_field
	
NUMFRXSEQ_VAL = 15	# number of frames of each sequence to include in validation dataset
VALSEQPATT = '*' # pattern for name of validation sequence

class ValDataset(Dataset):
	"""Validation dataset. Loads all the images in the dataset folder on memory.
	"""
	def __init__(self, valsetdir=None, gray_mode=True, num_input_frames=15, disp_status=True):
		self.gray_mode = gray_mode

		# Look for subdirs with individual sequences
		seqs_dirs = sorted(glob.glob(os.path.join(valsetdir, VALSEQPATT)))
		print(seqs_dirs)
		# open individual sequences and append them to the sequence list
		sequences = []
		seq_dirs_name = []
		seq_files = []
		for seq_dir in seqs_dirs:
			if not os.listdir(seq_dir):
				continue
			seq, _, _, file_paths = open_sequence(seq_dir, gray_mode, expand_if_needed=False, \
							 max_num_fr=num_input_frames,disp_status=disp_status)
			# seq is [num_frames, C, H, W]
			if gray_mode == True:
				seq = np.expand_dims(seq, axis=1)
			seq_files.append([os.path.basename(x) for x in file_paths])
			sequences.append(seq)
			name = seq_dir.split('/')
			seq_dirs_name.append(name[-1])

		self.sequences = sequences
		self.seq_dirs_name = seq_dirs_name
		self.seq_files = seq_files


	def __getitem__(self, index):
		return torch.from_numpy(self.sequences[index]), self.seq_dirs_name[index], self.seq_files[index]

	def __len__(self):
		return len(self.sequences)

class TestDataset(Dataset):
	"""Validation dataset. Loads all the images in the dataset folder on memory.
	"""
	def __init__(self, testsetdir=None, gray_mode=False, num_input_frames=NUMFRXSEQ_VAL):
		self.gray_mode = gray_mode

		# Look for subdirs with individual sequences
		seqs_dirs = sorted(glob.glob(os.path.join(testsetdir, VALSEQPATT)))

		# open individual sequences and append them to the sequence list
		sequences = []
		seq_dirs_name = []
		for seq_dir in seqs_dirs:
			seq, _, _, file_paths = open_sequence(seq_dir, gray_mode, expand_if_needed=False, \
							 max_num_fr=num_input_frames)
			# seq is [num_frames, C, H, W]
			if gray_mode == True:
				seq = np.expand_dims(seq, axis=1)

			sequences.append(seq)
			name = seq_dir.split('/')
			seq_dirs_name.append(name[-1])

		self.sequences = sequences
		self.seq_dirs_name = seq_dirs_name

	def __getitem__(self, index):
		return torch.from_numpy(self.sequences[index]), self.seq_dirs_name[index]

	def __len__(self):
		return len(self.sequences)

			
def test_MDIVDnet(input_folder, save_folder):
	# set seed
	dataset_test = ValDataset(valsetdir=input_folder, num_input_frames=150, disp_status=False)

	device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu') 
	device = torch.device('cpu') 
	os.makedirs(save_folder, exist_ok=True)
	frame_num_count = 0
	for seq, dir_name, file_names in dataset_test:
		dir_path = os.path.join(save_folder, dir_name)
		os.makedirs(dir_path, exist_ok=True)
		# print(dir_name)
		# print(seq.shape)
        # seq = torch.from_numpy(seq).to(device)
		seq = seq.to(device)
		[F_num, C, H, W] = seq.shape
		if C == 1:
			seq = seq.repeat(1, 3, 1, 1)
		# print("Shape of seq: ", seq.shape)		
		[F_num, C, H, W] = seq.shape
		frame_num_count += F_num

        # define noise class
		noise = ADDNOISE_class(image_size=(H, W))
		seqn = noise.addnoise(seq)
		# print("Shape of seqn: ", seqn.shape)
		 # 转换为CPU，转换为 numpy 格式
		seqn = seqn.cpu().numpy()
		file_bar = tqdm(file_names)
		for i, (frame, file_name) in enumerate(zip(seqn, file_bar)):
			if C == 1:
				frame = frame[0]  # 单通道
				frame = (frame * 255).astype('uint8')
			else:
				frame = (frame.transpose(1, 2, 0) * 255).astype('uint8')  # C,H,W -> H,W,C
			image_path = os.path.join(dir_path, file_name)
			cv2.imwrite(image_path, frame)  # 保存图片
			file_bar.set_description(f"Processing frame {dir_name} | {i+1} |  {file_name}")
			# print(f"Saved frame {i} to {image_path}")
        # # 确保通道数匹配
		# if C == 1:
		# 	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		# else:
		# 	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # # 创建视频写入器
		# video_path = os.path.join(output_dir, f"{name}.mp4")
		# out = cv2.VideoWriter(video_path, fourcc, 20.0, (W, H))
		# for i in range(F_num):
		# 	frame = seqn[i]  # i表示第 i 帧
		# 	if C == 1:
		# 		frame = frame[0]  # 如果是单通道，取出第一个通道
		# 		frame = (frame * 255).astype('uint8')  # 转换为 uint8
		# 	else:
		# 		frame = (frame.transpose(1, 2, 0) * 255).astype('uint8')  # C,H,W -> H,W,C
		# 	out.write(frame)  # 写入视频帧
		# out.release()  # 释放视频写入器
		# print(f"Saved video: {video_path}")
if __name__ == "__main__":
	# input_folder = "/home/whu/HDD_16T/timer/gmq/video/ours/HDO2/test/ir"
	# output_folder = "/home/whu/HDD_16T/timer/gmq/video/ours/HDO2/test/ir_noise"
	# input_folder = "/home/whu/HDD_16T/timer/gmq/video/TemCoCo/test_ST_select/IR/"
	# output_folder = "/home/whu/HDD_16T/timer/gmq/video/TemCoCo/test_ST_select/IR_noise"
	# input_dir = '/data/timer/VideoFusion/datasets/Video/train/infrared'
	# save_dir = '/data/timer/VideoFusion/datasets/Video/train/infrared_noise3'
	input_folder = "/media/wit/whu/timer/gmq/video/TemCoCo/MS3V/all_with_LE/train/IR/"
	output_folder = "/media/wit/whu/timer/gmq/video/TemCoCo/MS3V/all_with_LE/train/IR_noise2/"
	test_MDIVDnet(input_folder, output_folder)
