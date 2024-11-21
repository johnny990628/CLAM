import time
import os
import argparse
import pdb
from functools import partial

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from PIL import Image
import h5py
import openslide
from tqdm import tqdm

import numpy as np

from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP, Multi_Scale_Bag
from models import get_encoder

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def compute_w_loader(output_path, loader, model, verbose = 0):
	"""
	args:
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		verbose: level of feedback
	"""
	if verbose > 0:
		print(f'processing a total of {len(loader)} batches'.format(len(loader)))

	mode = 'w'
	for count, data in enumerate(tqdm(loader)):
		with torch.inference_mode():	
			batch = data['img']
			coords = data['coord'].numpy().astype(np.int32)
			batch = batch.to(device, non_blocking=True)
			
			features = model(batch)
			features = features.cpu().numpy().astype(np.float32)

			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	
	return output_path

def compute_multi_scale_features(output_path, loader, model, fusion_method='cat'):
	mode = 'w'
	for count, data in enumerate(tqdm(loader)):
		with torch.inference_mode():
			low_batch = data['low'].to(device, non_blocking=True)
			high_batches = [high.to(device, non_blocking=True) for high in data['high']]
			coords = data['coord'].numpy().astype(np.int32)
			# 提取低倍率特徵
			features_low = model(low_batch).cpu().numpy()
			# 建立一個列表來存放所有組合後的高倍率特徵
			combined_features_all = []
			for i, high_batch in enumerate(high_batches):
				features_high = model(high_batch).cpu().numpy()
				# 根據融合方式組合低倍率和高倍率特徵
				if fusion_method == 'fusion':
					combined_features = features_high + 0.25 * features_low
				else:  # 'cat'
					combined_features = np.concatenate((features_high, features_low), axis=-1)
				combined_features_all.append(combined_features)
			features = np.concatenate(combined_features_all, axis=0).astype(np.float32)
			# 將所有組合後的特徵堆疊並儲存為單一的 'features'
			asset_dict = {
				'features': features,
				'coords': coords
			}

			# 儲存所有組合後的特徵到 HDF5 檔案
			save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
			mode = 'a'  # 只用在處理多個 batch 時

	return output_path




parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--model_name', type=str, default='resnet50_trunc', choices=['resnet50_trunc', 'uni_v1', 'conch_v1', 'chief', 'gigapath'])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=224)
parser.add_argument('--magnification', type=str, default='single', choices=['single', 'low', 'high', 'tree'], help='Magnification level(s) to process')
parser.add_argument('--low_mag', type=str, default='5x')
parser.add_argument('--high_mag', type=str, default='10x')
parser.add_argument('--tree_fusion', type=str, default='cat', choices=['cat', 'fusion'], help='Fusion method for tree magnification')

args = parser.parse_args()


if __name__ == '__main__':
	print('initializing dataset')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError

	bags_dataset = Dataset_All_Bags(csv_path)
	
	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

	model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)

	_ = model.eval()
	model = model.to(device)
	total = len(bags_dataset)

	loader_kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}

	for bag_candidate_idx in tqdm(range(total)):
		slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
		bag_name = slide_id+'.h5'
		h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
		slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(slide_id)

		if not args.no_auto_skip and slide_id+'.pt' in dest_files:
			print('skipped {}'.format(slide_id))
			continue 

		output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
		time_start = time.time()
		wsi = openslide.open_slide(slide_file_path)
		if args.magnification == 'tree':
			low_mag = int(args.low_mag.replace('x',''))
			high_mag = int(args.high_mag.replace('x',''))
			dataset = Multi_Scale_Bag(file_path=h5_file_path, wsi=wsi, low_mag=low_mag, high_mag=high_mag, img_transforms=img_transforms)
			loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)
			output_file_path = compute_multi_scale_features(output_path, loader, model, fusion_method=args.tree_fusion)
		else:
			dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, wsi=wsi, img_transforms=img_transforms)
			loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)
			output_file_path = compute_w_loader(output_path, loader=loader, model=model, verbose=1)
			
		with h5py.File(output_file_path, "r") as file:
			features = file['features'][:]
			print('features size: ', features.shape)
			print('coordinates size: ', file['coords'].shape)

		features = torch.from_numpy(features)
		bag_base, _ = os.path.splitext(bag_name)
		torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))
			
		time_elapsed = time.time() - time_start
		print('\ncomputing features for {} took {} s'.format(slide_id, time_elapsed))

		



