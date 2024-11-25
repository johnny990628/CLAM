import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
import h5py

class Whole_Slide_Bag(Dataset):
	def __init__(self,
		file_path,
		img_transforms=None):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			roi_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.roi_transforms = img_transforms
		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['imgs']
			self.length = len(dset)

		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		with h5py.File(self.file_path, "r") as hdf5_file:
			dset = hdf5_file['imgs']
			for name, value in dset.attrs.items():
				print(name, value)

		print('transformations:', self.roi_transforms)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			img = hdf5_file['imgs'][idx]
			coord = hdf5_file['coords'][idx]
		
		img = Image.fromarray(img)
		img = self.roi_transforms(img)
		return {'img': img, 'coord': coord}

class Whole_Slide_Bag_FP(Dataset):
	def __init__(self,
		file_path,
		wsi,
		img_transforms=None):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			img_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.wsi = wsi
		self.roi_transforms = img_transforms

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['coords']
			self.patch_level = f['coords'].attrs['patch_level']
			self.patch_size = f['coords'].attrs['patch_size']
			self.scale_factor = f['coords'].attrs.get('scale_factor',1.0)
			self.length = len(dset)
			
		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['coords']
		for name, value in dset.attrs.items():
			print(name, value)
		print('\nfeature extraction settings')
		print('transformations: ', self.roi_transforms)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			coord = hdf5_file['coords'][idx]
		scaled_patch_size = int(float(self.patch_size)/float(self.scale_factor))
		img = self.wsi.read_region(coord, self.patch_level, (scaled_patch_size,scaled_patch_size)).convert('RGB')
		img = self.roi_transforms(img)
		return {'img': img, 'coord': coord}

class Dataset_All_Bags(Dataset):

	def __init__(self, csv_path):
		self.df = pd.read_csv(csv_path)
	
	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		return self.df['slide_id'][idx]

class Multi_Scale_Bag(Dataset):
    def __init__(self, file_path, wsi, low_mag, high_mag, img_transforms=None):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            wsi (OpenSlide): Whole slide image object.
            low_magnification (float): Low magnification level.
            high_magnification (float): High magnification level.
            img_transforms (callable, optional): Optional transform to be applied on images.
        """
        self.wsi = wsi
        self.roi_transforms = img_transforms
        self.file_path = file_path
        self.low_mag = low_mag
        self.high_mag = high_mag
        self.level_factor = int(high_mag / low_mag)

        with h5py.File(self.file_path, "r") as f:
            self.coords = f['coords'][:]
            self.patch_level = f['coords'].attrs['patch_level']
            self.patch_size = f['coords'].attrs['patch_size']
            self.length = len(self.coords)

        self.summary()

    def __len__(self):
        return self.length

    def summary(self):
        print('\nMultiResolutionDataset summary:')
        print(f'file_path: {self.file_path}')
        print(f'patch_level: {self.patch_level}')
        print(f'patch_size: {self.patch_size}')
        print(f'total patches: {self.length}')
        print(f'low magnification: {self.low_mag}')
        print(f'high magnification: {self.high_mag}')
        print(f'level factor: {self.level_factor}')
        print(f'transformations: {self.roi_transforms}')

    def __getitem__(self, idx):
        coord = self.coords[idx]
        x, y = coord

        # 低倍率图像
        low_img = self.wsi.read_region(
            (x, y), 
            self.patch_level, 
            (self.patch_size, self.patch_size)
        ).convert('RGB')
        
        # 高倍率图像
        high_imgs = []
        for i in range(self.level_factor):
            for j in range(self.level_factor):
                high_x = x * self.level_factor + i * self.patch_size
                high_y = y * self.level_factor + j * self.patch_size
                high_img = self.wsi.read_region(
                    (high_x, high_y), 
                    self.wsi.get_best_level_for_downsample(self.high_mag), 
                    (self.patch_size, self.patch_size)
                ).convert('RGB')
                if self.roi_transforms:
                    high_img = self.roi_transforms(high_img)
                high_imgs.append(high_img)

        if self.roi_transforms:
            low_img = self.roi_transforms(low_img)

        return {'low': low_img, 'high': high_imgs, 'coord': coord}


