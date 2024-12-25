import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle
from scipy import stats

from torch.utils.data import Dataset
import h5py

from utils.utils import generate_split, nth

def save_splits(split_datasets, column_keys, filename, boolean_style=False):
	splits = [split_datasets[i].slide_data['slide_id'] for i in range(len(split_datasets))]
	
	if not boolean_style:
		df = pd.concat(splits, ignore_index=True, axis=1)
		df.columns = column_keys
	else:
		df = pd.concat(splits, ignore_index=True, axis=0)
		index = df.values.tolist()
		one_hot = np.eye(len(split_datasets)).astype(bool)
		bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
		df = pd.DataFrame(bool_array, index=index, columns=column_keys)

	df.to_csv(filename, index=False)
	print(f"Splits saved to {filename}")


def save_splits_survival(split_datasets, column_keys, filename, boolean_style=False):
	"""
	Modified save_splits function for survival data
	Args:
		split_datasets (list): list of datasets for each split
		column_keys (list): list of keys for each split
		filename (str): path to save the splits
		boolean_style (bool): whether to save as boolean mask
	"""
	splits = [split_datasets[i].slide_data['slide_id'] for i in range(len(split_datasets))]
	
	if not boolean_style:
		df = pd.concat(splits, ignore_index=True, axis=1)
		df.columns = column_keys
	else:
		df = pd.concat(splits, ignore_index=True, axis=0)
		index = df.values.tolist()
		one_hot = np.eye(len(split_datasets)).astype(bool)
		bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
		df = pd.DataFrame(bool_array, index=index, columns=column_keys)

	df.to_csv(filename, index=False)
	print(f"Survival splits saved to {filename}")


class Generic_WSI_Classification_Dataset(Dataset):
	def __init__(self,
		csv_path = 'dataset_csv/ccrcc_clean.csv',
		shuffle = False, 
		seed = 7, 
		print_info = True,
		label_dict = {},
		filter_dict = {},
		ignore=[],
		patient_strat=False,
		label_col = None,
		patient_voting = 'max',
		):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			shuffle (boolean): Whether to shuffle
			seed (int): random seed for shuffling the data
			print_info (boolean): Whether to print a summary of the dataset
			label_dict (dict): Dictionary with key, value pairs for converting str labels to int
			ignore (list): List containing class labels to ignore
		"""
		self.label_dict = label_dict
		self.num_classes = len(set(self.label_dict.values()))
		self.seed = seed
		self.print_info = print_info
		self.patient_strat = patient_strat
		self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
		self.data_dir = None
		if not label_col:
			label_col = 'label'
		self.label_col = label_col

		slide_data = pd.read_csv(csv_path)
		slide_data = self.filter_df(slide_data, filter_dict)
		slide_data = self.df_prep(slide_data, self.label_dict, ignore, self.label_col)

		###shuffle data
		if shuffle:
			np.random.seed(seed)
			np.random.shuffle(slide_data)

		self.slide_data = slide_data

		self.patient_data_prep(patient_voting)
		self.cls_ids_prep()

		if print_info:
			self.summarize()

	def cls_ids_prep(self):
		# store ids corresponding each class at the patient or case level
		self.patient_cls_ids = [[] for i in range(self.num_classes)]		
		for i in range(self.num_classes):
			self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

		# store ids corresponding each class at the slide level
		self.slide_cls_ids = [[] for i in range(self.num_classes)]
		for i in range(self.num_classes):
			self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

	def patient_data_prep(self, patient_voting='max'):
		patients = np.unique(np.array(self.slide_data['case_id'])) # get unique patients
		patient_labels = []
		
		for p in patients:
			locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
			assert len(locations) > 0
			label = self.slide_data['label'][locations].values
			if patient_voting == 'max':
				label = label.max() # get patient label (MIL convention)
			elif patient_voting == 'maj':
				label = stats.mode(label)[0]
			else:
				raise NotImplementedError
			patient_labels.append(label)
		
		self.patient_data = {'case_id':patients, 'label':np.array(patient_labels)}

	@staticmethod
	def df_prep(data, label_dict, ignore, label_col):
		if label_col != 'label':
			data['label'] = data[label_col].copy()

		mask = data['label'].isin(ignore)
		data = data[~mask]
		data.reset_index(drop=True, inplace=True)
		for i in data.index:
			key = data.loc[i, 'label']
			data.at[i, 'label'] = label_dict[key]

		return data

	def filter_df(self, df, filter_dict={}):
		if len(filter_dict) > 0:
			filter_mask = np.full(len(df), True, bool)
			# assert 'label' not in filter_dict.keys()
			for key, val in filter_dict.items():
				mask = df[key].isin(val)
				filter_mask = np.logical_and(filter_mask, mask)
			df = df[filter_mask]
		return df

	def __len__(self):
		if self.patient_strat:
			return len(self.patient_data['case_id'])

		else:
			return len(self.slide_data)

	def summarize(self):
		print("label column: {}".format(self.label_col))
		print("label dictionary: {}".format(self.label_dict))
		print("number of classes: {}".format(self.num_classes))
		print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))
		for i in range(self.num_classes):
			print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
			print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))

	def create_splits(self, k = 3, val_num = (25, 25), test_num = (40, 40), label_frac = 1.0, custom_test_ids = None):
		"""
		val_num (tuple): number of samples for validation for each class (non_event, event)
		test_num (tuple): number of samples for testing for each class (non_event, event)
		"""
		settings = {
			'n_splits': k,
			'val_num': val_num,  # Now expecting a tuple
			'test_num': test_num,  # Now expecting a tuple
			'label_frac': label_frac,
			'seed': self.seed,
			'custom_test_ids': custom_test_ids
		}

		if self.patient_strat:
			# Create pseudo classes based on event status for stratification
			event_ids = np.where(self.patient_data['event'] == 1)[0]
			non_event_ids = np.where(self.patient_data['event'] == 0)[0]
			cls_ids = [non_event_ids, event_ids]
			settings.update({
				'cls_ids': cls_ids,
				'samples': len(self.patient_data['case_id'])
			})
		else:
			event_ids = np.where(self.slide_data[self.event_col] == 1)[0]
			non_event_ids = np.where(self.slide_data[self.event_col] == 0)[0]
			cls_ids = [non_event_ids, event_ids]
			settings.update({
				'cls_ids': cls_ids,
				'samples': len(self.slide_data)
			})

		self.split_gen = generate_split(**settings)

	def set_splits(self,start_from=None):
		if start_from:
			ids = nth(self.split_gen, start_from)

		else:
			ids = next(self.split_gen)

		if self.patient_strat:
			slide_ids = [[] for i in range(len(ids))] 

			for split in range(len(ids)): 
				for idx in ids[split]:
					case_id = self.patient_data['case_id'][idx]
					slide_indices = self.slide_data[self.slide_data['case_id'] == case_id].index.tolist()
					slide_ids[split].extend(slide_indices)

			self.train_ids, self.val_ids, self.test_ids = slide_ids[0], slide_ids[1], slide_ids[2]

		else:
			self.train_ids, self.val_ids, self.test_ids = ids

	def get_split_from_df(self, all_splits, split_key='train'):
		split = all_splits[split_key]
		split = split.dropna().reset_index(drop=True)

		if len(split) > 0:
			mask = self.slide_data['slide_id'].isin(split.tolist())
			df_slice = self.slide_data[mask].reset_index(drop=True)
			split = Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes)
		else:
			split = None
		
		return split

	def get_merged_split_from_df(self, all_splits, split_keys=['train']):
		merged_split = []
		for split_key in split_keys:
			split = all_splits[split_key]
			split = split.dropna().reset_index(drop=True).tolist()
			merged_split.extend(split)

		if len(split) > 0:
			mask = self.slide_data['slide_id'].isin(merged_split)
			df_slice = self.slide_data[mask].reset_index(drop=True)
			split = Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes)
		else:
			split = None
		
		return split


	def return_splits(self, from_id=True, csv_path=None):
		if from_id:
			if len(self.train_ids) > 0:
				train_data = self.slide_data.loc[self.train_ids].reset_index(drop=True)
				train_split = Generic_Split(train_data, data_dir=self.data_dir, num_classes=self.num_classes)

			else:
				train_split = None
			
			if len(self.val_ids) > 0:
				val_data = self.slide_data.loc[self.val_ids].reset_index(drop=True)
				val_split = Generic_Split(val_data, data_dir=self.data_dir, num_classes=self.num_classes)

			else:
				val_split = None
			
			if len(self.test_ids) > 0:
				test_data = self.slide_data.loc[self.test_ids].reset_index(drop=True)
				test_split = Generic_Split(test_data, data_dir=self.data_dir, num_classes=self.num_classes)
			
			else:
				test_split = None
			
		else:
			assert csv_path 
			all_splits = pd.read_csv(csv_path, dtype=self.slide_data['slide_id'].dtype)  # Without "dtype=self.slide_data['slide_id'].dtype", read_csv() will convert all-number columns to a numerical type. Even if we convert numerical columns back to objects later, we may lose zero-padding in the process; the columns must be correctly read in from the get-go. When we compare the individual train/val/test columns to self.slide_data['slide_id'] in the get_split_from_df() method, we cannot compare objects (strings) to numbers or even to incorrectly zero-padded objects/strings. An example of this breaking is shown in https://github.com/andrew-weisman/clam_analysis/tree/main/datatype_comparison_bug-2021-12-01.
			train_split = self.get_split_from_df(all_splits, 'train')
			val_split = self.get_split_from_df(all_splits, 'val')
			test_split = self.get_split_from_df(all_splits, 'test')
			
		return train_split, val_split, test_split

	def get_list(self, ids):
		return self.slide_data['slide_id'][ids]

	def getlabel(self, ids):
		return self.slide_data['label'][ids]

	def __getitem__(self, idx):
		return None

	def test_split_gen(self, return_descriptor=False):

		if return_descriptor:
			index = [list(self.label_dict.keys())[list(self.label_dict.values()).index(i)] for i in range(self.num_classes)]
			columns = ['train', 'val', 'test']
			df = pd.DataFrame(np.full((len(index), len(columns)), 0, dtype=np.int32), index= index,
							columns= columns)

		count = len(self.train_ids)
		print('\nnumber of training samples: {}'.format(count))
		labels = self.getlabel(self.train_ids)
		unique, counts = np.unique(labels, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'train'] = counts[u]
		
		count = len(self.val_ids)
		print('\nnumber of val samples: {}'.format(count))
		labels = self.getlabel(self.val_ids)
		unique, counts = np.unique(labels, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'val'] = counts[u]

		count = len(self.test_ids)
		print('\nnumber of test samples: {}'.format(count))
		labels = self.getlabel(self.test_ids)
		unique, counts = np.unique(labels, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'test'] = counts[u]

		assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0
		assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0
		assert len(np.intersect1d(self.val_ids, self.test_ids)) == 0

		if return_descriptor:
			return df

	def save_split(self, filename):
		train_split = self.get_list(self.train_ids)
		val_split = self.get_list(self.val_ids)
		test_split = self.get_list(self.test_ids)
		df_tr = pd.DataFrame({'train': train_split})
		df_v = pd.DataFrame({'val': val_split})
		df_t = pd.DataFrame({'test': test_split})
		df = pd.concat([df_tr, df_v, df_t], axis=1) 
		df.to_csv(filename, index = False)

	


class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset):
	def __init__(self,
		data_dir, 
		**kwargs):
	
		super(Generic_MIL_Dataset, self).__init__(**kwargs)
		self.data_dir = data_dir
		self.use_h5 = False

	def load_from_h5(self, toggle):
		self.use_h5 = toggle

	def __getitem__(self, idx):
		slide_id = self.slide_data['slide_id'][idx]
		label = self.slide_data['label'][idx]
		if type(self.data_dir) == dict:
			source = self.slide_data['source'][idx]
			data_dir = self.data_dir[source]
		else:
			data_dir = self.data_dir

		if not self.use_h5:
			if self.data_dir:
				full_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id))
				features = torch.load(full_path, weights_only=True)
				return features, label
			else:
				return slide_id, label

		else:
			full_path = os.path.join(data_dir,'h5_files','{}.h5'.format(slide_id))
			with h5py.File(full_path,'r') as hdf5_file:
				features = hdf5_file['features'][:]
				coords = hdf5_file['coords'][:]

			features = torch.from_numpy(features)
			return features, label, coords
	def getlabel(self, idx):
		"""
		Override getlabel to return event status for survival data
		"""
		return self.slide_data[self.event_col].iloc[idx]



class Generic_WSI_Survival_Dataset(Generic_WSI_Classification_Dataset):
	def __init__(self,
		csv_path = 'dataset_csv/survival_data.csv',
		shuffle = False, 
		seed = 7, 
		print_info = True,
		time_col = 'time',
		event_col = 'event',
		filter_dict = {},
		ignore=[],
		patient_strat=True,
		):
		"""
		Args:
			csv_path (string): Path to the csv file with survival data
			shuffle (boolean): Whether to shuffle
			seed (int): random seed for shuffling the data
			print_info (boolean): Whether to print a summary of the dataset
			time_col (string): Name of the survival time column
			event_col (string): Name of the event indicator column
			filter_dict (dict): Dictionary with key, value pairs for filtering the data
			ignore (list): List containing values to ignore
			patient_strat (boolean): Whether to stratify by patient
		"""
		self.time_col = time_col
		self.event_col = event_col
		self.seed = seed
		self.print_info = print_info
		self.patient_strat = patient_strat
		self.train_ids, self.val_ids, self.test_ids = (None, None, None)
		self.data_dir = None
		self.num_classes = 2

		self.patient_cls_ids = [[] for i in range(self.num_classes)]
		self.slide_cls_ids = [[] for i in range(self.num_classes)]


		slide_data = pd.read_csv(csv_path)
		slide_data = self.filter_df(slide_data, filter_dict)
		
		if shuffle:
			np.random.seed(seed)
			np.random.shuffle(slide_data)

		self.slide_data = slide_data
		self.patient_data_prep_survival()
		self.cls_ids_prep()
		
		if print_info:
			self.summarize_survival()

	def cls_ids_prep(self):
		# Store ids corresponding to each class (event vs non-event) at the patient level
		self.patient_cls_ids[0] = np.where(self.patient_data['event'] == 0)[0]  # non-event
		self.patient_cls_ids[1] = np.where(self.patient_data['event'] == 1)[0]  # event

		# Store ids corresponding to each class at the slide level
		self.slide_cls_ids[0] = np.where(self.slide_data[self.event_col] == 0)[0]  # non-event
		self.slide_cls_ids[1] = np.where(self.slide_data[self.event_col] == 1)[0]  # event

	def patient_data_prep_survival(self):
		patients = np.unique(np.array(self.slide_data['case_id']))
		patient_times = []
		patient_events = []
		
		for p in patients:
			locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
			assert len(locations) > 0
			
			# For patients with multiple slides, take the first record
			time = self.slide_data[self.time_col][locations[0]]
			event = self.slide_data[self.event_col][locations[0]]
			
			patient_times.append(time)
			patient_events.append(event)
		
		self.patient_data = {
			'case_id': patients, 
			'survival_time': np.array(patient_times),
			'event': np.array(patient_events)
		}

	def summarize_survival(self):
		print("Survival Analysis Dataset Summary:")
		print(f"Time column: {self.time_col}")
		print(f"Event column: {self.event_col}")
		print("\nSlide-level statistics:")
		print(f"Total number of slides: {len(self.slide_data)}")
		print(f"Number of events: {self.slide_data[self.event_col].sum()}")
		print(f"Median survival time: {self.slide_data[self.time_col].median():.2f}")
		
		print("\nPatient-level statistics:")
		print(f"Total number of patients: {len(self.patient_data['case_id'])}")
		print(f"Number of events: {sum(self.patient_data['event'])}")
		print(f"Median survival time: {np.median(self.patient_data['survival_time']):.2f}")
		
		print("\nClass distribution:")
		for i in range(self.num_classes):
			print(f'Patient-LVL; Number of samples in class {i}: {len(self.patient_cls_ids[i])}')
			print(f'Slide-LVL; Number of samples in class {i}: {len(self.slide_cls_ids[i])}')

	def test_split_gen(self, return_descriptor=False):
		"""Override the test_split_gen method to use survival-specific labels"""
		if return_descriptor:
			index = ['non_event', 'event']  # Use survival-specific labels
			columns = ['train', 'val', 'test']
			df = pd.DataFrame(np.full((len(index), len(columns)), 0, dtype=np.int32), 
							index=index,
							columns=columns)

		count = len(self.train_ids)
		print('\nnumber of training samples: {}'.format(count))
		events = self.slide_data[self.event_col].iloc[self.train_ids]
		unique, counts = np.unique(events, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format('event' if unique[u] == 1 else 'non_event', counts[u]))
			if return_descriptor:
				df.loc['event' if unique[u] == 1 else 'non_event', 'train'] = counts[u]
		
		count = len(self.val_ids)
		print('\nnumber of val samples: {}'.format(count))
		events = self.slide_data[self.event_col].iloc[self.val_ids]
		unique, counts = np.unique(events, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format('event' if unique[u] == 1 else 'non_event', counts[u]))
			if return_descriptor:
				df.loc['event' if unique[u] == 1 else 'non_event', 'val'] = counts[u]

		count = len(self.test_ids)
		print('\nnumber of test samples: {}'.format(count))
		events = self.slide_data[self.event_col].iloc[self.test_ids]
		unique, counts = np.unique(events, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format('event' if unique[u] == 1 else 'non_event', counts[u]))
			if return_descriptor:
				df.loc['event' if unique[u] == 1 else 'non_event', 'test'] = counts[u]

		assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0
		assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0
		assert len(np.intersect1d(self.val_ids, self.test_ids)) == 0

		if return_descriptor:
			return df

	def create_splits(self, k = 3, val_num = 25, test_num = 40, label_frac = 1.0, custom_test_ids = None):
		# For survival analysis, we'll stratify based on event status
		settings = {
			'n_splits': k,
			'val_num': val_num,
			'test_num': test_num,
			'label_frac': label_frac,
			'seed': self.seed,
			'custom_test_ids': custom_test_ids
		}

		if self.patient_strat:
			# Create pseudo classes based on event status for stratification
			event_ids = np.where(self.patient_data['event'] == 1)[0]
			non_event_ids = np.where(self.patient_data['event'] == 0)[0]
			cls_ids = [non_event_ids, event_ids]
			settings.update({
				'cls_ids': cls_ids,
				'samples': len(self.patient_data['case_id'])
			})
		else:
			event_ids = np.where(self.slide_data[self.event_col] == 1)[0]
			non_event_ids = np.where(self.slide_data[self.event_col] == 0)[0]
			cls_ids = [non_event_ids, event_ids]
			settings.update({
				'cls_ids': cls_ids,
				'samples': len(self.slide_data)
			})

		self.split_gen = generate_split(**settings)

	def return_splits(self, from_id=True, csv_path=None):
		if from_id:
			if len(self.train_ids) > 0:
				train_data = self.slide_data.loc[self.train_ids].reset_index(drop=True)
				train_split = Generic_Survival_Split(train_data, data_dir=self.data_dir, 
												num_classes=self.num_classes, 
												event_col=self.event_col, time_col=self.time_col)
			else:
				train_split = None
			
			if len(self.val_ids) > 0:
				val_data = self.slide_data.loc[self.val_ids].reset_index(drop=True)
				val_split = Generic_Survival_Split(val_data, data_dir=self.data_dir,
												num_classes=self.num_classes,
												event_col=self.event_col, time_col=self.time_col)
			else:
				val_split = None
			
			if len(self.test_ids) > 0:
				test_data = self.slide_data.loc[self.test_ids].reset_index(drop=True)
				test_split = Generic_Survival_Split(test_data, data_dir=self.data_dir,
												num_classes=self.num_classes,
												event_col=self.event_col, time_col=self.time_col)
			else:
				test_split = None
			
		else:
			assert csv_path 
			all_splits = pd.read_csv(csv_path, dtype=self.slide_data['slide_id'].dtype)
			train_split = self.get_split_from_df(all_splits, 'train')
			val_split = self.get_split_from_df(all_splits, 'val')
			test_split = self.get_split_from_df(all_splits, 'test')
			
		return train_split, val_split, test_split

	def get_split_from_df(self, all_splits, split_key='train'):
		"""Override get_split_from_df to use Generic_Survival_Split"""
		split = all_splits[split_key]
		split = split.dropna().reset_index(drop=True)

		if len(split) > 0:
			mask = self.slide_data['slide_id'].isin(split.tolist())
			df_slice = self.slide_data[mask].reset_index(drop=True)
			split = Generic_Survival_Split(df_slice, data_dir=self.data_dir, 
										num_classes=self.num_classes,
										event_col=self.event_col)
		else:
			split = None
		
		return split

class Generic_MIL_Survival_Dataset(Generic_WSI_Survival_Dataset):
	def __init__(self,
		data_dir,
		time_col = 'time',
		event_col = 'event', 
		**kwargs):
		"""
		Args:
			data_dir: path to data directory
			time_col: name of the survival time column in dataset
			event_col: name of the event indicator column in dataset
			**kwargs: arguments to pass to parent class
		"""
		self.time_col = time_col
		self.event_col = event_col
		super(Generic_MIL_Survival_Dataset, self).__init__(
			time_col=time_col,
			event_col=event_col,
			**kwargs
		)
		self.data_dir = data_dir
		self.use_h5 = False


	def __getitem__(self, idx):
		slide_id = self.slide_data['slide_id'][idx]
		survival_time = self.slide_data[self.time_col].iloc[idx]
		event = self.slide_data[self.event_col].iloc[idx]
		
		if type(self.data_dir) == dict:
			source = self.slide_data['source'][idx]
			data_dir = self.data_dir[source]
		else:
			data_dir = self.data_dir

		if not self.use_h5:
			if self.data_dir:
				full_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id))
				# h5_path = os.path.join(data_dir, 'h5_files', '{}.h5'.format(slide_id))
				# with h5py.File(h5_path,'r') as hdf5_file:
				# 	coords = hdf5_file['coords'][:]
				features = torch.load(full_path, weights_only=True)
				return features, torch.FloatTensor([survival_time]), torch.FloatTensor([event])
			else:
				return slide_id, torch.FloatTensor([survival_time]), torch.FloatTensor([event])
		else:
			full_path = os.path.join(data_dir,'h5_files','{}.h5'.format(slide_id))
			with h5py.File(full_path,'r') as hdf5_file:
				features = hdf5_file['features'][:]
				coords = hdf5_file['coords'][:]
			features = torch.from_numpy(features)
			return features, torch.FloatTensor([survival_time]), torch.FloatTensor([event]), coords

	def load_from_h5(self, toggle):
		self.use_h5 = toggle



class Generic_Split(Generic_MIL_Dataset):
	def __init__(self, slide_data, data_dir=None, num_classes=2):
		self.use_h5 = False
		self.slide_data = slide_data
		self.data_dir = data_dir
		self.num_classes = num_classes
		self.slide_cls_ids = [[] for i in range(self.num_classes)]
		for i in range(self.num_classes):
			self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

	def __len__(self):
		return len(self.slide_data)

class Generic_Survival_Split(Generic_MIL_Survival_Dataset):
	def __init__(self, slide_data, data_dir=None, num_classes=2, event_col='event', time_col='time'):
		self.use_h5 = False
		self.slide_data = slide_data
		self.data_dir = data_dir
		self.num_classes = num_classes
		self.event_col = event_col
		self.time_col = time_col
		self.slide_cls_ids = [[] for i in range(self.num_classes)]
		for i in range(self.num_classes):
			self.slide_cls_ids[i] = np.where(self.slide_data[self.event_col] == i)[0]

	def __len__(self):
		return len(self.slide_data)
