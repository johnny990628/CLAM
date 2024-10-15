import h5py
from wsi_core.WholeSlideImage import WholeSlideImage

# 假設您已經有一個 HDF5 檔案,其中包含了一張 WSI 影像的所有 patches
hdf5_file_path = '/work/u6658716/TCGA-LUAD/CLAM/RESULTS_DIRECTORY/patches/TCGA-4B-A93V-01Z-00-DX1.C263DC1C-298D-47ED-AAF8-128043828530.h5'

# 初始化 WholeSlideImage 物件
wsi_object = WholeSlideImage('/work/u6658716/TCGA-LUAD/DATASETS/TCGA/LUAD/TCGA-4B-A93V-01Z-00-DX1.C263DC1C-298D-47ED-AAF8-128043828530.svs')

# 讀取 HDF5 檔案
with h5py.File(hdf5_file_path, 'r') as f:
    # 獲取 HDF5 檔案中的資訊
    patch_size = f['coords'].attrs['patch_size']
    patch_level = f['coords'].attrs['patch_level']
    downsample = f['coords'].attrs['downsample']
    level_dim = f['coords'].attrs['level_dim']
    name = f['coords'].attrs['name']
    
    # 讀取 patches 的座標
    coords = f['coords'][:]

    # 遍歷每個 patch 的座標
    for coord in coords:
        x, y = coord
        
        # 從 WSI 影像中讀取 patch
        patch = wsi_object.wsi.read_region((x, y), patch_level, (patch_size, patch_size)).convert('RGB')

