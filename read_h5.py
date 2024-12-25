import h5py
import numpy as np
import matplotlib.pyplot as plt

# Usage example
if __name__ == "__main__":
    h5_file_path = "/work/u6658716/TCGA-LUAD/CLAM/PATCHES/New_LUAD/tiles-5x-s448/tiles-10x-s448/patches/TCGA-4B-A93V-01Z-00-DX1.C263DC1C-298D-47ED-AAF8-128043828530.h5"

    with h5py.File(h5_file_path, 'r') as hf:
        data_coords = hf['coords']

        psize = data_coords.attrs['patch_size']
        level = data_coords.attrs['patch_level']

        print(f"Patch Size: {psize}")
        print(f"Patch Level: {level}")
        print(f"Coords: {data_coords[:5]}")
        print(f"Shape: {data_coords.shape}")

