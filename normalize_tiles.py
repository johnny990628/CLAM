import os
import h5py
import numpy as np
from openslide import OpenSlide
import openslide
from multiprocessing import Pool, cpu_count
from skimage.color import rgb2gray
from skimage.transform import rescale
from wsi_core.WholeSlideImage import WholeSlideImage
import concurrent.futures

def load_patches_from_folder(folder, h5_folder):
    patches_data = []
    total_files = 0

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for root, _, files in os.walk(folder):
            for filename in files:
                if filename.endswith('.svs'):
                    total_files += 1
                    svs_path = os.path.join(root, filename)
                    wsi_object = WholeSlideImage(svs_path)

                    # 讀取對應的 H5 檔案
                    slide_id, _ = os.path.splitext(filename)
                    h5_path = os.path.join(h5_folder, slide_id + '.h5')
                    future = executor.submit(load_patches_from_h5, h5_path, wsi_object)
                    futures.append((filename, future))

        for filename, future in futures:
            patches = future.result()
            patches_data.append((filename, patches))
            print(f"Processing {filename} ({len(patches_data)}/{total_files})")

    return patches_data

def load_patches_from_h5(h5_path, wsi_object):
    patches = []
    with h5py.File(h5_path, 'r') as f:
        coords = f['coords'][:]
        patch_size = f['coords'].attrs['patch_size']
        patch_level = f['coords'].attrs['patch_level']

        for coord in coords:
            x, y = coord
            patch = wsi_object.wsi.read_region((x, y), patch_level, (patch_size, patch_size)).convert('RGB')
            patches.append(np.array(patch))

    return patches

# Function to extract amplitude
def extract_amp(img_np):
    fft = np.fft.fft2(img_np, axes=(-2, -1))
    amp_np = np.abs(fft)
    return amp_np

def compute_batch_avg_amp(batch):
    amp_list = [extract_amp(img_np) for img_np in batch]
    max_shape = tuple(max(s) for s in zip(*[amp.shape for amp in amp_list]))
    padded_amps = [np.pad(amp, [(0, max_shape[0] - amp.shape[0]), (0, max_shape[1] - amp.shape[1])], mode='constant') for amp in amp_list]
    return np.mean(padded_amps, axis=0)

# Function to mutate amplitudes
def mutate(amp_src, amp_trg, L=0.1):
    a_src = np.fft.fftshift(amp_src, axes=(-2, -1))
    a_trg = np.fft.fftshift(amp_trg, axes=(-2, -1))
    h = a_src.shape[1]
    w = a_src.shape[2]

    b = (np.floor(np.amin((h, w)) * L)).astype(int)
    c_h = np.floor(h / 2.0).astype(int)
    c_w = np.floor(w / 2.0).astype(int)

    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1

    a_src[:, h1:h2, w1:w2] = a_trg[:, h1:h2, w1:w2]
    a_src = np.fft.ifftshift(a_src, axes=(-2, -1))
    return a_src

# Function to normalize images using the global average amplitude
def normalize_image(args):
    img_np, amp_trg, L = args
    fft_src_np = np.fft.fft2(img_np, axes=(-2, -1))
    amp_src = np.abs(fft_src_np)
    pha_src = np.angle(fft_src_np)
    amp_src_ = mutate(amp_src, amp_trg, L=L)
    fft_src_ = amp_src_ * np.exp(1j * pha_src)
    src_in_trg = np.fft.ifft2(fft_src_, axes=(-2, -1))
    src_in_trg_real = np.real(src_in_trg)
    return src_in_trg_real

def process_tiles(input_folder, h5_folder, output_folder, batch_size=100):
    patches_data = load_patches_from_folder(input_folder, h5_folder)

    # Step 1: Compute average amplitude for all patches
    all_patches = []
    for _, patches in patches_data:
        all_patches.extend(patches)

    num_batches = (len(all_patches) + batch_size - 1) 
    avg_amps = []
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(all_patches))
        batch_patches = all_patches[start:end]
        avg_amps.append(compute_batch_avg_amp(batch_patches))

    global_avg_amp = np.mean(avg_amps, axis=0)
    print("Global average amplitude calculated.")

    # Step 2: Normalize WSIs using the global average amplitude
    for filename, patches in patches_data:
        slide_id, _ = os.path.splitext(filename)
        svs_path = os.path.join(input_folder, filename)
        wsi_object = WholeSlideImage(svs_path)

        normalized_wsi = normalize_wsi(wsi_object, patches, global_avg_amp)

        # Save normalized WSI as SVS file
        output_path = os.path.join(output_folder, slide_id + '.svs')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        normalized_wsi.wsi.write_wsi(output_path)

def normalize_wsi(wsi_object, patches, global_avg_amp):
    normalized_patches = []
    for img_np in patches:
        normalized_patch = normalize_image((img_np, global_avg_amp, 0.1))
        normalized_patches.append(normalized_patch)

    # Reconstruct normalized WSI from normalized patches
    normalized_wsi = WholeSlideImage(wsi_object.getOpenSlide())
    for i, (x, y) in enumerate(wsi_object.hdf5_file['coords'][:]):
        patch_size = wsi_object.hdf5_file['coords'].attrs['patch_size']
        patch_level = wsi_object.hdf5_file['coords'].attrs['patch_level']
        normalized_wsi.wsi.write_region(location=(x, y), level=patch_level, data=normalized_patches[i], size=(patch_size, patch_size))

    return normalized_wsi

input_dir = '/work/u6658716/TCGA-LUAD/dsmil-wsi/WSI/TCGA-lung_test/1'
h5_dir = '/work/u6658716/TCGA-LUAD/CLAM/RESULTS_DIRECTORY/patches'
output_dir = '/work/u6658716/TCGA-LUAD/CLAM/NORMALIZED'

process_tiles(input_dir, h5_dir, output_dir)
