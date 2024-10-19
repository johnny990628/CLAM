import os
import h5py
import pandas as pd

# 指定包含 .h5 檔案的資料夾路徑
folder_path = '/work/u6658716/TCGA-LUAD/CLAM/PATCHES_10x_448/patches'

# 創建一個列表來存儲結果
results = []

# 遍歷資料夾中的所有檔案
for filename in os.listdir(folder_path):
    if filename.endswith('.h5'):
        file_path = os.path.join(folder_path, filename)
        
        try:
            with h5py.File(file_path, 'r') as f:
                # 獲取 coords 數量
                coords_count = len(f['coords'])
                
                # 獲取其他相關資訊
                patch_size = f['coords'].attrs.get('patch_size', 'N/A')
                patch_level = f['coords'].attrs.get('patch_level', 'N/A')
                downsample = f['coords'].attrs.get('downsample', 'N/A')
                level_dim = f['coords'].attrs.get('level_dim', 'N/A')
                name = f['coords'].attrs.get('name', filename)
                
                # 將結果添加到列表中
                results.append({
                    'Filename': filename,
                    'Name': name,
                    'Coords Count': coords_count,
                    'Patch Size': patch_size,
                    'Patch Level': patch_level,
                    'Downsample': downsample,
                    'Level Dim': level_dim
                })
                
                print(f"Processed: {filename}, Coords Count: {coords_count}")
        
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

# 創建 DataFrame
df = pd.DataFrame(results)

# 將結果保存為 CSV 檔案
output_path = '/work/u6658716/TCGA-LUAD/CLAM/10x_h5_analysis_results.csv'
df.to_csv(output_path, index=False)

print(f"Results saved to: {output_path}")
