import os
import openslide
import pandas as pd
from collections import Counter

# 定義常見的倍率
COMMON_MAGNIFICATIONS = [1, 2.5, 5, 10, 20, 40, 100]

def get_nearest_magnification(value, common_magnifications):
    """取得最接近的常見倍率."""
    return min(common_magnifications, key=lambda x: abs(x - value))


def get_magnifications(svs_path):
    """取得 SVS 檔案中每個階層的 magnification，並轉為整數."""
    try:
        slide = openslide.OpenSlide(svs_path)
        magnifications = []
        base_magnification = float(slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER, 0))
        for level in range(slide.level_count):
            downsample = slide.level_downsamples[level]
            raw_magnification = base_magnification / downsample
            adjusted_magnification = get_nearest_magnification(raw_magnification, COMMON_MAGNIFICATIONS)
            magnifications.append(adjusted_magnification)
        return magnifications
    except Exception as e:
        print(f"Error reading {svs_path}: {e}")
        return []

def process_svs_folder(folder_path):
    """處理資料夾中所有 SVS 檔案，統計 magnifications."""
    data = []
    all_magnifications = []
    for file in os.listdir(folder_path):
        if file.endswith(".svs"):
            svs_path = os.path.join(folder_path, file)
            magnifications = get_magnifications(svs_path)
            all_magnifications.extend(magnifications)
            data.append({"file_name": file, "magnifications": magnifications})
    return data, all_magnifications

def save_magnification_statistics(all_magnifications, output_csv_path):
    """統計每個 magnification 的出現次數，並儲存為 CSV."""
    counts = Counter(all_magnifications)
    df = pd.DataFrame(counts.items(), columns=["magnification", "count"])
    df.sort_values(by="magnification", inplace=True)
    df.to_csv(output_csv_path, index=False)
    print(f"Magnification statistics saved to {output_csv_path}")

def save_magnifications_per_file(data, output_csv_path):
    """將每個檔案的 magnifications 資訊存為 CSV."""
    rows = []
    for item in data:
        file_name = item["file_name"]
        for level, magnification in enumerate(item["magnifications"]):
            rows.append({"file_name": file_name, "level": level, "magnification": magnification})
    
    df = pd.DataFrame(rows)
    df.to_csv(output_csv_path, index=False)
    print(f"Per-file magnifications saved to {output_csv_path}")

# 設定資料夾路徑與輸出檔案路徑
input_folder = "/work/u6658716/TCGA-LUAD/DATASETS/TCGA/LUAD"
output_csv_file_magnifications = "./LUAD_mag.csv"
output_csv_magnification_stats = "./mag_stat.csv"

# 執行統計
svs_data, all_magnifications = process_svs_folder(input_folder)

# 儲存每個檔案的 magnifications
save_magnifications_per_file(svs_data, output_csv_file_magnifications)

# 統計 magnifications 出現次數並儲存
save_magnification_statistics(all_magnifications, output_csv_magnification_stats)
