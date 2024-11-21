import os
import openslide

# 指定要統計的資料夾路徑
folder_path = "/work/u6658716/TCGA-LUAD/DATASETS/TCGA/LUAD"

value_count = {'40':0, '20':0, 'error':0}

# 遍歷資料夾中的所有檔案
for filename in os.listdir(folder_path):
    
    # 檢查檔案是否為.svs檔案
    if filename.endswith(".svs"):
        # 構建完整的檔案路徑
        file_path = os.path.join(folder_path, filename)
        
        try:
            # 使用openslide讀取.svs檔案
            slide = openslide.OpenSlide(file_path)
            objective_power = slide.properties.get("openslide.objective-power")

            if objective_power:
                print(f"File: {filename}, OP: {objective_power}")
                value_count[objective_power] = int(value_count[objective_power])+1
            else:
                print(f"File: {filename}, OP not found in metadata.")
                os.remove(file_path)
                value_count['error'] = int(value_count['error'])+1
        
        except openslide.OpenSlideError as e:
            print(f"Error opening file {filename}: {e}")
        
        finally:
            # 關閉openslide物件
            if "slide" in locals():
                slide.close()
print(value_count)
