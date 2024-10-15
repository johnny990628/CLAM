import os
import openslide

# 指定要統計的資料夾路徑
folder_path = "/work/u6658716/TCGA-LUAD/DATASETS/TCGA/LUAD"

# 遍歷資料夾中的所有檔案
for filename in os.listdir(folder_path):
    # 檢查檔案是否為.svs檔案
    if filename.endswith(".svs"):
        # 構建完整的檔案路徑
        file_path = os.path.join(folder_path, filename)
        
        try:
            # 使用openslide讀取.svs檔案
            slide = openslide.OpenSlide(file_path)
            
            # 從metadata中提取mpp值
            mpp = slide.properties.get("openslide.mpp-x", None)
            
            # 如果找到mpp值,則輸出
            if mpp:
                print(f"File: {filename}, MPP: {mpp}")
            else:
                print(f"File: {filename}, MPP not found in metadata.")
        
        except openslide.OpenSlideError as e:
            print(f"Error opening file {filename}: {e}")
        
        finally:
            # 關閉openslide物件
            if "slide" in locals():
                slide.close()
