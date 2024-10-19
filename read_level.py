import openslide
import os
import pandas as pd

def get_slide_info(slide_path):
    slide = openslide.OpenSlide(slide_path)
    
    info = {
        'filename': os.path.basename(slide_path),
        'dimensions': slide.dimensions,
        'level_count': slide.level_count,
    }
    
    for i in range(slide.level_count):
        info[f'level_{i}_dimensions'] = slide.level_dimensions[i]
        info[f'level_{i}_downsample'] = slide.level_downsamples[i]
    
    slide.close()
    return info

def process_slides_in_directory(directory):
    slide_info_list = []
    
    for filename in os.listdir(directory):
        if filename.endswith('.svs'):
            slide_path = os.path.join(directory, filename)
            try:
                slide_info = get_slide_info(slide_path)
                slide_info_list.append(slide_info)
                print(f"Processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    return slide_info_list

# 使用方法
directory = '/work/u6658716/TCGA-LUAD/DATASETS/TCGA/LUAD'
slide_info_list = process_slides_in_directory(directory)

# 將結果轉換為 DataFrame 並儲存為 CSV
df = pd.DataFrame(slide_info_list)
df.to_csv('slide_info.csv', index=False)
print("Results saved to slide_info.csv")

