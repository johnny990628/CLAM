patch_size=$1   # 從命令行接收的變數 # sbatch tiling.sh 448 LUAD 20
task=$2
magnification=$3
cd /work/u6658716/TCGA-LUAD/CLAM

nohup python -u create_patches_fp.py \
  --source /work/u6658716/TCGA-LUAD/DATASETS/TCGA/${task} \
  --save_dir /work/u6658716/TCGA-LUAD/CLAM/PATCHES/${task}/${magnification}x_${patch_size} \
  --patch_size ${patch_size} \
  --step_size ${patch_size} \
  --magnification ${magnification} \
  --seg \
  --patch \
  --stitch &