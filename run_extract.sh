MAG=40x
PATCHSIZE=224
DATASET=LUAD
ENCODER=GIGAPATH
ENCODER_NAME=gigapath
BATCH_SIZE=256

H5_DIR=./PATCHES/${DATASET}/${MAG}_${PATCHSIZE}
SLIDE_DIR=../DATASETS/TCGA/${DATASET}
FEAT_DIR=./FEATURES/${ENCODER}/${DATASET}/${MAG}_${PATCHSIZE}
CSV_PATH=./PATCHES/${DATASET}/${MAG}_${PATCHSIZE}/process_list_autogen.csv
TIMESHIMP=$(date +"%Y%m%d%H%M")
LOG_DIR=./RECORDS/${ENCODER}/EXTRACT
LOG_FILE=${MAG}_${PATCHSIZE}_${TIMESHIMP}.log

if [ ! -d "$LOG_DIR" ]; then
  mkdir -p "$LOG_DIR"
fi

nohup python extract_features_fp.py --data_h5_dir $H5_DIR \
 --data_slide_dir $SLIDE_DIR \
 --csv_path $CSV_PATH \
 --feat_dir $FEAT_DIR \
 --batch_size $BATCH_SIZE \
 --slide_ext .svs \
 --model_name $ENCODER_NAME \
 --target_patch_size $PATCHSIZE > ${LOG_DIR}/${LOG_FILE} 2>&1 &
