MAG=20x
PATCHSIZE=224
DATASET=LUAD
TASK=task_tp53_mutation
ENCODER=GIGAPATH
ENCODER_NAME=gigapath
BATCH_SIZE=256
EMBED_DIM=1536
DATA_SPLIT=712
MODEL_TYPE=clam_sb

LEARNING_RATE=1e-5
FOLD=1
BAG_LOSS=ce
INST_LOSS=SVM
WEIGHT_DECAY=1e-4
DROP_OUT=0.25

SPLIT_DIR=${TASK}_${DATA_SPLIT}
EXP_CODE=${DATA_SPLIT}_${PATCHSIZE}_${MAG}_${LEARNING_RATE}_${WEIGHT_DECAY}
DATA_DIR=./FEATURES/${ENCODER}/${DATASET}/${MAG}_${PATCHSIZE}
RESULT_DIR=./RESULTS/${DATASET}/${TASK}/${ENCODER}
TIMESHIMP=$(date +"%Y%m%d%H%M")
LOG_DIR=./RECORDS/${ENCODER}/TRAIN
LOG_FILE=${DATA_SPLIT}_${MAG}_${PATCHSIZE}_${TIMESHIMP}.log

if [ ! -d "$LOG_DIR" ]; then
  mkdir -p "$LOG_DIR"
fi

nohup python main.py \
 --early_stopping --weighted_sample --log_data --subtyping \
 --lr $LEARNING_RATE --k $FOLD --bag_loss $BAG_LOSS --inst_loss $INST_LOSS --reg $WEIGHT_DECAY  \
 --embed_dim $WEIGHT_DECAY --drop_out $DROP_OUT \
 --exp_code $EXP_CODE \
 --task $TASK --model_type $DATA_DIR \
 --split_dir $SPLIT_DIR \
 --results_dir $RESULT_DIR \
 --data_root_dir $DATA_DIR > ${LOG_DIR}/${LOG_FILE} 2>&1 &