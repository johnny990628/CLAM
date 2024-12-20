HIGH_MAG=40x
LOW_MAG=5x
MAG=${HIGH_MAG}${LOW_MAG}
PATCHSIZE=448
DATASET=LUAD
TASK=task_tp53_mutation
ENCODER=CONCH
EMBED_DIM=512
DATA_SPLIT=712
MODEL_TYPE=clam_sb

LEARNING_RATE=1e-5
FOLD=1
BAG_LOSS=ce
INST_LOSS=svm
WEIGHT_DECAY=1e-4
DROP_OUT=0.25
WARMUP_EPOCHS=10

SPLIT_DIR=${TASK}_318_${DATA_SPLIT}
EXP_CODE=${DATA_SPLIT}_${LEARNING_RATE}_${WEIGHT_DECAY}
DATA_DIR=./FEATURES/${ENCODER}/${DATASET}/${MAG}_${PATCHSIZE}
RESULT_DIR=./RESULTS/${DATASET}/${TASK}/${ENCODER}/${MAG}
TIMESHIMP=$(date +"%Y%m%d%H%M")
LOG_DIR=./RECORDS/${ENCODER}/TRAIN
LOG_FILE=${DATA_SPLIT}_${MAG}_${PATCHSIZE}_${TIMESHIMP}.log

if [ ! -d "$LOG_DIR" ]; then
  mkdir -p "$LOG_DIR"
fi

nohup python main.py \
 --early_stopping --weighted_sample --log_data --subtyping --lr_scheduler --multi_scale \
 --lr $LEARNING_RATE --k $FOLD --bag_loss $BAG_LOSS --inst_loss $INST_LOSS --reg $WEIGHT_DECAY  \
 --embed_dim $EMBED_DIM --drop_out $DROP_OUT --warmup_epochs $WARMUP_EPOCHS \
 --exp_code $EXP_CODE \
 --task $TASK --model_type $MODEL_TYPE \
 --split_dir $SPLIT_DIR \
 --results_dir $RESULT_DIR \
 --data_root_dir $DATA_DIR > ${LOG_DIR}/${LOG_FILE} 2>&1 &