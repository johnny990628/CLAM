HIGH_MAG=
LOW_MAG=40x
MAG=${HIGH_MAG}${LOW_MAG}
PATCHSIZE=448
DATASET=LUAD
TASK=task_survival
ENCODER=NEW_CONCH
EMBED_DIM=512
DATA_SPLIT=811
MODEL_TYPE=clam_survival
BATCH_SIZE=351

LEARNING_RATE=1e-4
FOLD=1
BAG_LOSS=ce
INST_LOSS=svm
WEIGHT_DECAY=1e-4
DROP_OUT=0.25
WARMUP_EPOCHS=20
MAX_EPOCH=180

SPLIT_DIR=${TASK}_486_${DATA_SPLIT}
EXP_CODE=ple
DATA_DIR=./FEATURES/${ENCODER}/${DATASET}/${MAG}_${PATCHSIZE}
RESULT_DIR=./RESULTS/${DATASET}/${TASK}/${ENCODER}/${MAG}
TIMESHIMP=$(date +"%Y%m%d%H%M")
LOG_FILE=${TIMESHIMP}.log

if [ ! -d "$RESULT_DIR" ]; then
  mkdir -p "$RESULT_DIR"
fi

export HF_TOKEN=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

# nohup python main.py \
#  --log_data --lr_scheduler \
#  --lr $LEARNING_RATE --k $FOLD --bag_loss $BAG_LOSS --inst_loss $INST_LOSS --reg $WEIGHT_DECAY --max_epochs $MAX_EPOCH \
#  --embed_dim $EMBED_DIM --drop_out $DROP_OUT --warmup_epochs $WARMUP_EPOCHS --batch_size $BATCH_SIZE \
#  --exp_code $EXP_CODE \
#  --task $TASK --model_type $MODEL_TYPE \
#  --split_dir $SPLIT_DIR \
#  --results_dir $RESULT_DIR \
#  --data_root_dir $DATA_DIR > ${RESULT_DIR}/${LOG_FILE} 2>&1 &

nohup python main.py \
 --early_stopping --log_data --lr_scheduler \
 --lr $LEARNING_RATE --k $FOLD --bag_loss $BAG_LOSS --inst_loss $INST_LOSS --reg $WEIGHT_DECAY  \
 --embed_dim $EMBED_DIM --drop_out $DROP_OUT --warmup_epochs $WARMUP_EPOCHS --batch_size $BATCH_SIZE \
 --exp_code $EXP_CODE \
 --task $TASK --model_type $MODEL_TYPE \
 --split_dir $SPLIT_DIR \
 --results_dir $RESULT_DIR \
 --data_root_dir $DATA_DIR > ${RESULT_DIR}/${LOG_FILE} 2>&1 &