HIGH_MAG=
LOW_MAG=20x
MAG=${HIGH_MAG}${LOW_MAG}
PATCHSIZE=448
DATASET=LUAD
TASK=task_survival
ENCODER=NEW_CONCH
EMBED_DIM=512
DATA_SPLIT=712
MODEL_TYPE=clam_survival

LEARNING_RATE=1e-5
FOLD=1
WEIGHT_DECAY=1e-4

SPLIT_DIR=./splits/${TASK}_486_${DATA_SPLIT}
EXP_CODE=${DATA_SPLIT}_${LEARNING_RATE}_${WEIGHT_DECAY}
SAVE_DIR=./EVAL_RESULTS/${DATASET}/${TASK}/${ENCODER}/${MAG}/${EXP_CODE}
DATA_DIR=./FEATURES/${ENCODER}/${DATASET}/${MAG}_${PATCHSIZE}
RESULT_DIR=./RESULTS/${DATASET}/${TASK}/${ENCODER}/${MAG}
TIMESHIMP=$(date +"%Y%m%d%H%M")
LOG_FILE=${TIMESHIMP}.log

if [ ! -d "$SAVE_DIR" ]; then
  mkdir -p "$SAVE_DIR"
fi


nohup python eval.py \
 --embed_dim $EMBED_DIM --model_type $MODEL_TYPE --k $FOLD \
 --models_exp_code $EXP_CODE \
 --save_exp_code $SAVE_DIR \
 --task $TASK \
 --data_root_dir $DATA_DIR \
 --results_dir $RESULT_DIR \
 --splits_dir $SPLIT_DIR > ${SAVE_DIR}/${LOG_FILE} 2>&1 &