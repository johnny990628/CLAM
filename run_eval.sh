MAG=10x
PATCHSIZE=224
DATASET=LUAD
TASK=task_tp53_mutation
ENCODER=GIGAPATH
ENCODER_NAME=gigapath
EMBED_DIM=1536
DATA_SPLIT=712
MODEL_TYPE=clam_sb

LEARNING_RATE=1e-5
FOLD=1
WEIGHT_DECAY=1e-4

SPLIT_DIR=./splits/${TASK}_${DATA_SPLIT}
EXP_CODE=${DATA_SPLIT}_${MAG}_${PATCHSIZE}_${LEARNING_RATE}_${WEIGHT_DECAY}
SAVE_DIR=./EVAL_RESULTS/${DATASET}/${TASK}/${ENCODER}/${EXP_CODE}
DATA_DIR=./FEATURES/${ENCODER}/${DATASET}/${MAG}_${PATCHSIZE}
RESULT_DIR=./RESULTS/${DATASET}/${TASK}/${ENCODER}
TIMESHIMP=$(date +"%Y%m%d%H%M")
LOG_DIR=./RECORDS/${ENCODER}/EVAL
LOG_FILE=${DATA_SPLIT}_${MAG}_${PATCHSIZE}_${TIMESHIMP}.log

if [ ! -d "$LOG_DIR" ]; then
  mkdir -p "$LOG_DIR"
fi

nohup python eval.py \
 --embed_dim $EMBED_DIM --model_type $MODEL_TYPE --k $FOLD \
 --models_exp_code $EXP_CODE \
 --save_exp_code $SAVE_DIR \
 --task $TASK \
 --data_root_dir $DATA_DIR \
 --results_dir $RESULT_DIR \
 --splits_dir $SPLIT_DIR > ${LOG_DIR}/${LOG_FILE} 2>&1 &