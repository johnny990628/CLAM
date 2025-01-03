TASK=task_survival
TEST_RATIO=0.1
VAL_RATIO=0.1
FOLD=1

TIMESHIMP=$(date +"%Y%m%d%H%M")
LOG_DIR=./RECORDS/SPLIT
LOG_FILE=${TASK}_${VAL_RATIO}_${TEST_RATIO}_${TIMESHIMP}.log

if [ ! -d "$LOG_DIR" ]; then
  mkdir -p "$LOG_DIR"
fi

nohup python create_splits_seq.py \
 --test_frac $TEST_RATIO --val_frac $VAL_RATIO --seed 87 --k $FOLD \
 --task $TASK > ${LOG_DIR}/${LOG_FILE} 2>&1 &
