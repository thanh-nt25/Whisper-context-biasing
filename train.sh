!python scripts/train.py \
    --output $OUTPUT_DIR \
    --bias_weight $BIAS_WEIGHT \
    --data_root $DATA_ROOT \
    --data_dir $DATA_DIR \
    --jsonl_data $JSONL_DATA \
    --batch $BATCH_SIZE \
    --epoch $EPOCHS \
    --lr $LEARNING_RATE \
    --hf_token $HF_TOKEN \
    $( [ "$PROMPT" = "True" ] && echo "--prompt" ) \
    $( [ "$RANDOM" = "True" ] && echo "--random" ) \
    $( [ "$RESUME" = "True" ] && echo "--resume" )