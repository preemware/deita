export WANDB_PROJECT="Deita-Scorers"
MODELPATH="meta-llama/Meta-Llama-3-8B"
DATAPATH="deita-10k-v0.json"
MODEL_SIZE="8B"
RUNNAME="Deita-8B-Scorers"
OUTPUTPATH="out"
TOTALBSZ=512
BSZPERDEV=1
DEVICES="0,1"
NUMGPUS=$(echo $DEVICES | awk -F',' '{print NF}')
GRADACC=$(($TOTALBSZ/$NUMGPUS/$BSZPERDEV))
EPOCHNUM=6
echo "Training llama-3-8b model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BSZPERDEV batch size per GPU, $GRADACC gradient accumulation steps"

deepspeed --include localhost:$DEVICES --master_port 29502 src/deita/alignment/train_scorers.py \
    --model_name_or_path ${MODELPATH} \
    --data_path ${DATAPATH} \
    --output_dir ${OUTPUTPATH}/${RUNNAME} \
    --num_train_epochs ${EPOCHNUM} \
    --per_device_train_batch_size ${BSZPERDEV} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${GRADACC} \
    --eval_steps 50 \
    --save_strategy "no" \
    --save_steps 100 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --do_eval False \
    --evaluation_strategy "no" \
    --model_max_length 2048 \
    --lazy_preprocess True \
    --conv_template "scorer" \
    --mask_user True \
    --report_to "wandb" \
    --run_name ${RUNNAME} \
    --bf16 True \
    --deepspeed src/deita/ds_configs/deepspeed_config_zero2_no_offload.json
