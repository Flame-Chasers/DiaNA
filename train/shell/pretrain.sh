set -x

GPUS=8
BATCH_SIZE=64

export OMP_NUM_THREADS=1
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

OUTPUT_DIR='work_dirs/pretrain'
VISION_PRETRAINED='/data/byyoung/data/swin/swinv2_huggingface/swinv2-base-patch4-window12-192-22k'
LLAMA_PRETRAINED='/data/byyoung/data/llama3/Llama-3.2-1B'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 8
# batch size per gpu: 64
# gradient accumulation steps: 1
# total batch size: 512
# epoch: 3
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  main/train/train.py \
  --dataset_name 'mals' \
  --dataset_config_path 'configs/dataset_config.yaml' \
  --vision_pretrained $VISION_PRETRAINED \
  --llama_pretrained $LLAMA_PRETRAINED \
  --output_dir $OUTPUT_DIR \
  --overwrite_output_dir True \
  --force_image_size 384 192 \
  --drop_path_rate 0.2 \
  --dataloader_num_workers 8 \
  --pad_to_max_length True \
  --bf16 True \
  --num_train_epochs 3 \
  --per_device_train_batch_size $BATCH_SIZE \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --save_strategy "epoch" \
  --save_total_limit 1 \
  --learning_rate 1e-4 \
  --weight_decay 0.05 \
  --warmup_steps 1000 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --log_level "error" \
  --max_seq_length 56 \
  --version "customized" \
  --do_train True \
  --optim adamw_torch \
  --adam_beta1 0.9 \
  --adam_beta2 0.98 \
  --enable_refiner True \
  --num_query 24 \
  --mask_probability 0.15 \
  --enable_sdm False \
  --sim_type "avg" \
  --report_to "none" \
  --deepspeed "zero_stage3_config.json"