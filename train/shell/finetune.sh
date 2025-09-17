set -x

GPUS=4
BATCH_SIZE=32

export OMP_NUM_THREADS=1
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34230
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch
export CUDA_VISIBLE_DEVICES=0,1,2,3

PRETRAINED='work_dirs/pretrain'
OUTPUT_DIR='work_dirs/finetune'
VISION_PRETRAINED='<Swin Transformer Root>/swinv2-base-patch4-window12-192-22k'
LLAMA_PRETRAINED='<Llama-3.2 Root>/Llama-3.2-1B'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 4
# batch size per gpu: 32
# gradient accumulation steps: 1
# total batch size: 128
# epoch: 10
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  main/train/train.py \
  --dataset_name 'chatpedes' \
  --pretrained $PRETRAINED \
  --vision_pretrained $VISION_PRETRAINED \
  --llama_pretrained $LLAMA_PRETRAINED \
  --output_dir $OUTPUT_DIR \
  --overwrite_output_dir True \
  --force_image_size 384 192 \
  --drop_path_rate 0.2 \
  --dataloader_num_workers 8 \
  --pad_to_max_length True \
  --bf16 True \
  --num_train_epochs 10 \
  --per_device_train_batch_size $BATCH_SIZE \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --save_strategy "epoch" \
  --learning_rate 1e-5 \
  --weight_decay 0.05 \
  --warmup_steps 50 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --log_level "error" \
  --max_seq_length 400 \
  --version "customized" \
  --do_train True \
  --optim adamw_torch \
  --adam_beta1 0.9 \
  --adam_beta2 0.999 \
  --enable_refiner True \
  --num_query 16 \
  --sim_type "avg" \
  --mask_probability 0.15 \
  --enable_random_turn True \
  --enable_sdm True \
  --report_to "none" \
  --deepspeed "zero_stage3_config.json"