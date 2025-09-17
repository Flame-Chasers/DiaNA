set -x

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=4

OUTPUT_DIR='work_dirs/finetune'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

python clip_benchmark/cli.py \
  --pretrained "../train/work_dirs/finetune" \
  --dataset "chatpedes" \
  --dataset_config_path "configs/dataset_config.yaml" \
  --batch_size 128 \
  --max_seq_length 400 \
  --turn_limit -1 \
  --mode "global" \
  --version customized \
  --output_dir $OUTPUT_DIR

