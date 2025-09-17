import collections
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional, Union, List

import transformers
import yaml
from PIL import Image, ImageFile, PngImagePlugin
from transformers import (HfArgumentParser, TrainingArguments, default_data_collator, set_seed)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.logging import enable_default_handler, enable_explicit_format, set_verbosity

from main.dist_utils import init_dist
from main.model.diana import DiaNAConfig, DiaNAModel, Llama3Tokenizer
from main.train.dataset import build_dataset
from main.train.trainer import Trainer
from main.train.trainer_monkey_patch import replace_create_optimizer
from main.utils import print_model_param_info

IGNORE_INDEX = -100
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    vision_pretrained: str = field(
        default=None,
        metadata={'help': 'Path to the pretrained vision model'}
    )
    llama_pretrained: str = field(
        default=None,
        metadata={'help': 'Path to the pretrained llama model'}
    )
    pretrained: str = field(
        default=None,
        metadata={'help': 'Path to the pretrained vision model'}
    )
    enable_refiner: bool = field(
        default=False,
        metadata={'help': 'whether to enable the refiner module'},
    )
    enable_sdm: bool = field(
        default=False,
        metadata={'help': 'whether to enable the refiner module'},
    )
    num_query: int = field(
        default=16,
        metadata={'help': 'number of query for the refiner module'},
    )
    sim_type: str = field(
        default='avg',
        metadata={'help': 'similarity type of the refiner module'},
    )
    freeze_model: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the entire model.'},
    )
    freeze_vision_model: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the vision backbone of the model.'},
    )
    freeze_llama: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the LLaMA of the model.'},
    )
    unfreeze_llama_head: bool = field(
        default=False,
        metadata={'help': 'Set to True to unfreeze the head of the LLaMA.'},
    )
    unfreeze_llama_projection: bool = field(
        default=False,
        metadata={'help': 'Set to True to unfreeze the text projection of the LLaMA.'},
    )
    use_vision_lora: int = field(
        default=0, metadata={'help': 'If non-zero, indicates the use of LoRA in the vision backbone of the model'}
    )
    use_llama_lora: int = field(
        default=0, metadata={'help': 'If non-zero, indicates the use of LoRA in the LLaMA of the model'}
    )
    use_custom_trainer: bool = field(
        default=False, metadata={'help': 'Set to True to enable the use of a custom trainer.'},
    )
    drop_path_rate: float = field(
        default=0.0, metadata={'help': 'Specify the value of drop path rate in the vision backbone. Default is 0.'}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default='flickr30k',
        metadata={'help': 'Specify the name of dataset to be used.'},
    )
    dataset_config_path: Optional[str] = field(
        default='configs/dataset_config.yaml',
        metadata={'help': 'The dataset configuration path'},
    )
    split: Optional[str] = field(
        default='train',
        metadata={'help': 'Dataset split to use'},
    )
    max_seq_length: Optional[int] = field(
        default=80,
        metadata={
            'help': (
                'The maximum total input sequence length after tokenization. Sequences longer '
                'than this will be truncated, sequences shorter will be padded.'
            )
        },
    )
    mask_probability: float = field(
        default=0.,
        metadata={'help': 'mask probability of the input ids'},
    )
    enable_random_turn: bool = field(
        default=False,
        metadata={'help': 'whether to enable the random turn of dialogue'},
    )
    force_image_size: Union[str, List[str]] = field(
        default="224",
        metadata={'help': 'Specify the image size for training models.'},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            'help': (
                'Whether to pad all samples to model maximum sentence length. '
                'If False, will pad the samples dynamically when batching to the maximum length in the batch. More '
                'efficient on GPU but very bad for TPU.'
            )
        },
    )
    turn_limit: Optional[int] = field(
        default=-1,
        metadata={
            'help': (
                'Controls the maximum number of turns in a dialogue during model training. '
                'If set to a value less than 0, it indicates that all dialogue turns should be used without any limit.'
            )
        },
    )
    version: Optional[str] = field(
        default="v0",
        metadata={
            'help': (
                'Chat version.'
            )
        },
    )


def main():
    # Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # If use DeepSpeed zero3, init_dist must before HfArgumentParser
    launcher = os.environ.get('LAUNCHER', 'slurm')
    init_dist(launcher=launcher, backend='nccl')
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry('finetune Flickr30K', model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}'
        + f'distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}'
    )
    logger.info(f'Training/evaluation parameters {training_args}')

    # Detecting last checkpoint and eventually continue from last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f'Output directory ({training_args.output_dir}) already exists and is not empty. '
                'Use --overwrite_output_dir to overcome.'
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f'Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change '
                'the `--output_dir` or add `--overwrite_output_dir` to train from scratch.'
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model, tokenizer, and image processor
    tokenizer = Llama3Tokenizer.from_pretrained(
        model_args.llama_pretrained,
        add_bos_token=True,
        add_eos_token=True,
    )

    # build dataset
    with open(data_args.dataset_config_path, "r") as f:
        dataset_config = yaml.load(f, Loader=yaml.FullLoader)
    train_dataset = build_dataset(
        dataset_name=data_args.dataset_name,
        metas=dataset_config[data_args.dataset_name],
        tokenizer=tokenizer,
        data_args=data_args
    )

    if model_args.pretrained:
        config = DiaNAConfig.from_pretrained(model_args.pretrained)
    else:
        config = DiaNAConfig.from_pretrained('configs')

    config.refiner_config.enable_refiner = model_args.enable_refiner
    config.enable_sdm = model_args.enable_sdm
    config.refiner_config.enable_sdm = model_args.enable_sdm

    config.refiner_config.num_query = model_args.num_query
    config.refiner_config.sim_type = model_args.sim_type
    config.vision_config.drop_path_rate = model_args.drop_path_rate
    config.vision_config.pretrained = model_args.vision_pretrained
    config.llama_config.pretrained = model_args.llama_pretrained
    if isinstance(data_args.force_image_size, str):
        data_args.force_image_size = int(data_args.force_image_size)
    elif isinstance(data_args.force_image_size, collections.Iterable):
        data_args.force_image_size = [int(size) for size in data_args.force_image_size]
    config.force_image_size = data_args.force_image_size
    config.vision_config.image_size = data_args.force_image_size

    config.llama_config.pad_token_id = tokenizer.pad_token_id

    if model_args.pretrained:
        model = DiaNAModel.from_pretrained(
            model_args.pretrained,
            config=config,
        )
    else:
        model = DiaNAModel(config).to(device=training_args.device, dtype=config.torch_dtype)

    model.config.use_cache = False
    model.config.llama_config.use_cache = False
    model.vision_model.gradient_checkpointing_enable()
    model.llama.gradient_checkpointing_enable()

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_model:
        _freeze_params(model)

    if model_args.freeze_vision_model:
        model.vision_model = model.vision_model.eval()
        _freeze_params(model.vision_model)

    if model_args.freeze_llama:
        model.llama = model.llama.eval()
        _freeze_params(model.llama)

    if model_args.use_llama_lora:
        model.wrap_llama_lora(r=model_args.use_llama_lora, lora_alpha=model_args.use_llama_lora * 2)
        model.config.use_llama_lora = model_args.use_llama_lora

    if model_args.unfreeze_llama_projection:
        model.text_projection.requires_grad = True

    # print trainable parameters
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.requires_grad)
    print_model_param_info(model)

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    # Initialize our Trainer
    if model_args.use_custom_trainer:
        replace_create_optimizer()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics['train_samples'] = len(train_dataset)
        trainer.log_metrics('train', metrics)
        # trainer.save_metrics('train', metrics)
        # trainer.save_state()


if __name__ == '__main__':
    main()
