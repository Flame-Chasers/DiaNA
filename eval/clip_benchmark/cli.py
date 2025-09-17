"""Console script for clip_benchmark."""
import argparse
import logging
import sys
from copy import copy

import torch
import yaml
from torch.utils.data import DataLoader

from clip_benchmark.datasets.builder import build_dataset
from clip_benchmark.metrics.retrieval import evaluate, EvaluationMode
from clip_benchmark.models.diana import load_diana

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

from transformers.utils import logging as logging_utils

logging_utils.set_verbosity_error()


def get_parser_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='cifar10', nargs='+',
                        help="Dataset(s) to use for the benchmark. Can be the name of a dataset, or a collection name ('vtab', 'vtab+', 'imagenet_robustness', 'retrieval') or path of a text file where each line is a dataset name")
    parser.add_argument('--dataset_config_path', default='configs/dataset_config.yaml', type=str,
                        help="The dataset configuration path")
    parser.add_argument('--split', type=str, default='test', help='Dataset split to use')
    parser.add_argument('--pretrained', type=str, default='', nargs='+',
                        help="Pre-trained model(s) to use.")
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--seed', default=0, type=int, help='random seed.')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--max_seq_length', default=80, type=int,
                        help='The maximum total input sequence length after tokenization. Sequences longer '
                             'than this will be truncated, sequences shorter will be padded.')
    parser.add_argument('--turn_limit', default=-1, type=int, nargs='+',
                        help='Controls the maximum number of turns in a dialogue during model training. '
                             'If set to a value less than 0, it indicates that all dialogue turns should be used without any limit.')
    parser.add_argument('--version', default="", type=str, help='conversion version')
    parser.add_argument('--mode', default="global", type=str, choices=["global", "local", "all"],
                        help='conversion version')
    parser.add_argument('--output_dir', default='', type=str, help="output dictionary")

    args = parser.parse_args()
    return args


def main():
    base = get_parser_args()
    # Get list of pre-trained models to evaluate
    pretrained_list = _as_list(base.pretrained)
    if 'checkpoint-' in pretrained_list[0]:
        pretrained_list = sorted(pretrained_list, key=lambda x: int(x.split('checkpoint-')[1]))

    # Ge list of datasets to evaluate on
    datasets = _as_list(base.dataset)
    turn_limit_list = _as_list(base.turn_limit)

    best_metrics = None
    for i, pretrained in enumerate(pretrained_list):
        print(f'Evaluate the {i + 1}/{len(pretrained_list)}-th model.')
        for dataset in datasets:
            for turn_limit in turn_limit_list:
                # We iterative over all possible model/dataset
                # TODO: possibility to parallelize evaluation here
                args = copy(base)
                args.pretrained = pretrained
                args.dataset = dataset
                args.turn_limit = turn_limit
                args.steps = int(pretrained.split('checkpoint-')[1]) if 'checkpoint-' in pretrained else "-"
                metrics = run(args)
                if best_metrics is None or metrics.get_best_metric() > best_metrics.get_best_metric():
                    best_metrics = metrics

    print('Best metrics:')
    print(best_metrics)


def _as_list(l):
    if not l:
        return []
    return [l] if type(l) != list else l


def run(args):
    """Console script for clip_benchmark."""
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # set seed.
    torch.manual_seed(args.seed)
    dataset_name = args.dataset
    print(f"Running evaluation under 'round={args.turn_limit}' with model '{args.pretrained}'.")

    with open(args.dataset_config_path, "r") as f:
        dataset_config = yaml.load(f, Loader=yaml.FullLoader)

    model, tokenizer = load_diana(
        pretrained=args.pretrained,
        device=args.device,
    )
    model.eval()

    args.force_image_size = model.config.vision_config.image_size

    dataset = build_dataset(
        dataset_name=dataset_name,
        metas=dataset_config[dataset_name],
        tokenizer=tokenizer,
        args=args
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
    )

    metrics = evaluate(
        model,
        dataloader,
        tokenizer,
        max_seq_length=args.max_seq_length,
        mode=EvaluationMode(args.mode),
    )
    metrics.set_steps(args.steps)
    print(metrics, '\n')

    return metrics


if __name__ == '__main__':
    sys.exit(main())  # pragma: no cover
