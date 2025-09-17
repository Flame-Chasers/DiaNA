from contextlib import suppress
from datetime import datetime
from enum import Enum

import torch
import torch.nn.functional as F
from easydict import EasyDict
from prettytable import PrettyTable
from tqdm import tqdm


class EvaluationMode(Enum):
    GLOBAL = "global"
    LOCAL = "local"
    ALL = "all"


def evaluate(model, dataloader, tokenizer, max_seq_length, mode=EvaluationMode.GLOBAL):
    """
    Evaluate the model on the given dataset

    Parameters
    ----------

    model: torch.nn,Module
        CLIP-like model with `encode_image` and `encode_text`

    dataloader: torch.utils.data.Dataloader
        dataloader to use for evaluation

    tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase
        tokenizer to use for evaluation

    max_seq_length: int
        maximum length of the text

    mode: evaluation mode, can be "global", "local" or "all"

    Returns
    -------

    dict of retrieval metrics
    """
    batch_images_emb_list = []
    batch_texts_emb_list = []
    batch_images_attr_list = []
    batch_texts_attr_list = []
    # for each text, we collect the corresponding image index, as each image can have multiple corresponding texts
    for batch_images in tqdm(dataloader, desc="Computing image embeddings"):
        # compute the embedding of images
        with torch.no_grad(), suppress():
            batch_image_embeds, batch_image_dense = model.encode_image(batch_images)
            batch_images_emb = F.normalize(batch_image_embeds, dim=-1)
            batch_images_emb_list.append(batch_images_emb)

            if mode in (EvaluationMode.LOCAL, EvaluationMode.ALL):
                batch_images_attr = model.refine(batch_image_dense)
                batch_images_attr_list.append(batch_images_attr)

    dataset = dataloader.dataset
    captions = dataset.captions
    num_text = len(captions)
    text_bs = 256
    for i in tqdm(range(0, num_text, text_bs), desc="Computing text embeddings"):
        batch_caption = captions[i: min(num_text, i + text_bs)]
        batch_caption = tokenizer(
            batch_caption,
            max_length=max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        ).to(model.device)
        with torch.no_grad(), suppress():
            batch_text_embeds, batch_text_dense = model.encode_text(batch_caption.input_ids, batch_caption.attention_mask)
            batch_texts_emb = F.normalize(batch_text_embeds, dim=-1)
            batch_texts_emb_list.append(batch_texts_emb)

            if mode in (EvaluationMode.LOCAL, EvaluationMode.ALL):
                batch_text_attr = model.refine(batch_text_dense, batch_caption.attention_mask).cpu()
                batch_texts_attr_list.append(batch_text_attr)

    # concatenate all embeddings
    images_emb = torch.cat(batch_images_emb_list)
    texts_emb = torch.cat(batch_texts_emb_list)
    images_pids = torch.tensor(dataset.image_ids, dtype=torch.long)
    texts_pids = torch.tensor(dataset.caption_ids, dtype=torch.long)

    # metrics list
    metrics_list = []

    # get the score for each text and image pair
    scores_global = texts_emb @ images_emb.t()
    scores_global = scores_global.cpu()
    metrics_global = metric_eval(scores_global, images_pids, texts_pids)
    metrics_list.append(EasyDict(mode=EvaluationMode.GLOBAL, **metrics_global))

    if mode in (EvaluationMode.LOCAL, EvaluationMode.ALL):
        images_attr = torch.cat(batch_images_attr_list)
        texts_attr = torch.cat(batch_texts_attr_list)

        scores_local_list = []
        for text_chunk_attr in texts_attr.chunk(8192, dim=0):
            score_chunk_local = model.refine_sim(text_chunk_attr.cuda(), images_attr)
            scores_local_list.append(score_chunk_local)
        scores_local = torch.cat(scores_local_list)
        scores_local = scores_local.cpu()

        metrics_local = metric_eval(scores_local, images_pids, texts_pids)
        metrics_list.append(EasyDict(mode=EvaluationMode.LOCAL, **metrics_local))

        if mode == EvaluationMode.ALL:
            metrics_all = metric_eval(scores_global + scores_local, images_pids, texts_pids)
            metrics_list.append(EasyDict(mode=EvaluationMode.ALL, **metrics_all))

    metrics_table = MetricTable(metrics_list)
    return metrics_table


def metric_eval(sims_matrix, image_pid, text_pid):
    device = sims_matrix.device
    image_pid = image_pid.to(device=device)
    text_pid = text_pid.to(device=device)

    index = torch.argsort(sims_matrix, dim=-1, descending=True)
    pred_pid = image_pid[index]
    matches = (text_pid.view(-1, 1).eq(pred_pid)).long()

    def acc_k(matches, k=1):
        matches_k = matches[:, :k].sum(dim=-1)
        matches_k = torch.sum((matches_k > 0))
        return 100.0 * matches_k / matches.size(0)

    # Compute metrics
    r1 = acc_k(matches, k=1).item()
    r5 = acc_k(matches, k=5).item()
    r10 = acc_k(matches, k=10).item()

    real_num = matches.sum(dim=-1)
    tmp_cmc = matches.cumsum(dim=-1).float()
    order = torch.arange(start=1, end=matches.size(1) + 1, dtype=torch.long, device=device)
    tmp_cmc /= order
    tmp_cmc *= matches
    AP = tmp_cmc.sum(dim=-1) / real_num
    mAP = (AP.mean() * 100.0).item()

    mfr = matches.argmax(dim=-1).float().mean().item()

    return dict(
        r1=r1,
        r5=r5,
        r10=r10,
        mAP=mAP,
        mfr=mfr,
    )


class MetricTable(object):

    def __init__(self, metrics_list, steps="-"):
        self.metrics_list = metrics_list
        self.steps = steps

    def set_steps(self, steps):
        self.steps = steps

    def get_best_metric(self):
        return max([metric.r1 for metric in self.metrics_list])

    def __str__(self):
        def format_table(tb, field_names, fmt_func):
            for name in field_names:
                tb.custom_format[name] = fmt_func

        table = PrettyTable(["Steps", "Mode", "R1", "R5", "R10", "mAP", "mFR", "Time"])
        format_table(table, field_names=["R1", "R5", "R10", "mAP", "mFR"], fmt_func=lambda f, v: f"{round(v, 2):.2f}")

        cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for metric in self.metrics_list:
            table.add_row([self.steps, metric.mode.value,
                           metric.r1, metric.r5, metric.r10,
                           metric.mAP, metric.mfr, cur_time])

        return table.get_string()
