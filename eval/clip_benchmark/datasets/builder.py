import json
import os.path as op
import random
import re
from collections import defaultdict
from typing import Dict
from typing import Sequence

import torchvision.transforms as T
from PIL import Image
from timm.models.layers import to_2tuple
from torch.utils.data import Dataset

from clip_benchmark.datasets import conversation as conversation_lib

PROMPT_POOL = [
    "Please describe the person you saw.",
    "What is the person wearing in the picture?",
    "Can you detail the clothing and accessories of the person?",
    "Describe the appearance of the individual in the image.",
    "How would you describe the attire of the person in the image?",
    "Please provide details about the person's outfit.",
    "Describe the attire and notable features of the individual in the picture.",
]


def build_dataset(dataset_name, metas, tokenizer, args):
    """
    Main function to use in order to build a dataset instance,

    dataset_name: str
        name of the dataset

    root: str
        root folder where the dataset is downloaded and stored. can be shared among datasets.

    transform: torchvision transform applied to images

    annotation_file: str or None
        only for datasets with captions (used for retrieval) such as COCO
        and Flickr.

    max_words: int
        the maximum total input sequence length
    """
    __DATASET_FACTOR__ = {
        'chatpedes': ChatPedesDataset,
    }

    assert dataset_name in __DATASET_FACTOR__.keys(), f'Unsupported dataset: {dataset_name}'

    return __DATASET_FACTOR__[dataset_name](metas, tokenizer, args)


class BaseDataset(Dataset):

    def __init__(self, metas, tokenizer, args):
        super(BaseDataset, self).__init__()
        self.metas = metas
        self.tokenizer = tokenizer
        self.transform = self.build_transform(args.force_image_size)

        self.args = args
        split = args.split
        ann_name = split + "_annotation"

        # adjust conversation lib
        if args.version == "customized":
            self.conversation_lib = conversation_lib.conv_templates[args.version]
        else:
            raise NotImplementedError

        ann_files = metas[ann_name]
        if isinstance(metas[ann_name], str):
            ann_files = [ann_files]

        self.anns = []
        for ann_file in ann_files:
            self.anns += json.load(open(ann_file, 'r'))

        self.images = []
        self.captions = []
        self.image_ids = []
        self.caption_ids = []

    def __len__(self):
        return len(self.images)

    def print_info(self):
        print(f'There are {len(set(self.images))} images.')
        print(f'There are {len(set(self.image_ids))} ids.')
        print(f'There are {len(self.captions)} captions.')

    def build_transform(self, input_size):
        mean = self.metas['mean']
        std = self.metas['std']

        input_size = to_2tuple(input_size)
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(input_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

        return transform

    def preprocess_conversation(self, sources: Sequence[str]):
        conv = self.conversation_lib.copy()
        roles = {"gpt": conv.roles[0], "user": conv.roles[1]}

        # Apply prompt templates
        conv.messages = []
        for i, source in enumerate(sources):
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
        conversation = conv.get_prompt()
        return conversation

    def add_prompt(self, caption: str):
        conv = self.conversation_lib.copy()

        # Apply prompt templates
        conv.messages = []
        prompt = random.choice(PROMPT_POOL)
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], caption)

        conversion = conv.get_prompt()
        return conversion

    def process_single_caption(self, caption):
        max_words = self.args.max_seq_length
        caption = re.sub(r"([.!\"()*#:;~])", ' ', caption.lower())
        caption = re.sub(r'\s{2,}', ' ', caption)
        caption = caption.rstrip('\n')
        caption = caption.strip(' ')

        # truncate caption
        caption_words = caption.split(' ')
        if len(caption_words) > max_words:
            caption = ' '.join(caption_words[: max_words])
        return caption

    def preprocess_inputs(self, image, caption):
        model_inputs = dict()

        # input image
        image_transform = self.build_transform(self.args.force_image_size)
        image = Image.open(image)
        image = image.convert('RGB')
        image = image_transform(image)
        model_inputs['image'] = image

        # for image-text contrastive learning
        text_inputs = self.tokenizer(
            caption,
            max_length=self.args.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        model_inputs['input_ids'] = text_inputs['input_ids']
        model_inputs['attention_mask'] = text_inputs['attention_mask']

        return model_inputs

    def __getitem__(self, i):
        image = Image.open(self.images[i])
        image = image.convert('RGB')
        image = self.transform(image)
        return image


class FlickrDataset(BaseDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, metas, tokenizer, args):
        super(FlickrDataset, self).__init__(metas, tokenizer, args)

        for ann in self.anns:
            image = ann['image']
            image_path = op.join(metas['image_root'], image)
            image_id = int(ann['image_id'])
            captions = [self.process_single_caption(caption) for caption in ann['captions']]
            self.images.append(image_path)
            self.image_ids.append(image_id)
            self.captions.append(captions)

        self.print_info()


class PedesDataset(BaseDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, metas, tokenizer, args):
        super(PedesDataset, self).__init__(metas, tokenizer, args)

        person_id2idx = defaultdict(lambda: len(person_id2idx))
        for image_i, ann in enumerate(self.anns):
            person_idx = person_id2idx[ann['id']]
            image_path = op.join(metas['image_root'], ann['file_path'])
            self.images.append(image_path)
            self.image_ids.append(person_idx)

            for caption in ann['captions']:
                caption = self.process_single_caption(caption)
                # caption = self.add_prompt(caption)
                self.captions.append(caption)
                self.caption_ids.append(person_idx)

        self.print_info()


class ChatPedesDataset(BaseDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, metas, tokenizer, args):
        super(ChatPedesDataset, self).__init__(metas, tokenizer, args)

        self.turn_limit = args.turn_limit

        person_id2idx = defaultdict(lambda: len(person_id2idx))
        for image_i, ann in enumerate(self.anns):
            person_id = ann['id']
            person_idx = person_id2idx[person_id]
            image_path = op.join(metas['image_root'], ann['file_path'])
            self.images.append(image_path)
            self.image_ids.append(person_idx)

            chats = ann['chats']
            if isinstance(chats[0][0], Dict):
                chats = [chats]

            for chat in chats:
                chat = chat[:self.turn_limit] if self.turn_limit > 0 else chat
                caption = self.preprocess_conversation(chat)
                self.captions.append(caption)
                self.caption_ids.append(person_idx)

        self.print_info()
