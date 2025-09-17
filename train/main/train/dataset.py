import json
import os.path as op
import random
import re
from collections import defaultdict
from typing import Dict, Sequence, List

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from timm.models.layers import to_2tuple
from torch.utils.data import Dataset

from main import conversation as conversation_lib

PROMPT_POOL = [
    "Please describe the person you saw.",
    "What is the person wearing in the picture?",
    "Can you detail the clothing and accessories of the person?",
    "Describe the appearance of the individual in the image.",
    "How would you describe the attire of the person in the image?",
    "Please provide details about the person's outfit.",
    "Describe the attire and notable features of the individual in the picture.",
]


def build_dataset(dataset_name, metas, tokenizer, data_args):
    __DATASET_FACTOR__ = {
        'chatpedes': ChatPedesDataset,
        'mals': PretrainChatPedesDataset
    }

    assert dataset_name in __DATASET_FACTOR__, f'{dataset_name} is not supported.'

    return __DATASET_FACTOR__[dataset_name](metas, tokenizer, data_args)


class TransformChoose:
    def __init__(self, transform_pool, input_size, mean, std, n=2):
        self.transform_pool = transform_pool
        self.input_size = input_size
        self.mean = mean
        self.std = std
        self.n = n

    def __call__(self, image):
        aug_choice = np.random.choice(self.transform_pool, self.n)
        return T.Compose([
            T.Resize(self.input_size),
            T.ToTensor(),
            *aug_choice,
            T.Normalize(mean=self.mean, std=self.std),
        ])(image)


class BaseDataset(Dataset):

    def __init__(self, metas, tokenizer, data_args):
        super(BaseDataset, self).__init__()
        self.metas = metas
        self.tokenizer = tokenizer
        self.data_args = data_args

        # adjust conversation lib
        if data_args.version == "customized":
            self.conversation_lib = conversation_lib.conv_templates[data_args.version]
        else:
            raise NotImplementedError

        split = data_args.split
        ann_name = split + "_annotation"

        self.anns = []
        if isinstance(metas[ann_name], List):
            for ann_file in metas[ann_name]:
                if ann_file.endswith('.json'):
                    self.anns += json.load(open(ann_file, 'r'))
        else:
            self.anns += json.load(open(metas[ann_name], 'r'))

        self.images = []
        self.image_ids = []
        self.captions = []

    def print_info(self):
        print(f'There are {len(set(self.images))} images.')
        print(f'There are {len(set(self.image_ids))} ids.')
        print(f'There are {len(self.captions)} captions.')

    def __len__(self):
        return len(self.captions)

    def build_vision_transform(self, input_size):
        mean = self.metas['mean']
        std = self.metas['std']

        input_size = to_2tuple(input_size)
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(input_size),
            T.RandomHorizontalFlip(),
            T.Pad(10),
            T.RandomCrop(input_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(scale=(0.02, 0.4), value=mean),
        ])

        ##########  stolen from TBPR-CLIP (https://arxiv.org/abs/2308.10045) #############
        # transform_pool = [
        #     T.ColorJitter(.1, .1, .1, 0),
        #     T.RandomRotation(15),
        #     T.RandomResizedCrop(input_size, (0.9, 1.0), antialias=True),
        #     T.RandomGrayscale(),
        #     T.RandomHorizontalFlip(),
        #     T.RandomErasing(scale=(0.10, 0.20)),
        # ]
        #
        # transform = TransformChoose(transform_pool, input_size, mean, std, n=2)

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
        max_words = self.data_args.max_seq_length
        caption = re.sub(r"([.!\"()*#:;~])", ' ', caption.lower())
        caption = re.sub(r'\s{2,}', ' ', caption)
        caption = caption.rstrip('\n')
        caption = caption.strip(' ')

        # truncate caption
        caption_words = caption.split(' ')
        if len(caption_words) > max_words:
            caption = ' '.join(caption_words[: max_words])
        return caption

    def mask_sentence(self, sentence: str, mask_probability: float = 0.) -> str:
        if mask_probability == 0.:
            return sentence

        words = sentence.split()
        masked_words = [
            self.tokenizer.mask_token if random.random() < mask_probability else word
            for word in words
        ]
        masked_sentence = ' '.join(masked_words)

        # remove the space before the mask token
        pattern = r'\s*' + re.escape(self.tokenizer.mask_token)
        return re.sub(pattern, self.tokenizer.mask_token, masked_sentence)

    def mask(self, input_ids: torch.LongTensor, mask_probability: float = 0.) -> torch.LongTensor:
        if mask_probability == 0.:
            return input_ids

        probability_matrix = torch.full(input_ids.shape, mask_probability)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        masked_indices[input_ids == self.tokenizer.bos_token_id] = False
        masked_indices[input_ids == self.tokenizer.eos_token_id] = False
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_token_ids = torch.randint(self.tokenizer.solid_token_id_up_limit, input_ids.shape).long()
        input_ids[indices_random] = random_token_ids[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        return input_ids

    def preprocess_inputs(self, image, caption):
        model_inputs = dict()

        # input image
        image_transform = self.build_vision_transform(self.data_args.force_image_size)
        image = Image.open(image)
        image = image.convert('RGB')
        pixel_values = image_transform(image)
        model_inputs['pixel_values'] = pixel_values

        # for image-text contrastive learning
        text_inputs = self.tokenizer(
            caption,
            max_length=self.data_args.max_seq_length,
            padding='max_length' if self.data_args.pad_to_max_length else False,
            truncation=True,
            return_tensors='pt',
        )
        model_inputs['input_ids'] = self.mask(text_inputs['input_ids'], self.data_args.mask_probability)
        model_inputs['attention_mask'] = text_inputs['attention_mask']

        return model_inputs

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        i = i % len(self.images)
        image = self.images[i]
        caption = self.add_prompt(self.captions[i])
        ret = self.preprocess_inputs(image, caption)
        # for image-text contrastive learning
        ret['input_ids'] = ret['input_ids'][0]
        ret['attention_mask'] = ret['attention_mask'][0]
        ret['image_ids'] = torch.Tensor([self.image_ids[i]]).long()
        return ret


class FlickrDataset(BaseDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, metas, tokenizer, data_args):
        super(FlickrDataset, self).__init__(metas, tokenizer, data_args)

        for ann in self.anns:
            image = ann['image']
            image_path = op.join(metas['image_root'], image)
            image_id = int(ann['image_id'])
            caption = ann['caption']
            caption = self.process_single_caption(caption)
            self.images.append(image_path)
            self.image_ids.append(image_id)
            self.captions.append(caption)

        self.print_info()


class PedesDataset(BaseDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, metas, tokenizer, data_args):
        super(PedesDataset, self).__init__(metas, tokenizer, data_args)

        person_id2idx = defaultdict(lambda: len(person_id2idx))
        for image_i, ann in enumerate(self.anns):
            person_id = ann['id']
            person_idx = person_id2idx[person_id]
            image_path = op.join(metas['image_root'], ann['file_path'])
            for caption in ann['captions']:
                caption = self.process_single_caption(caption)
                self.images.append(image_path)
                self.image_ids.append(person_idx)
                self.captions.append(caption)

        self.print_info()


class ChatPedesDataset(BaseDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, metas, tokenizer, data_args):
        super(ChatPedesDataset, self).__init__(metas, tokenizer, data_args)

        self.turn_limit = data_args.turn_limit

        person_id2idx = defaultdict(lambda: len(person_id2idx))
        for image_i, ann in enumerate(self.anns):
            person_id = ann['id']
            person_idx = person_id2idx[person_id]
            image_path = op.join(metas['image_root'], ann['file_path'])

            chats = ann['chats']
            if isinstance(chats[0][0], Dict):
                chats = [chats]

            for chat in chats:
                caption = chat[:self.turn_limit] if self.turn_limit > 0 else chat
                self.images.append(image_path)
                self.image_ids.append(person_idx)
                self.captions.append(caption)

        self.print_info()

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        i = i % len(self.images)
        image = self.images[i]

        caption = self.captions[i]
        if self.data_args.enable_random_turn:
            random_turn = random.randint(1, len(caption))
            caption = caption[:random_turn]

        caption = self.preprocess_conversation(caption)
        ret = self.preprocess_inputs(image, caption)
        # for image-text contrastive learning
        ret['input_ids'] = ret['input_ids'][0]
        ret['attention_mask'] = ret['attention_mask'][0]
        ret['image_ids'] = torch.Tensor([self.image_ids[i]]).long()
        return ret


class PretrainChatPedesDataset(BaseDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, metas, tokenizer, data_args):
        super(PretrainChatPedesDataset, self).__init__(metas, tokenizer, data_args)

        image_id2idx = defaultdict(lambda: len(image_id2idx))
        for ann in self.anns:
            image_id = ann['image_id']
            image_idx = image_id2idx[image_id]
            image_path = op.join(metas['image_root'], ann['image'])
            self.images.append(image_path)
            self.image_ids.append(image_idx)

            caption = self.process_single_caption(ann['caption'])
            self.captions.append(caption)

        self.print_info()
