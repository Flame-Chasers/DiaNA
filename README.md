# 🔥【CVPR 2025】Chat-based Person Retrieval via Dialogue-Refined Cross-Modal Alignment

This repository offers the official implementation of DiaNA in PyTorch.

In the meantime, check out our related papers if you are interested:
+ 【AAAI 2024】 An Empirical Study of CLIP for Text-based Person Search [[paper](https://arxiv.org/abs/2308.10045) | [code](https://github.com/Flame-Chasers/TBPS-CLIP)]
+ 【ACM MM 2023】 Text-based Person Search without Parallel Image-Text Data [[paper](https://arxiv.org/abs/2305.12964)]
+ 【IJCAI 2023】 RaSa: Relation and Sensitivity Aware Representation Learning for Text-based Person Search [[paper](https://arxiv.org/abs/2305.13653) | [code](https://github.com/Flame-Chasers/RaSa)]
+ 【ICASSP 2022】 Learning Semantic-Aligned Feature Representation for Text-based Person Search [[paper](https://arxiv.org/abs/2112.06714) | [code](https://github.com/reallsp/SAF)]


## 📖 Overview

DiaNA is a novel dialogue-refined cross-modal framework for chat-based person retrieval 
that leverages two adaptive attribute refiner modules to bottleneck 
the conversational and visual information for fine-grained cross-modal alignment.

![DiaNA Architecture](figure/architecture.png)




## 📌 TODO
- ⏳ Release code
- ⏳ Release checkpoints
- ✅ Release dataset



## 🗂️ Data Preparation

### 🔹 Pretraining Dataset
- [MALS](https://github.com/Shuyu-XJTU/APTM), a large-scale synthetic TPR dataset with 1.5M image-text pairs

### 🔹 Fine-tuning Dataset: ChatPedes
1. Download images from [CUHK-PEDES](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description)
2. Download ChatPedes annotation files from [here](https://drive.google.com/drive/folders/1-4TjZZ4Z5ANIn3Rx_iBP-MAsu7X28Cob?usp=sharing)
3. Organize the dataset as follows:
```
<ROOT>/ChatPedes
    - train_reid.json
    - test_reid.json
    - imgs00
        - cam_a
        - cam_b
        - ...
```


## 🏋️‍♂️ Training

---

### 🔹 Stage 1: Pretraining on MALS
- **Image Encoder:** [Swin Transformer v2-B](https://huggingface.co/microsoft/swinv2-base-patch4-window12-192-22k)
- **Dialogue Encoder:** [Llama 3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)

**Run Pretraining:**
```bash
cd DiaNA/train
bash shell/pretrain.sh
```

**Resources:**

- 🤗 [Pretrained Checkpoint](https://huggingface.co/byougert/DiaNA-Pretrain/tree/main)
- 📜 [Training Log](https://drive.google.com/file/d/1_6tMAPXI8asZcQVGCOh_qciBHzBv3NCN/view?usp=sharing)

---

### 🔹 Stage 2: Fine-tuning on ChatPedes
**Run Fine-tuning:**
```bash
cd DiaNA/train
bash shell/finetune.sh
```

**Resources:**

- 🤗 [Fine-tuned Checkpoint](https://huggingface.co/byougert/DiaNA/tree/main)
- 📜 [Training Log](https://drive.google.com/file/d/1KBmX0JFMOA0e_ehMh3pKr-KqKnWaXydJ/view?usp=sharing)



## ✨ Citation
If you find this paper useful, please consider staring 🌟 this repo and citing 📑 our paper:
```
@InProceedings{bai2025chat,
    author    = {Bai, Yang and Ji, Yucheng and Cao, Min and Wang, Jinqiao and Ye, Mang},
    title     = {Chat-based Person Retrieval via Dialogue-Refined Cross-Modal Alignment},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages     = {3952--3962},
    month     = {June},
    year      = {2025}
}
```

## ⚖️ License
This code is distributed under an MIT LICENSE.