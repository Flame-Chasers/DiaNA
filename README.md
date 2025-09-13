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
- [ ] Release code
- [ ] Release checkpoints
- [x] Release dataset (ChatPedes)



## 🗂️ Data Preparation
1. Download images from [CUHK-PEDES](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description).
2. Download ChatPedes annotation files from [here](https://drive.google.com/drive/folders/1-4TjZZ4Z5ANIn3Rx_iBP-MAsu7X28Cob?usp=sharing).
3. Organize the dataset as follows:
```
<ROOT>/ChatPedes
    - train_reid.json
    - test_reid.json
    - imgs
        - cam_a
        - cam_b
        - ...
```



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