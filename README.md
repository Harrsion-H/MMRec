
# MMRec项目介绍
## 数据集
data 是数据集入口，目前的microlens 是网上人提供的数据源，提取了图片特征，以及文本特征。
mydata 是草稿，我之前做了一些的大模型，生成数据的结果。（tag不需要了，原数据集中有）
## 进度
跑通了MGCN，结果如下：


### 实验参数
- seed: 999
- cl_loss: 0.01

### 验证集结果

| 指标        | @5     | @10    | @20    | @50    |
|------------|--------|--------|--------|--------|
| Recall     | 0.0456 | 0.0751 | 0.1147 | 0.1812 |
| NDCG       | 0.0288 | 0.0383 | 0.0484 | 0.0617 |
| Precision  | 0.0095 | 0.0078 | 0.0060 | 0.0038 |
| MAP        | 0.0230 | 0.0270 | 0.0297 | 0.0318 |

### 测试集结果

| 指标        | @5     | @10    | @20    | @50    |
|------------|--------|--------|--------|--------|
| Recall     | 0.0455 | 0.0755 | 0.1130 | 0.1784 |
| NDCG       | 0.0288 | 0.0386 | 0.0483 | 0.0615 |
| Precision  | 0.0099 | 0.0082 | 0.0062 | 0.0039 |
| MAP        | 0.0228 | 0.0268 | 0.0295 | 0.0317 |


好像很差


# MMRec


## Toolbox
<p>
<img src="./images/MMRec.png" width="500">
</p>

## Supported Models
source code at: `src\models`

| **Model**       | **Paper**                                                                                             | **Conference/Journal** | **Code**    |
|------------------|--------------------------------------------------------------------------------------------------------|------------------------|-------------|
| **General models**  |                                                                                                        |                        |             |
| SelfCF              | [SelfCF: A Simple Framework for Self-supervised Collaborative Filtering](https://arxiv.org/abs/2107.03019)                                 | ACM TORS'23            | selfcfed_lgn.py  |
| LayerGCN            | [Layer-refined Graph Convolutional Networks for Recommendation](https://arxiv.org/abs/2207.11088)                                          | ICDE'23                | layergcn.py  |
| **Multimodal models**  |                                                                                                        |                        |             |
| VBPR              | [VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1510.01784)                                              | AAAI'16                 | vbpr.py      |
| MMGCN             | [MMGCN: Multi-modal Graph Convolution Network for Personalized Recommendation of Micro-video](https://staff.ustc.edu.cn/~hexn/papers/mm19-MMGCN.pdf)               | MM'19                  | mmgcn.py  |
| ItemKNNCBF             | [Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches](https://arxiv.org/abs/1907.06902)               | RecSys'19              | itemknncbf.py  |
| GRCN              | [Graph-Refined Convolutional Network for Multimedia Recommendation with Implicit Feedback](https://arxiv.org/abs/2111.02036)            | MM'20                  | grcn.py    |
| MVGAE             | [Multi-Modal Variational Graph Auto-Encoder for Recommendation Systems](https://ieeexplore.ieee.org/abstract/document/9535249)              | TMM'21                 | mvgae.py   |
| DualGNN           | [DualGNN: Dual Graph Neural Network for Multimedia Recommendation](https://ieeexplore.ieee.org/abstract/document/9662655)                   | TMM'21                 | dualgnn.py   |
| LATTICE           | [Mining Latent Structures for Multimedia Recommendation](https://arxiv.org/abs/2104.09036)                                               | MM'21                  | lattice.py  |
| SLMRec            | [Self-supervised Learning for Multimedia Recommendation](https://ieeexplore.ieee.org/document/9811387) | TMM'22                 |                  slmrec.py |
| **Newly added**  |                                                                                                        |                        |             |
| BM3         | [Bootstrap Latent Representations for Multi-modal Recommendation](https://dl.acm.org/doi/10.1145/3543507.3583251)                                          | WWW'23                 | bm3.py |
| FREEDOM | [A Tale of Two Graphs: Freezing and Denoising Graph Structures for Multimodal Recommendation](https://arxiv.org/abs/2211.06924)                                 | MM'23                  | freedom.py  |
| MGCN     | [Multi-View Graph Convolutional Network for Multimedia Recommendation](https://arxiv.org/abs/2308.03588)                       | MM'23               | mgcn.py          |
| DRAGON  | [Enhancing Dyadic Relations with Homogeneous Graphs for Multimodal Recommendation](https://arxiv.org/abs/2301.12097)                                 | ECAI'23                | dragon.py  |
| MG  | [Mirror Gradient: Towards Robust Multimodal Recommender Systems via Exploring Flat Local Minima](https://arxiv.org/abs/2402.11262)                                 | WWW'24                | [trainer.py](src/common/trainer.py#L166-L185)  |
| LGMRec  | [LGMRec: Local and Global Graph Learning for Multimodal Recommendation](https://arxiv.org/abs/2312.16400)                                 | AAAI'24                | lgmrec.py |


#### Please consider to cite our paper if this framework helps you, thanks:
```
@inproceedings{zhou2023bootstrap,
author = {Zhou, Xin and Zhou, Hongyu and Liu, Yong and Zeng, Zhiwei and Miao, Chunyan and Wang, Pengwei and You, Yuan and Jiang, Feijun},
title = {Bootstrap Latent Representations for Multi-Modal Recommendation},
booktitle = {Proceedings of the ACM Web Conference 2023},
pages = {845–854},
year = {2023}
}

@article{zhou2023comprehensive,
      title={A Comprehensive Survey on Multimodal Recommender Systems: Taxonomy, Evaluation, and Future Directions}, 
      author={Hongyu Zhou and Xin Zhou and Zhiwei Zeng and Lingzi Zhang and Zhiqi Shen},
      year={2023},
      journal={arXiv preprint arXiv:2302.04473},
}

@inproceedings{zhou2023mmrec,
  title={Mmrec: Simplifying multimodal recommendation},
  author={Zhou, Xin},
  booktitle={Proceedings of the 5th ACM International Conference on Multimedia in Asia Workshops},
  pages={1--2},
  year={2023}
}
```
