# MEASURE: Multi-Scale Representation Learning for Cross-Subject Sleep Staging

## Motivation
![motivation](./assets/fig1.png)
(a) Illustration of contrastive learning. It maximizes the similarity between $\boldsymbol{z}_i$ and $\boldsymbol{z}_p$, where $\boldsymbol{z}_i$ denotes the feature of the $i$-th sample $\boldsymbol{v}_i$, and $\boldsymbol{z}_p$ represents that of a positive sample $\boldsymbol{v}_p$. From an information-theoretic view, this induces sufficient learning (b) but also introduces superfluous information $I(\boldsymbol{z}_i;\boldsymbol{v}_i\mid\boldsymbol{v}_p)$  \cite{tsai2021self}. In particular, the domain-induced excess domain-relevant information $I(\boldsymbol{z}_i;d_i\mid\boldsymbol{v}_p)$ hinders domain-invariant learning, where $D$ denotes domain factors and $d_i$ is the domain label of $\boldsymbol{v}_i$. (c) Minimal Sufficient Representation Learning: It mitigates the superfluous information $I(\boldsymbol{z}_i;\boldsymbol{v}_i\mid\boldsymbol{v}_p)$, thereby reducing the excess domain-relevant information and promoting the learning of domain-invariant features.


## Overall framework
Overall framework
![Overall framework](./assets/fig2.png)


## Environment Setup
* Python 3.9
* Cuda 12.1
* Pytorch 2.31
* Required libraries are listed in requirements.txt.

```bash
pip install -r requirements.txt
```

## Data Preprosessing
Download the [SleepEDF20](https://www.physionet.org/content/sleep-edfx/1.0.0/), and [MASS3](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/9MYUCS) and put them the data dir.

Convert the data to .npz format.
```bash
python Preprocessing.py
```
## Run
Our model consist of pretrain and fintuing part.
### Pretrain
First, model's feature extractor learn the domain-invarint feature via multi-scale minimal sufficient learning.
```bash
python Pretrain.py
```

### Pretrain
Second, To demonstrate the performance of the feature extractor, we train a transformer-based classifier while keeping the parameters of the feature extractor fixed. The transformer-based classifier follows the model proposed in prior work [SleePyCo](https://www.sciencedirect.com/science/article/pii/S0957417423030531) for sleep scoring.

```bash
python FineTuning.py
```

## Acknowledgement
The code is inspired by prior awesome works:

[SleePyCo: Automatic sleep scoring with feature pyramid and contrastive learning
](https://www.sciencedirect.com/science/article/pii/S0957417423030531) (Expert Systems with Applications 2024)

[MVEB: Self-Supervised Learning With Multi-View Entropy Bottleneck
](https://ieeexplore.ieee.org/document/10477543) (Transactions on Pattern Analysis and Machine Intelligence 2024)


