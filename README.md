# Progressive neural network for multi-horizon time series forecasting
Information Sciences 661, 120112

Author: [Yang Lin](https://yanglin1997.github.io/)

E-mail: linyang1997@yahoo.com.au


## Abstract
<p align="justify">
In this paper, we introduce ProNet, an novel deep learning approach designed for multi-horizon time series forecasting, adaptively blending autoregressive (AR) and non-autoregressive (NAR) strategies. Our method involves dividing the forecasting horizon into segments, predicting the most crucial steps in each segment non-autoregressively, and the remaining steps autoregressively. The segmentation process relies on latent variables, which effectively capture the significance of individual time steps through variational inference. In comparison to AR models, ProNet showcases remarkable advantages, requiring fewer AR iterations, resulting in faster prediction speed, and mitigating error accumulation. On the other hand, when compared to NAR models, ProNet takes into account the interdependency of predictions in the output space, leading to improved forecasting accuracy. Our comprehensive evaluation, encompassing four large datasets, and an ablation study, demonstrate the effectiveness of ProNet, highlighting its superior performance in terms of accuracy and prediction speed, outperforming state-of-the-art AR and NAR forecasting models.

This repository provides an implementation for ProNet as described in the paper:

> Progressive neural network for multi-horizon time series forecasting.
> Yang Lin.
> Information Sciences.
> [[Paper]](https://arxiv.org/pdf/2310.19322)

**Citing**

If you find ProNet and the new datasets useful in your research, please consider adding the following citation:

```bibtex
@article{LIN2024120112,
title = {Progressive neural network for multi-horizon time series forecasting},
journal = {Information Sciences},
volume = {661},
pages = {120112},
year = {2024},
issn = {0020-0255},
doi = {https://doi.org/10.1016/j.ins.2024.120112},
url = {https://www.sciencedirect.com/science/article/pii/S0020025524000252},
author = {Yang Lin},
keywords = {Time series forecasting, Deep learning, Transformer, Variational inference},
}
```

## List of Implementations:

Sanyo: http://dkasolarcentre.com.au/source/alice-springs/dka-m4-b-phase

Hanergy: http://dkasolarcentre.com.au/source/alice-springs/dka-m16-b-phase

Solar: https://www.nrel.gov/grid/solar-power-data.html
