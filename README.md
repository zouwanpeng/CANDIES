# Enhancing spatial domain identification from spatial multi-omics data through cross-modal denoising with CANDIES

## Abstract
Spatial multi-omics data provide a powerful platform for capturing diverse molecular profiles, enabling the simultaneous analysis of multiple data modalities within the same tissue section. However, raw data are often plagued by significant noise, which poses a major challenge for accurate interpretation and integration analysis. To overcome this limitation, we introduce CANDIES, a novel and robust framework that harnesses the strengths of denoising diffusion probabilistic models (DDPM) and contrastive learning, effectively denoise spatial multi-omics data while generating a unified and comprehensive representation. This representation serves as a foundation for a wide array of downstream analyses, including denoising, spatial domain identification, pseudo-Spatiotemporal Map (pSM) construction, and trait domain association map generation. Through extensive evaluation across diverse datasets, we demonstrate that CANDIES consistently outperforms existing state-of-the-art methods, delivering more accurate, reliable, and biologically insightful results. As such, CANDIES stands as an invaluable tool for advancing spatial multi-omics research, offering researchers a powerful solution to unravel the intricacies of tissue organization and function.

![](https://github.com/zouwanpeng/CANDIES/blob/main/CANDIES.png)

## Requirements
You'll need to install the following packages in order to run the codes.
* ﻿anndata==0.11.0
* einops==0.8.0
* esda==2.7.0
* networkx==2.6
* numpy==1.26.4
* rpy2==3.5.16
* scanpy==1.10.3
* scikit-learn==1.5.2
* squidpy==1.6.2
* torch==2.5.1
* torch-geometric==2.6.1
* torchvision==0.20.1
* R==4.0.3

## Tutorial
For the step-by-step tutorial, please refer to: https://github.com/zouwanpeng/CANDIES/tree/main/CANDIES/tutorial

## Benchmarking methods
In the paper, we compared CANDIES with 7 unimodal clustering methods, 2 non-spatial multi-omics approaches, and 4 spatial multi-omics techniques.

## Downstream analyses
