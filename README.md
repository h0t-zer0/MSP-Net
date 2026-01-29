# RETHINKING MULTI-SCALE PERCEPTION FOR CAMOUFLAGED OBJECT DETECTION (ICASSP 2026)

> **Authors:** 
> Chunyuan Chen,
> Weiyun Liang,
> Mingyang Yu,
> Ji Du,
> Shujuan Li,
> and Jing Xu.

## 1. Overview

- This repository provides code for "_**RETHINKING MULTI-SCALE PERCEPTION FOR CAMOUFLAGED OBJECT DETECTION**_", IEEE International Conference on Acoustics, Speech, and Signal Processing, 2026. [[Paper]](url-link for the paper)

### 1.1 Introduction

Camouflaged object detection (COD) is highly challenging due to the strong visual similarity between camouflaged objects and their surroundings in colors, textures, and shapes, as well as the variability in object scale and number. Multi-scale perception has been widely adopted to address this issue, yet existing approaches often fail to fully capture both semantic context and fine details. In this paper, we revisit multi-scale perception for COD and propose a novel Multi-Scale Perception Network (MSPNet). MSPNet is built upon two key modules: the Dynamic Perception Module, which enhances progressive global context through adaptive local perception branches, and the Mixed Perception Module, which further refines features via a nested multi-receptive-field structure to strengthen semantic awareness and spatial detail. Extensive experiments prove that MSPNet outperforms 15 state-of-the-art methods on three COD datasets.

<p align="center">
    <img src="imgs/FINet.png"/> <br />
    <em> 
    <b>Fig. 1:</b> <b>(a)</b> Illustrations of the progressive global perception (left) and our proposed dynamic perception (right). <b>(b)</b> Demonstration of the multi-receptive-field perception and our proposed mixed perception (equipped with the mixed perception layer). For simplicity, we only demonstrate the mixed perception layer on a single branch.
    </em>
</p>
 
Source code and results for submission to **ICASSP 2026: RETHINKING MULTI-SCALE PERCEPTION FOR CAMOUFLAGED OBJECT DETECTION**

[Prediction Results](https://drive.google.com/drive/folders/1i0OyK_Wy21_ybnY9QKIWUjsG-kvG9-qr?usp=sharing)
