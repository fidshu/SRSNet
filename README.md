# SRSNet
SRS: Siamese Reconstruction-Segmentation Network based on Dynamic-Parameter Convolution

Official pytorch code base for TIP 2025 "SRS: Siamese Reconstruction-Segmentation Network based on Dynamic-Parameter Convolution"

**News** 🥰:
- <font color="#dd0000" size="4">**CMUNeXt is accepted by TIP !**</font> 🎉
- <font color="#dd0000" size="4">**Paper is rejected by ECCV **</font> 😭😭😭🥹🥹🥹
- <font color="#dd0000" size="4">**Paper is rejected By TCSVT **</font> 😭😭😭🥹🥹🥹
- <font color="#dd0000" size="4">**Paper is rejected by CVPR **</font> 😭😭😭🥹🥹🥹


## Introduction
Dynamic convolution demonstrates outstanding representation capabilities, which is crucial for natural image segmentation. However, they fail when applied to medical image segmentation (MIS) and infrared small target segmentation (IRSTS) due to limited data and limited fitting capacity. In this paper, we propose a new type of dynamic convolution called dynamic parameter convolution (DPConv) which shows superior fitting capacity, and it can efficiently leverage features from deep layers of encoder in reconstruction tasks to generate DPConv kernels that adapt to input variations.
Moreover, we observe that DPConv, built upon deep features derived from reconstruction tasks, significantly enhances downstream segmentation performance. 
We refer to the segmentation network integrated with DPConv generated from reconstruction network as the siamese reconstruction-segmentation network (SRS). We conduct extensive experiments on seven datasets including five medical datasets and two infrared datasets, and the experimental results demonstrate that our method can show superior performance over several recently proposed methods. Furthermore, the zero-shot segmentation under unseen modality demonstrates the generalization of DPConv.

### SRSNet:


