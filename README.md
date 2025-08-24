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
![framework](SRS/imgs/structure.pdf)


### Datasets
Please put the [BUS]([https://www.kaggle.com/aryashah2k/breast-ultrasound-images-dataset](http://cvprip.cs.usu.edu/busbench/)) dataset and NUDT-KBT2019([https://pan.baidu.com/s/1qGVszDUMYamk8VBsyF1lTQ](提取码:jaq9)) or your own dataset as the following architecture. 
NUDT-KBT2019 is resampled from "A dataset for infrared detection and tracking of dim-small aircraft targets under ground / air background"([http://www.csdata.org/p/387/])
```
    ├── DataSet
        ├── bus
            ├── images
            |   ├── benign (10).png
            │   ├── malignant (17).png
            │   ├── ...
            |
            └── masks
                ├── benign (10).png
                ├── malignant (17).png
                ├── ...
            └── train.txt
            └── val.txt
        ├── KBT2019
            ├── images
            |   ├── 0.bmp
            │   ├── ...
            |
            └── masks
                ├── 0.bmp
                ├── ...
            └── train.txt
            └── val.txt
```
## Performance
To mitigate randomness, each model is retrained three times per dataset depend on three random division, and the final results are derived from the mean performance.
| Methods | BUS IoU | F1 | NUDT-KBT IoU | F1 |
|---------|-----------|----|-----------|----|
| **SRS** | **88.29** | **93.44** | **86.34** | **92.62** |

## Quick Evaluation
Please download the BUS, NUDT-KBT2019 dataset. To quickly evaluation, we give the division of middle performance.
The Weight can be download from baidu disk: BUS([https://pan.baidu.com/s/1NYlrboVL_GaouB8lKiljNA](提取码:r3d1)), NUDT-KBT2019([https://pan.baidu.com/s/1kTps0GLIiDuRv5pun4Jw8g](提取码:beip))
```python
python evaluation.py --base_dir ******/DataSet/bus  --train_file_dir train.txt --val_file_dir val.txt --batch_size 1 --Dataset BUS
```
```python
python evaluation.py --base_dir ******/DataSet/KBT2019  --train_file_dir train.txt --val_file_dir val.txt --batch_size 1 --Dataset KBT2019
```
## Train
You can directly train the reconstruction network and segmentation network:
```python
python train Iris.py
```
```python
python train_Medicine.py
```
## Citation

If you use our code, please cite our paper:


