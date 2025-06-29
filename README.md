# RKSegPlus


This repository is for [RKSeg+: make full use of Runge–Kutta methods in medical image segmentation](https://doi.org/10.1007/s00530-024-01263-6).

### citation
If you find RKSeg+ useful for your research, please consider citing:

    Zhu, M., Fu, C. & Wang, X. RKSeg+: make full use of Runge–Kutta methods in medical image segmentation. Multimedia Systems 30, 65 (2024). https://doi.org/10.1007/s00530-024-01263-6

We implement RKSeg+ based on [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1). You can copy our code into the nnUNet framework for training.

Example of training RKSeg+-E for 150 epochs on Task008_HepaticVessel:

```bash
python createRKplans.py Task008_HepaticVessel 2 E 40
nnUNet_train 2d RKSegPlusETrainerV2_noDeepSupervision_150epochs 8 0 -p nnUNetPlansv2.1_RK-E_40_pool5
```

Example of training RKSeg+-I for 150 epochs on Task008_HepaticVessel:

```bash
python createRKplans.py Task008_HepaticVessel 2 I 40
nnUNet_train 2d RKSegPlusIRTrainerV2_noDeepSupervision_150epochs 8 0 -p nnUNetPlansv2.1_RK-I_40_pool5
```

Example of training RKSeg+-R for 150 epochs on Task008_HepaticVessel:

```bash
python createRKplans.py Task008_HepaticVessel 2 R 40
nnUNet_train 2d RKSegPlusIRTrainerV2_noDeepSupervision_150epochs 8 0 -p nnUNetPlansv2.1_RK-R_40_pool5
```

This article and repository are used for semantic segmentation. This work is based on an earlier piece of work, [RKSeg](https://github.com/ZhuMai/RKSeg). If you are interested in image classification, you can refer to [RKCNN](https://github.com/ZhuMai/RKCNN).
