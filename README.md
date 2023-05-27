# DART: Diversify-Aggregate-Repeat Training
This repository contains codes for the training and evaluation of our CVPR-23 paper [DART:Diversify-Aggregate-Repeat Training Improves Generalization of Neural Networks] (https://openaccess.thecvf.com/content/CVPR2023/papers/Jain_DART_Diversify-Aggregate-Repeat_Training_Improves_Generalization_of_Neural_Networks_CVPR_2023_paper.pdf) and [supplementary] (https://openaccess.thecvf.com/content/CVPR2023/supplemental/Jain_DART_Diversify-Aggregate-Repeat_Training_CVPR_2023_supplemental.pdf). The arxiv  link for the paper is also [available](https://arxiv.org/pdf/2302.14685.pdf).

 # Environment Settings 
* Python 3.6.9
* PyTorch 1.8
* Torchvision 0.8.0
* Numpy 1.19.2

The checkpoints can be found at [Google Drive]()
# Training
For training DAJAT: 
```
python train_DAJAT.py --use_defaults ['NONE','CIFAR10_RN18', 'CIFAR10_WRN','CIFAR100_WRN', 'CIFAR100_RN18']
```
For training ACAT: 
```
python train_DAJAT.py --use_defaults ['NONE','CIFAR10_RN18', 'CIFAR10_WRN','CIFAR100_WRN', 'CIFAR100_RN18']  --num_autos 0 --epochs 110 --beta --train_budget 'low'
```
# Evaluation
The GAMA-PGD-100 evaluation code is provided in eval.py.
For evaluation of the trained model: 
```
python eval.py --trained_model 'PATH OF TRAINED MODEL' 
```
Further all the running details are provided in run.sh. It is recommended to use this file for training and evaluation of DAJAT.

# Results


Results obtained using higher number of attack steps and 200 epochs for training:
<p float="left">
  <img src="/DAJAT_200.png" width="600" />
</p>

# Citing this work
```
@inproceedings{jain2023dart,
  title={DART: Diversify-Aggregate-Repeat Training Improves Generalization of Neural Networks},
  author={Jain, Samyak and Addepalli, Sravanti and Sahu, Pawan Kumar and Dey, Priyam and Babu, R Venkatesh},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16048--16059},
  year={2023}
}
```
