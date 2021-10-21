# Adversarial Attacks

### Environments
Currently, requires following packages
- python 3.6+
- torch 1.7.1+
- torchvision 0.5+
- CUDA 10.1+
- scikit-learn 0.22+
- [autoattack](https://github.com/fra31/auto-attack)

### Training
To train the models, run this command:
```train
CUDA_VISIBLE_DEVICES=0 python train_fgsm.py --lr-max <lr> --prob-p <APR-P> --prob-s <APR-S> --epochs <epochs> --out-dir <out dir>
```
**Optimal hyper-parameters:**

| **Method**  | **lr-max** | **epochs** | **prob-p** | **prob-s** |
|-------------|------------|------------|------------|------------|
| APR-P       | 0.22       | 55         | 0.16       | 0.00       |
| APR-S       | 0.23       | 65         | 0.00       | 1.00       |
| APR-SP      | 0.20       | 90         | 0.16       | 0.90       |

### Evaluation
To evaluate the models, run this command:
```train
CUDA_VISIBLE_DEVICES=0 python eval.py --model <model path>
```

### Pretrained models
- Pretrained ResNet-18 are available:
1. [APR-P](https://drive.google.com/drive/folders/1ezfU3AZCGUDADoLwi-63NnuAWyhSQLaH?usp=sharing)
2. [APR-S](https://drive.google.com/drive/folders/1HuNrBsbdEu5MePPK8Bwt9bCbhFZZgH-S?usp=sharing)
3. [APR-SP](https://drive.google.com/drive/folders/1XEB5kU8-CWVlEY4mC7-XC5o1pGUzN4qs?usp=sharing)