# Experimental Results

## Dataset Performance

### FigShare Dataset
| Metric | Value |
|--------|-------|
| Dice Coefficient | 97.76% |
| Jaccard Index | 97.36% |
| Precision | 95.14% |
| Sensitivity | 93.42% |

### TCIA LGG Dataset
| Metric | Value |
|--------|-------|
| Dice Coefficient | 94.0% |
| Accuracy | 99.9% |
| Jaccard Index | 94.4% |
| Sensitivity | 90.01% |

## Comparison with State-of-the-Art

### FigShare Dataset
| Model | Dice Coefficient |
|-------|-----------------|
| U-Net | 92.5% |
| Attention U-Net | 95.1% |
| Res-U-Net | 96.2% |
| U-Net++ | 96.8% |
| **W-AGRU-Net (Ours)** | **97.76%** |

### TCIA Dataset
| Model | Dice Coefficient |
|-------|-----------------|
| U-Net | 89.3% |
| Attention U-Net | 91.5% |
| Res-U-Net | 92.8% |
| U-Net++ | 93.1% |
| **W-AGRU-Net (Ours)** | **94.0%** |

## Training Details
- Epochs: 100
- Batch Size: 2
- Optimizer: Adam (lr=1e-4)
- Loss: Dice Loss
- Framework: TensorFlow 2.8
