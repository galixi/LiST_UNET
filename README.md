# LiST-UNet: A Spatiotemporal Dependency-Aware Lightweight CNN-ViT Network for 3D MRF with a Balanced Acceleration Strategy

## Overview

Official PyTorch implementation of **LiST-UNet**, a lightweight hybrid CNN-ViT network for 3D Magnetic Resonance Fingerprinting (MRF) parameter map reconstruction. LiST-UNet reconstructs quantitative T1/T2 parameter maps from undersampled MRF acquisitions using a U-Net-style encoder-decoder architecture equipped with a novel Spatio-Temporal (ST) attention mechanism.

**Key features:**
- Lightweight CNN-ViT hybrid block combining local depthwise convolutions and global spatiotemporal attention
- 3D volumetric processing preserving full spatial context across all slice directions
- Balanced acceleration strategy enabling fast, high-fidelity MRF reconstruction
- Encoder-decoder design with multi-scale skip connections for detail recovery

---

## Method

### Architecture

LiST-UNet follows a symmetric U-Net encoder-decoder structure operating on full 3D volumes.

```
Input (5, 192, 192, 128)
        |
   [Encoder Level 1]  DepthwiseConv Downsample ×2  →  (32, 96, 96, 64)   ─┐ skip
   [Encoder Level 2]  DepthwiseConv Downsample ×2  →  (64, 48, 48, 32)   ─┤ skip
   [Encoder Level 3]  DepthwiseConv Downsample ×2  →  (128, 24, 24, 16)  ─┤ skip
   [Encoder Level 4]  DepthwiseConv Downsample ×2  →  (96, 12, 12, 8)    ─┤ skip
        |                Flatten + Positional Embedding
   [Bottleneck]
        |
   [Decoder Level 1]  TransConv Upsample ×2  + skip  →  (128, 24, 24, 16)
   [Decoder Level 2]  TransConv Upsample ×2  + skip  →  (64, 48, 48, 32)
   [Decoder Level 3]  TransConv Upsample ×2  + skip  →  (32, 96, 96, 64)
   [Decoder Level 4]  Reconstruction Head           →  (1, 192, 192, 128)
        |
Output T1 Map (1, 192, 192, 128)
```

### Core Block (LiST-UNet Block)

Each block in the encoder/decoder combines local and global feature extraction with residual connections:

- Local representations (grouped convolutions) → LineConv MLP
- Spatio-Temporal Attention → Local Reverse Diffusion → LineConv MLP

### Spatio-Temporal Attention (ST_Attention)

The ST attention module decouples spatial and temporal dependencies:
- **Temporal path**: Captures dependencies across the MRF time-series dimension via average-pooled queries and keys
- **Spatial path**: Captures intra-slice spatial correlations
- **Fusion**: Element-wise addition of temporal and spatial attention outputs

---

## Repository Structure

```
LiST_UNET/
├── train.py                # Training and inference pipeline
├── LiST_UNET.py           # Top-level model (RecEncoder + RecDecoder)
├── Encoder.py              # RecEncoder: 4-level downsampling encoder
├── Decoder.py              # RecDecoder: 4-level upsampling decoder
├── LiST_UNET_Block.py     # Core LiST block, ST_Attention, helper modules
├── loss.py                 # Loss functions (L1+SSIM, L2, etc.)
├── optimizer.py            # Linear warmup + cosine annealing scheduler
├── mydataset_3DMRF.py     # PyTorch Dataset for 3D MRF .mat/.nii data
└── pytorch_ssim/
    └── __init__.py         # 2D/3D SSIM metric implementation
```

---

## Requirements

```bash
pip install torch torchvision torchaudio
pip install numpy scipy h5py nibabel
```

| Package | Purpose |
|---------|---------|
| `torch` | Model training |
| `numpy` | Array operations |
| `scipy` | Loading `.mat` ground-truth files |
| `h5py` | Loading HDF5-format MRF input files |
| `nibabel` | Loading NIfTI brain masks |

---

## Data Preparation

The dataset loader (`mydataset_3DMRF.py`) expects the following directory layout:

```
<root_dir>/
├── <subject_0>/
│   └── 48gro_1000/
│       ├── T1T2.mat             # Ground truth T1/T2 maps
│       └── T1w_synthseg.nii    # Binary brain mask (NIfTI)
├── <subject_1>/
│   └── ...

<root_dir>_48_500/
├── <subject_0>/
│   └── 48g_500tr/
│       └── Rec_dl_input.mat    # Undersampled MRF reconstruction (HDF5)
├── <subject_1>/
│   └── ...
```

### File Formats

| File | Format | Content |
|------|--------|---------|
| `Rec_dl_input.mat` | HDF5 `.mat` | Complex MRF data: `Rec/real`, `Rec/imag` — shape `(5, H, W, T)` |
| `T1T2.mat` | MATLAB `.mat` | `T1_find_all` (ms), `T2_find_all` (ms) |
| `T1w_synthseg.nii` | NIfTI | Binary brain mask |

### Preprocessing (applied automatically)

- Crop from 220×220×220 to **192×192×128**
- Per-channel normalization to [0, 1] within brain mask
- Label normalization: T1 ÷ 4000, T2 ÷ 300

---

## Training

```bash
python train.py \
  --root_dir /path/to/dataset \
  --device cuda:0 \
  --batch_size 1 \
  --max_epochs 300 \
  --loss_name L1retest \
  --ckpt_dir /path/to/checkpoints \
  --result_dir /path/to/results \
  --person 0
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--root_dir` | — | Root path of the dataset (required) |
| `--device` | `cuda:0` | Training device |
| `--batch_size` | `1` | Batch size |
| `--max_epochs` | `300` | Total training epochs |
| `--loss_name` | `L1retest` | Loss function: `L1retest`, `L2re`, `ssim`, `L1.5re` |
| `--ckpt_dir` | — | Directory to save model checkpoints |
| `--result_dir` | — | Directory to save inference outputs |

### Optimizer & Scheduler

- **Optimizer**: AdamW (`lr=1e-3`, `betas=(0.9, 0.95)`, `weight_decay=0.05`)
- **Scheduler**: Linear warmup (5 epochs) + cosine annealing to 0

### Loss Function

The default loss `L1retest` is:

```
L = L1(pred, target) + (1 - SSIM3D(pred, target))
```

computed only within the brain mask region.

---

## Outputs

After each epoch, the following files are saved:

```
<ckpt_dir>/
└── modelpara_{epoch}.pth          # Model weights

<result_dir>/<person>/<epoch>/
├── loss.mat                        # Per-sample loss values
└── result.mat                      # Predictions, labels, and masks
```

`result.mat` contains:
- `input`: Model predictions (T1 map)
- `label`: Ground truth T1 map
- `mask`: Brain mask

---

## Model Configuration

| Hyperparameter | Value |
|---------------|-------|
| Input channels | 5 |
| Output channels | 1 |
| Embed dim | 96 |
| Channel progression | (32, 64, 128) |
| Blocks per level | (1, 2, 3, 2) |
| Attention heads | (1, 2, 4, 4) |
| Reduction ratios | (4, 2, 2, 1) |
| Dropout | 0.3 |

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{LiST-UNet2024,
  title   = {A Spatiotemporal Dependency-Aware Lightweight CNN-ViT Network for 3D MRF with a Balanced Acceleration Strategy},
  journal = {},
  year    = {2024}
}
```

---

## License

This project is for research purposes only.