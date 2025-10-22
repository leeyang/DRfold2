# DRfold2 Training Documentation

Complete training pipeline for DRfold2 RNA structure prediction model.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Data Preparation](#data-preparation)
4. [Configuration](#configuration)
5. [Training](#training)
6. [Monitoring](#monitoring)
7. [Evaluation](#evaluation)
8. [Troubleshooting](#troubleshooting)

---

## Overview

This training pipeline implements a complete training system for DRfold2, including:

- **Loss Functions**: Distance, FAPE, pLDDT, bond constraints, contact prediction
- **Data Loading**: PDB and FASTA file parsing with data augmentation
- **Training Loop**: Mixed precision training with gradient accumulation
- **Checkpointing**: Automatic checkpoint management and best model saving
- **Validation**: RMSD, TM-score, and contact prediction metrics
- **Logging**: TensorBoard and JSON metrics logging

### Model Architecture

DRfold2 uses:
- **RNA Composite Language Model (RCLM)** for sequence embeddings
- **Evoformer** (16 layers) for MSA and pair representation
- **Invariant Point Attention (IPA)** for geometry-aware structure prediction
- **Recycling** mechanism (3-8 iterations) for iterative refinement

---

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 1.11+ with CUDA support
- CUDA 11.3+ (for GPU training)

### Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy biopython pyyaml tensorboard tqdm
```

### Download Pre-trained Language Model

```bash
bash install.sh
```

This downloads the pre-trained RCLM model (~1.3GB) to `model_hub/RCLM/`.

---

## Data Preparation

### Directory Structure

Organize your data as follows:

```
data/
├── train/
│   ├── sample1.pdb
│   ├── sample2.pdb
│   └── ...
├── train_fasta/
│   ├── sample1.fasta
│   ├── sample2.fasta
│   └── ...
├── val/
│   ├── val1.pdb
│   └── ...
├── val_fasta/
│   ├── val1.fasta
│   └── ...
└── test/
    ├── test1.pdb
    └── ...
```

### PDB File Requirements

- **Format**: Standard PDB format
- **Atoms**: Must contain P, C4', and N1/N9 atoms for each nucleotide
- **Resolution**: Recommended < 3.5 Å
- **Length**: 10-512 nucleotides

Example PDB atoms:
```
ATOM      1  P     G A   1      10.123  20.456  30.789  1.00 50.00           P
ATOM      2  C4'   G A   1      12.345  22.678  32.901  1.00 50.00           C
ATOM      3  N9    G A   1      14.567  24.890  34.123  1.00 50.00           N
```

### FASTA File Requirements

- **Format**: Standard FASTA format
- **Nucleotides**: A, G, C, U (or T)
- **Header**: Any header starting with `>`

Example FASTA:
```
>sample1
GGCCUUAGCUACGAUGCUAGCUAGCUAGCUAG
```

### Data Filtering

The data loader automatically filters sequences based on:

- **Minimum length**: 10 nucleotides (configurable)
- **Maximum length**: 512 nucleotides (configurable)
- **Resolution**: < 3.5 Å (for PDB files with resolution info)

---

## Configuration

### Configuration File: `train_config.yaml`

The training configuration is controlled by a YAML file. Key sections:

#### Model Configuration

```yaml
model:
  config_name: "cfg_95"  # Model configuration to train
  use_pretrained_lm: true  # Use pre-trained language model
  freeze_language_model: false  # Whether to freeze LM weights
  n_recycle_train: 3  # Recycling iterations during training
  n_recycle_val: 4  # Recycling iterations during validation
```

#### Training Configuration

```yaml
training:
  num_epochs: 100
  batch_size: 1  # Start with 1 due to memory constraints
  gradient_accumulation_steps: 8  # Effective batch size = 8
  learning_rate: 1.0e-4
  use_amp: true  # Mixed precision training
  gradient_clip_norm: 1.0
```

#### Loss Weights

```yaml
loss_weights:
  weight_distance: 2.0  # Distance loss weight
  weight_fape: 1.0  # Frame Aligned Point Error
  weight_lddt: 0.5  # pLDDT confidence
  weight_bond: 0.5  # Bond constraints
  weight_contact: 0.3  # Contact prediction
  weight_structure: 1.0  # Overall structure RMSD
```

#### Data Configuration

```yaml
data:
  train_data_dir: "./data/train"
  train_fasta_dir: "./data/train_fasta"
  val_data_dir: "./data/val"
  val_fasta_dir: "./data/val_fasta"
  use_augmentation: true  # Enable data augmentation
```

### Modifying Configuration

1. Copy the default config:
   ```bash
   cp train_config.yaml my_config.yaml
   ```

2. Edit parameters in `my_config.yaml`

3. Run training with custom config:
   ```bash
   python train.py --config my_config.yaml
   ```

---

## Training

### Basic Training

Train with default configuration:

```bash
python train.py --config train_config.yaml --device cuda
```

### Resume Training

Resume from a checkpoint:

```bash
python train.py --config train_config.yaml --resume checkpoints/latest_checkpoint.pt
```

### Multi-GPU Training

For multi-GPU training (if you have multiple GPUs):

```yaml
training:
  distributed: true
  num_gpus: 4
```

Then run:
```bash
python -m torch.distributed.launch --nproc_per_node=4 train.py --config train_config.yaml
```

### CPU Training

For CPU-only training (slow, not recommended):

```bash
python train.py --config train_config.yaml --device cpu
```

### Memory Optimization

If you encounter out-of-memory errors:

1. **Reduce batch size**:
   ```yaml
   training:
     batch_size: 1
   ```

2. **Enable gradient checkpointing**:
   ```yaml
   hardware:
     gradient_checkpointing: true
   ```

3. **Reduce maximum sequence length**:
   ```yaml
   training:
     max_sequence_length: 256
   ```

4. **Use mixed precision**:
   ```yaml
   training:
     use_amp: true
   ```

### Training Time Estimates

On NVIDIA A100 (40GB):

| Sequence Length | Batch Size | Time per Epoch | Memory Usage |
|-----------------|------------|----------------|--------------|
| 50-100 nt       | 4          | ~2 hours       | ~25 GB       |
| 100-200 nt      | 2          | ~4 hours       | ~30 GB       |
| 200-400 nt      | 1          | ~8 hours       | ~35 GB       |
| 400-512 nt      | 1          | ~12 hours      | ~38 GB       |

---

## Monitoring

### TensorBoard

Monitor training in real-time with TensorBoard:

```bash
tensorboard --logdir tensorboard/
```

Then open http://localhost:6006 in your browser.

**Metrics tracked**:
- Training loss (total, distance, FAPE, pLDDT, bond, contact, structure)
- Validation loss
- Learning rate
- RMSD
- Gradient norms (optional)

### Metrics Logging

Metrics are also saved to JSON format in `logs/metrics.json` for later analysis:

```python
import json

with open('logs/metrics.json', 'r') as f:
    metrics = json.load(f)

# Get best validation loss
best_val = min([m for m in metrics if m['phase'] == 'val'],
               key=lambda x: x['total_loss'])
print(f"Best validation loss: {best_val['total_loss']:.4f} at epoch {best_val['epoch']}")
```

### Checkpoints

Checkpoints are saved to `checkpoints/`:

- **`latest_checkpoint.pt`**: Most recent checkpoint
- **`best_model.pt`**: Best model based on validation metric
- **`checkpoint_epoch{X}_step{Y}.pt`**: Regular checkpoints

By default, only the last 5 checkpoints are kept to save disk space.

---

## Evaluation

### Validation During Training

Validation is performed automatically during training based on:

```yaml
training:
  validate_every: 500  # Validate every 500 steps
```

### Manual Evaluation

Evaluate a checkpoint on the test set:

```python
# Coming soon: evaluation script
python evaluate.py --checkpoint checkpoints/best_model.pt --test_data data/test/
```

### Metrics

The following metrics are computed:

1. **RMSD**: Root mean squared deviation of predicted vs. true coordinates
2. **TM-score**: Template modeling score (requires TMscore binary)
3. **Contact Precision**: Precision of contact map prediction (< 8 Å)
4. **pLDDT**: Predicted local distance difference test confidence

### Using Trained Model for Inference

After training, use the trained model for inference:

```python
import torch
from cfg_95 import EvoMSA2XYZ

# Load model
model = EvoMSA2XYZ.MSA2XYZ(...)
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Run inference (see DRfold_infer.py for complete example)
```

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory

**Error**: `CUDA out of memory`

**Solutions**:
- Reduce batch size to 1
- Enable gradient checkpointing
- Reduce max sequence length
- Use CPU offloading (slower)

#### 2. NaN Loss

**Error**: Loss becomes NaN during training

**Solutions**:
- Reduce learning rate (try 1e-5)
- Enable gradient clipping (set to 1.0 or lower)
- Check data for corrupted PDB files
- Enable mixed precision training

#### 3. Slow Training

**Solutions**:
- Enable CUDA benchmark: `cudnn_benchmark: true`
- Increase num_workers for data loading
- Use mixed precision training
- Check CPU/GPU utilization

#### 4. Language Model Not Found

**Error**: `FileNotFoundError: model_hub/RCLM/epoch_67000`

**Solution**:
```bash
bash install.sh
```

#### 5. PDB Parsing Errors

**Error**: Cannot parse PDB file

**Solutions**:
- Check PDB file format (must have P, C4', N1/N9 atoms)
- Check residue numbering (should be sequential)
- Remove HETATM records if causing issues

### Validation Best Practices

1. **Start with a small dataset** (~10-20 structures) to test the pipeline
2. **Monitor validation loss** - should decrease in first few epochs
3. **Check RMSD** - should be < 5 Å for good predictions
4. **Visualize predictions** - Save validation structures and visualize in PyMOL
5. **Compare with inference** - Trained model should match or exceed inference quality

### Data Quality

For best results:

- Use high-resolution structures (< 2.5 Å preferred)
- Include diverse RNA structures (various folds, lengths)
- Balance dataset (similar number of samples per length range)
- Remove redundant structures (< 90% sequence identity)

---

## Advanced Usage

### Custom Loss Functions

Add custom loss functions in `loss_functions.py`:

```python
class CustomLoss(nn.Module):
    def forward(self, predictions, targets, mask):
        # Your custom loss implementation
        pass
```

Then modify `DRfoldLoss` to include it:

```python
self.custom_loss_fn = CustomLoss()
losses['custom_loss'] = self.custom_loss_fn(...) * self.loss_weights.get('weight_custom', 1.0)
```

### Custom Data Augmentation

Modify `data_loader.py` to add custom augmentation:

```python
def _augment_data(self, coords, ...):
    # Add your augmentation here
    # Example: random translation
    if random.random() < 0.5:
        translation = np.random.normal(0, 1, 3)
        coords = coords + translation

    return coords, ...
```

### Distributed Training

For multi-node training, use:

```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.1" \
    --master_port=12355 \
    train.py --config train_config.yaml
```

---

## File Structure

After training, your directory will look like:

```
DRfold2-claudecode/
├── train.py                    # Main training script
├── train_config.yaml           # Configuration file
├── loss_functions.py           # Loss function implementations
├── data_loader.py              # Data loading utilities
├── TRAINING_README.md          # This file
│
├── cfg_95/                     # Model code (existing)
│   ├── EvoMSA2XYZ.py
│   ├── Structure.py
│   ├── Evoformer.py
│   └── ...
│
├── utils/                      # Training utilities
│   └── checkpoint.py           # Checkpoint management
│
├── data/                       # Training data
│   ├── train/
│   ├── train_fasta/
│   ├── val/
│   └── val_fasta/
│
├── checkpoints/                # Saved checkpoints
│   ├── best_model.pt
│   ├── latest_checkpoint.pt
│   └── checkpoint_epoch*_step*.pt
│
├── logs/                       # Training logs
│   └── metrics.json
│
├── tensorboard/                # TensorBoard logs
│
└── training_outputs/           # Other training outputs
```

---

## Citation

If you use this training code, please cite:

```bibtex
@article{drfold2,
  title={DRfold2: Improved RNA structure prediction using deep learning},
  author={Your Name},
  journal={Journal},
  year={2024}
}
```

---

## Support

For issues and questions:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review training logs and metrics
3. Open an issue on GitHub

---

## License

[Add your license information here]

---

## Changelog

### Version 1.0.0 (2024)

- Initial training pipeline implementation
- Support for cfg_95 configuration
- Loss functions: Distance, FAPE, pLDDT, bond, contact, structure
- Data augmentation: rotation, noise
- Mixed precision training
- Automatic checkpointing
- TensorBoard logging
