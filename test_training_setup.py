"""
Test script to validate training setup
Checks all components without running full training
"""

import os
import sys
import torch
import numpy as np
import yaml
from pathlib import Path

# Add cfg_95 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cfg_95'))

print("=" * 80)
print("DRfold2 Training Setup Validation")
print("=" * 80)

# Test 1: Import all modules
print("\n[1/8] Testing module imports...")
try:
    from loss_functions import DRfoldLoss, compute_rmsd, compute_contact_map
    from data_loader import RNAStructureDataset, create_data_loaders
    from utils.checkpoint import CheckpointManager, MetricsLogger
    print("✓ All modules imported successfully")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test 2: Load configuration
print("\n[2/8] Testing configuration loading...")
try:
    with open('train_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print("✓ Configuration loaded successfully")
    print(f"  - Model: {config['model']['config_name']}")
    print(f"  - Epochs: {config['training']['num_epochs']}")
    print(f"  - Learning rate: {config['training']['learning_rate']}")
except Exception as e:
    print(f"✗ Configuration error: {e}")
    sys.exit(1)

# Test 3: Test loss functions
print("\n[3/8] Testing loss functions...")
try:
    loss_fn = DRfoldLoss(config)

    # Create dummy data
    L = 20  # Sequence length
    pred_coords = torch.randn(L, 3, 3)
    true_coords = torch.randn(L, 3, 3)
    rotation = torch.eye(3).unsqueeze(0).repeat(L, 1, 1)
    translation = torch.randn(L, 3)
    mask = torch.ones(L)
    z = torch.randn(L, L, 64)

    predictions = {
        'coords': pred_coords,
        'rotation': rotation,
        'translation': translation,
        'pair_repr': z
    }

    targets = {
        'coords': true_coords,
        'rotation': rotation,
        'translation': translation,
        'lddt_dist': torch.cdist(true_coords[:, 1, :], true_coords[:, 1, :]),
        'contacts': (torch.cdist(true_coords[:, 1, :], true_coords[:, 1, :]) < 8.0).float()
    }

    losses = loss_fn(predictions, targets, mask)

    print("✓ Loss functions working")
    print(f"  - Total loss: {losses['total_loss'].item():.4f}")
    print(f"  - Distance loss: {losses.get('distance_loss', 0):.4f}")
    print(f"  - FAPE loss: {losses.get('fape_loss', 0):.4f}")

    # Test RMSD computation
    rmsd = compute_rmsd(pred_coords, true_coords, mask)
    print(f"  - RMSD: {rmsd:.4f}")

except Exception as e:
    print(f"✗ Loss function error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test checkpoint manager
print("\n[4/8] Testing checkpoint manager...")
try:
    checkpoint_dir = Path("test_checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint_manager = CheckpointManager(str(checkpoint_dir), keep_last_n=3)
    metrics_logger = MetricsLogger(str(checkpoint_dir))

    # Create dummy model
    dummy_model = torch.nn.Linear(10, 10)
    dummy_optimizer = torch.optim.Adam(dummy_model.parameters())

    # Test saving
    checkpoint_manager.save_checkpoint(
        epoch=0,
        step=0,
        model=dummy_model,
        optimizer=dummy_optimizer,
        metrics={'test_loss': 0.5}
    )

    # Test loading
    checkpoint_info = checkpoint_manager.load_latest_checkpoint(
        dummy_model,
        dummy_optimizer,
        device='cpu'
    )

    print("✓ Checkpoint manager working")
    print(f"  - Saved checkpoint at: {checkpoint_dir}")

    # Clean up
    import shutil
    shutil.rmtree(checkpoint_dir)

except Exception as e:
    print(f"✗ Checkpoint manager error: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Check model hub
print("\n[5/8] Checking model hub...")
try:
    model_hub_path = Path("model_hub/RCLM/epoch_67000")
    if model_hub_path.exists():
        print("✓ Pre-trained language model found")
        print(f"  - Path: {model_hub_path}")
    else:
        print("⚠ Pre-trained language model not found")
        print("  - Run: bash install.sh")
        print("  - Or download manually to model_hub/RCLM/")
except Exception as e:
    print(f"✗ Model hub error: {e}")

# Test 6: Check data directories
print("\n[6/8] Checking data directories...")
try:
    data_dirs = [
        config['data']['train_data_dir'],
        config['data']['train_fasta_dir'],
        config['data']['val_data_dir'],
        config['data']['val_fasta_dir']
    ]

    all_exist = True
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            num_files = len(os.listdir(data_dir))
            print(f"  ✓ {data_dir} exists ({num_files} files)")
        else:
            print(f"  ⚠ {data_dir} does not exist")
            all_exist = False

    if all_exist:
        print("✓ All data directories exist")
    else:
        print("⚠ Some data directories missing - create them before training")
        print("  - See TRAINING_README.md for data preparation")

except Exception as e:
    print(f"✗ Data directory error: {e}")

# Test 7: Test CUDA availability
print("\n[7/8] Checking CUDA availability...")
try:
    if torch.cuda.is_available():
        print("✓ CUDA is available")
        print(f"  - Device: {torch.cuda.get_device_name(0)}")
        print(f"  - CUDA version: {torch.version.cuda}")
        print(f"  - Number of GPUs: {torch.cuda.device_count()}")

        # Test CUDA memory
        device = torch.device('cuda:0')
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
        print(f"  - GPU memory: {total_memory:.1f} GB")

        if total_memory < 16:
            print("  ⚠ Warning: GPU has less than 16 GB memory")
            print("    - Consider using batch_size: 1 and gradient accumulation")
    else:
        print("⚠ CUDA not available - training will use CPU (very slow)")
        print("  - Install CUDA-enabled PyTorch for GPU training")
except Exception as e:
    print(f"✗ CUDA check error: {e}")

# Test 8: Test model creation (basic)
print("\n[8/8] Testing model structure...")
try:
    # Import model components
    import EvoMSA2XYZ

    # Create small model for testing
    seq_dim = 6
    msa_dim = 7
    m_dim = 32  # Reduced for testing
    s_dim = 32
    z_dim = 32
    N_ensemble = 1
    N_cycle = 1

    # Temporarily set device
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0], 'cpu']

    model = EvoMSA2XYZ.MSA2XYZ(
        seq_dim=seq_dim - 1,
        msa_dim=msa_dim,
        N_ensemble=N_ensemble,
        N_cycle=N_cycle,
        m_dim=m_dim,
        s_dim=s_dim,
        z_dim=z_dim
    )

    sys.argv = original_argv

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("✓ Model created successfully")
    print(f"  - Total parameters: {num_params:,}")
    print(f"  - Trainable parameters: {num_trainable:,}")

except Exception as e:
    print(f"✗ Model creation error: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 80)
print("Validation Complete")
print("=" * 80)

print("\nNext steps:")
print("1. Prepare your training data (see TRAINING_README.md)")
print("2. Adjust train_config.yaml for your setup")
print("3. Run training: python train.py --config train_config.yaml --device cuda")

print("\nFor more information, see TRAINING_README.md")
