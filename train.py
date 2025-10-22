"""
Main Training Script for DRfold2
Complete training pipeline for RNA structure prediction
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
import random
from tqdm import tqdm
import time

# Add cfg_95 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cfg_95'))

# Import model components
import EvoMSA2XYZ
import data as data_utils

# Import training utilities
from loss_functions import DRfoldLoss, compute_rmsd, compute_contact_map, compute_distance_matrix
from data_loader import create_data_loaders
from utils.checkpoint import CheckpointManager, MetricsLogger, save_config


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict, device: str) -> nn.Module:
    """
    Create DRfold2 model.

    Args:
        config: Configuration dictionary
        device: Device to load model on

    Returns:
        Model instance
    """
    model_config = config['model']

    # Model parameters (matching test_modeldir.py)
    seq_dim = 6  # RNA nucleotides + 1
    msa_dim = 6 + 1  # With masking
    m_dim = model_config['m_dim']
    s_dim = model_config['s_dim']
    z_dim = model_config['z_dim']
    N_ensemble = 3  # Number of ensemble predictions
    N_cycle = model_config['n_recycle_train']

    # Create model
    model = EvoMSA2XYZ.MSA2XYZ(
        seq_dim=seq_dim - 1,
        msa_dim=msa_dim,
        N_ensemble=N_ensemble,
        N_cycle=N_cycle,
        m_dim=m_dim,
        s_dim=s_dim,
        z_dim=z_dim
    )

    model = model.to(device)

    # Optionally freeze language model
    if model_config.get('freeze_language_model', False):
        print("Freezing RNA language model parameters")
        # The language model (RNAlm) is loaded in EvoMSA2XYZ and should be frozen
        for name, param in model.named_parameters():
            if 'lm' in name.lower() or 'rnalm' in name.lower():
                param.requires_grad = False

    return model


def create_optimizer(model: nn.Module, config: dict) -> optim.Optimizer:
    """Create optimizer"""
    training_config = config['training']

    optimizer_name = training_config['optimizer'].lower()
    lr = training_config['learning_rate']
    weight_decay = training_config['weight_decay']

    if optimizer_name == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(training_config['adam_beta1'], training_config['adam_beta2']),
            eps=training_config['adam_eps'],
            weight_decay=weight_decay
        )
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(training_config['adam_beta1'], training_config['adam_beta2']),
            eps=training_config['adam_eps'],
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return optimizer


def create_scheduler(optimizer: optim.Optimizer, config: dict):
    """Create learning rate scheduler"""
    training_config = config['training']

    scheduler_name = training_config['lr_scheduler'].lower()
    num_epochs = training_config['num_epochs']

    if scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=training_config['min_lr']
        )
    elif scheduler_name == 'linear':
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=training_config['min_lr'] / training_config['learning_rate'],
            total_iters=num_epochs
        )
    elif scheduler_name == 'constant':
        scheduler = optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=num_epochs
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    return scheduler


def prepare_batch(batch: dict, device: str) -> dict:
    """Move batch to device and prepare for model input"""
    prepared_batch = {}

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            prepared_batch[key] = value.to(device)
        else:
            prepared_batch[key] = value

    return prepared_batch


def train_step(model: nn.Module,
               batch: dict,
               loss_fn: DRfoldLoss,
               optimizer: optim.Optimizer,
               scaler: GradScaler,
               config: dict,
               device: str) -> dict:
    """
    Perform a single training step.

    Returns:
        Dictionary of loss values
    """
    model.train()

    # Prepare batch
    batch = prepare_batch(batch, device)

    # Extract data
    seq_encoding = batch['seq_encoding'][0]  # Remove batch dimension for now (batch size = 1)
    coords = batch['coords'][0]
    mask = batch['mask'][0]
    base_atoms = batch['base_atoms'][0]
    sequence = batch['sequence'][0]

    # Prepare MSA (duplicate sequence as in test_modeldir.py)
    msa = torch.nn.functional.one_hot(seq_encoding.long(), 6).float()
    msa = torch.stack([msa, msa], dim=0)  # Create dummy MSA

    # Prepare seq_idx
    seq_idx = torch.arange(len(sequence)) + 1
    seq_idx = seq_idx.to(device)

    # Forward pass with mixed precision
    use_amp = config['training']['use_amp']

    with autocast(enabled=use_amp):
        # Note: The model's forward method expects (msa, idx, ss, true_x, base_x, alphas)
        # ss (secondary structure) can be None for now
        # We need to modify the forward call to work with our training setup

        # For now, call the model's internal components
        # Get embeddings and structure prediction
        L = seq_encoding.shape[0]
        m1_pre, z_pre = 0, 0
        x_pre = torch.zeros(L, 3, 3).to(device)

        # Run one iteration through the model
        m1, z, s, _, _, _ = model.msaxyzone(msa, seq_idx, None, m1_pre, z_pre, x_pre, 0, list(sequence))

        # Get structure prediction
        pred_coords, rotation, translation, plddt = model.structurenet.pred(s, z, base_atoms)

        # Prepare predictions and targets for loss computation
        predictions = {
            'coords': pred_coords,
            'rotation': rotation,
            'translation': translation,
            'lddt_logits': None,  # Not available in this forward pass
            'pair_repr': z
        }

        targets = {
            'coords': coords,
            'rotation': batch['rotation'][0],
            'translation': batch['translation'][0],
            'lddt_dist': batch['distance_matrix'][0],
            'contacts': batch['contact_map'][0]
        }

        # Compute losses
        losses = loss_fn(predictions, targets, mask)

    # Backward pass
    optimizer.zero_grad()

    if use_amp:
        scaler.scale(losses['total_loss']).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip_norm'])
        scaler.step(optimizer)
        scaler.update()
    else:
        losses['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip_norm'])
        optimizer.step()

    # Convert losses to dict of floats
    loss_dict = {k: v.item() for k, v in losses.items()}

    return loss_dict


@torch.no_grad()
def validate(model: nn.Module,
            val_loader,
            loss_fn: DRfoldLoss,
            config: dict,
            device: str) -> dict:
    """
    Validate the model.

    Returns:
        Dictionary of validation metrics
    """
    model.eval()

    total_losses = {}
    num_batches = 0

    for batch in tqdm(val_loader, desc="Validating", leave=False):
        batch = prepare_batch(batch, device)

        # Extract data (similar to train_step)
        seq_encoding = batch['seq_encoding'][0]
        coords = batch['coords'][0]
        mask = batch['mask'][0]
        base_atoms = batch['base_atoms'][0]
        sequence = batch['sequence'][0]

        # Prepare MSA
        msa = torch.nn.functional.one_hot(seq_encoding.long(), 6).float()
        msa = torch.stack([msa, msa], dim=0)

        # Prepare seq_idx
        seq_idx = torch.arange(len(sequence)) + 1
        seq_idx = seq_idx.to(device)

        # Forward pass
        L = seq_encoding.shape[0]
        m1_pre, z_pre = 0, 0
        x_pre = torch.zeros(L, 3, 3).to(device)

        # Run through model
        m1, z, s = model.msaxyzone.pred(msa, seq_idx, None, m1_pre, z_pre, x_pre, 0, list(sequence))
        pred_coords, rotation, translation, plddt = model.structurenet.pred(s, z, base_atoms)

        # Prepare predictions and targets
        predictions = {
            'coords': pred_coords,
            'rotation': rotation,
            'translation': translation,
            'pair_repr': z
        }

        targets = {
            'coords': coords,
            'rotation': batch['rotation'][0],
            'translation': batch['translation'][0],
            'lddt_dist': batch['distance_matrix'][0],
            'contacts': batch['contact_map'][0]
        }

        # Compute losses
        losses = loss_fn(predictions, targets, mask)

        # Accumulate losses
        for k, v in losses.items():
            if k not in total_losses:
                total_losses[k] = 0
            total_losses[k] += v.item()

        # Compute RMSD
        rmsd = compute_rmsd(pred_coords, coords, mask)
        if 'rmsd' not in total_losses:
            total_losses['rmsd'] = 0
        total_losses['rmsd'] += rmsd

        num_batches += 1

    # Average losses
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}

    return avg_losses


def train(config: dict, resume_from: str = None):
    """
    Main training loop.

    Args:
        config: Configuration dictionary
        resume_from: Optional checkpoint path to resume from
    """
    # Set random seed
    set_seed(config['reproducibility']['seed'])

    # Setup device
    device = config['hardware']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    print(f"Using device: {device}")

    # Create output directories
    output_dir = Path(config['output']['output_dir'])
    checkpoint_dir = Path(config['output']['checkpoint_dir'])
    log_dir = Path(config['output']['log_dir'])
    tensorboard_dir = Path(config['output']['tensorboard_dir'])

    for dir_path in [output_dir, checkpoint_dir, log_dir, tensorboard_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Save config
    save_config(config, str(output_dir / 'config.json'))

    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(config)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    print("Creating model...")
    model = create_model(config, device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    # Create loss function
    loss_fn = DRfoldLoss(config)

    # Create gradient scaler for mixed precision
    scaler = GradScaler(enabled=config['training']['use_amp'])

    # Create checkpoint manager and metrics logger
    checkpoint_manager = CheckpointManager(
        str(checkpoint_dir),
        keep_last_n=config['training']['keep_last_n_checkpoints']
    )
    metrics_logger = MetricsLogger(str(log_dir))

    # Create tensorboard writer
    writer = SummaryWriter(str(tensorboard_dir))

    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0

    if resume_from:
        checkpoint_info = checkpoint_manager.load_checkpoint(
            resume_from,
            model,
            optimizer,
            scheduler,
            load_optimizer=config['resume']['load_optimizer_state'],
            load_scheduler=config['resume']['load_scheduler_state'],
            device=device
        )
        if not config['resume']['reset_epoch']:
            start_epoch = checkpoint_info['epoch']
            global_step = checkpoint_info['step']
        print(f"Resumed from epoch {start_epoch}, step {global_step}")

    # Training loop
    print("Starting training...")
    num_epochs = config['training']['num_epochs']

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Training phase
        model.train()
        epoch_losses = {}
        num_train_steps = 0

        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(pbar):
            try:
                # Training step
                loss_dict = train_step(model, batch, loss_fn, optimizer, scaler, config, device)

                # Accumulate losses
                for k, v in loss_dict.items():
                    if k not in epoch_losses:
                        epoch_losses[k] = 0
                    epoch_losses[k] += v

                num_train_steps += 1
                global_step += 1

                # Update progress bar
                pbar.set_postfix({'loss': f"{loss_dict['total_loss']:.4f}"})

                # Log to tensorboard
                if global_step % config['training']['log_every'] == 0:
                    for k, v in loss_dict.items():
                        writer.add_scalar(f'train/{k}', v, global_step)
                    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)

                # Log to metrics logger
                if global_step % config['training']['log_every'] == 0:
                    metrics_logger.log(epoch, global_step, loss_dict, phase='train')

                # Validate
                if global_step % config['training']['validate_every'] == 0:
                    print("\nRunning validation...")
                    val_metrics = validate(model, val_loader, loss_fn, config, device)

                    # Log validation metrics
                    for k, v in val_metrics.items():
                        writer.add_scalar(f'val/{k}', v, global_step)

                    metrics_logger.log(epoch, global_step, val_metrics, phase='val')

                    print(f"Validation - Loss: {val_metrics['total_loss']:.4f}, RMSD: {val_metrics.get('rmsd', 0):.4f}")

                # Save checkpoint
                if global_step % config['training']['checkpoint_every'] == 0:
                    # Check if this is the best model
                    is_best = False
                    if config['training']['save_best_model']:
                        val_metrics = validate(model, val_loader, loss_fn, config, device)
                        metric_value = val_metrics[config['training']['validation_metric']]
                        is_best = checkpoint_manager.update_best_metric(metric_value, lower_is_better=True)

                    checkpoint_manager.save_checkpoint(
                        epoch=epoch,
                        step=global_step,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        metrics=val_metrics if is_best else None,
                        is_best=is_best,
                        config=config
                    )

            except Exception as e:
                print(f"Error in training step: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Epoch end
        avg_epoch_losses = {k: v / num_train_steps for k, v in epoch_losses.items()}
        print(f"Epoch {epoch + 1} - Train Loss: {avg_epoch_losses['total_loss']:.4f}")

        # End of epoch validation
        print("Running end-of-epoch validation...")
        val_metrics = validate(model, val_loader, loss_fn, config, device)
        print(f"Epoch {epoch + 1} - Val Loss: {val_metrics['total_loss']:.4f}, RMSD: {val_metrics.get('rmsd', 0):.4f}")

        # Update scheduler
        scheduler.step()

        # Save checkpoint at end of epoch
        is_best = checkpoint_manager.update_best_metric(
            val_metrics[config['training']['validation_metric']],
            lower_is_better=True
        )

        checkpoint_manager.save_checkpoint(
            epoch=epoch + 1,
            step=global_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=val_metrics,
            is_best=is_best,
            config=config
        )

    print("\nTraining completed!")
    writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train DRfold2 RNA structure prediction model')
    parser.add_argument('--config', type=str, default='train_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (overrides config)')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override device if specified
    if args.device:
        config['hardware']['device'] = args.device

    # Temporarily modify sys.argv for EvoMSA2XYZ import
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0], config['hardware']['device']]

    try:
        # Run training
        train(config, resume_from=args.resume)
    finally:
        # Restore original sys.argv
        sys.argv = original_argv


if __name__ == '__main__':
    main()
