"""
Loss Functions for DRfold2 Training
Implements all loss functions for RNA structure prediction training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


class DRfoldLoss(nn.Module):
    """
    Main loss module for DRfold2 training.
    Combines multiple loss functions with configurable weights.
    """

    def __init__(self, config: Dict):
        super(DRfoldLoss, self).__init__()
        self.config = config
        self.loss_weights = config.get('loss_weights', {})

        # Initialize individual loss components
        self.distance_loss_fn = DistanceLoss(
            weight_pp=self.loss_weights.get('weight_pp', 1.0),
            weight_cc=self.loss_weights.get('weight_cc', 1.0),
            weight_nn=self.loss_weights.get('weight_nn', 1.0)
        )

        self.fape_loss_fn = FAPELoss(
            clamp_distance=self.loss_weights.get('fape_clamp_distance', 10.0),
            loss_unit_distance=self.loss_weights.get('fape_loss_unit_distance', 10.0)
        )

        self.lddt_loss_fn = LDDTLoss()
        self.bond_loss_fn = BondLoss()
        self.contact_loss_fn = ContactLoss(
            contact_threshold=config.get('validation', {}).get('contact_threshold', 8.0)
        )

    def forward(self, predictions: Dict, targets: Dict, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute all losses.

        Args:
            predictions: Dictionary containing:
                - 'coords': Predicted coordinates (L, 3, 3) for P, C4', N atoms
                - 'rotation': Predicted rotations (L, 3, 3)
                - 'translation': Predicted translations (L, 3)
                - 'lddt_logits': pLDDT prediction logits (L, L, 5)
                - 'pair_repr': Pair representation for contact prediction (L, L, D)
            targets: Dictionary containing:
                - 'coords': Ground truth coordinates (L, 3, 3)
                - 'rotation': Ground truth rotations (L, 3, 3)
                - 'translation': Ground truth translations (L, 3)
                - 'lddt_dist': Ground truth distance matrix for LDDT
                - 'contacts': Ground truth contact map (L, L)
            mask: Optional sequence mask (L,)

        Returns:
            Dictionary of loss values
        """
        losses = {}

        # Get sequence length
        L = predictions['coords'].shape[0]

        # Create default mask if not provided
        if mask is None:
            mask = torch.ones(L, device=predictions['coords'].device)

        # 1. Distance Loss (for P, C4', N atoms)
        if 'coords' in predictions and 'coords' in targets:
            dist_loss = self.distance_loss_fn(
                predictions['coords'],
                targets['coords'],
                mask
            )
            losses['distance_loss'] = dist_loss * self.loss_weights.get('weight_distance', 2.0)

        # 2. FAPE Loss (Frame Aligned Point Error)
        if all(k in predictions for k in ['coords', 'rotation', 'translation']) and \
           all(k in targets for k in ['coords', 'rotation', 'translation']):
            fape_loss = self.fape_loss_fn(
                predictions['coords'],
                targets['coords'],
                predictions['rotation'],
                predictions['translation'],
                targets['rotation'],
                targets['translation'],
                mask
            )
            losses['fape_loss'] = fape_loss * self.loss_weights.get('weight_fape', 1.0)

        # 3. pLDDT Loss (confidence prediction)
        if 'lddt_logits' in predictions and 'lddt_dist' in targets:
            lddt_loss = self.lddt_loss_fn(
                predictions['lddt_logits'],
                targets['lddt_dist'],
                mask
            )
            losses['lddt_loss'] = lddt_loss * self.loss_weights.get('weight_lddt', 0.5)

        # 4. Bond/Angle/Torsion Constraints
        if 'coords' in predictions:
            bond_loss = self.bond_loss_fn(predictions['coords'], mask)
            losses['bond_loss'] = bond_loss * self.loss_weights.get('weight_bond', 0.5)

        # 5. Contact Prediction Loss
        if 'pair_repr' in predictions and 'contacts' in targets:
            contact_loss = self.contact_loss_fn(
                predictions['pair_repr'],
                targets['contacts'],
                mask
            )
            losses['contact_loss'] = contact_loss * self.loss_weights.get('weight_contact', 0.3)

        # 6. Structure RMSD Loss (optional, for direct coordinate supervision)
        if 'coords' in predictions and 'coords' in targets:
            structure_loss = self._compute_rmsd_loss(
                predictions['coords'],
                targets['coords'],
                mask
            )
            losses['structure_loss'] = structure_loss * self.loss_weights.get('weight_structure', 1.0)

        # Compute total loss
        losses['total_loss'] = sum(losses.values())

        return losses

    def _compute_rmsd_loss(self, pred_coords: torch.Tensor, true_coords: torch.Tensor,
                          mask: torch.Tensor) -> torch.Tensor:
        """
        Compute RMSD loss after optimal alignment.

        Args:
            pred_coords: Predicted coordinates (L, 3, 3)
            true_coords: True coordinates (L, 3, 3)
            mask: Sequence mask (L,)
        """
        # Flatten atom dimension
        pred_flat = pred_coords.reshape(-1, 3)  # (L*3, 3)
        true_flat = true_coords.reshape(-1, 3)  # (L*3, 3)

        # Expand mask for all atoms
        mask_flat = mask.unsqueeze(1).repeat(1, 3).reshape(-1)  # (L*3,)

        # Center coordinates
        pred_center = (pred_flat * mask_flat.unsqueeze(1)).sum(0) / (mask_flat.sum() + 1e-8)
        true_center = (true_flat * mask_flat.unsqueeze(1)).sum(0) / (mask_flat.sum() + 1e-8)

        pred_centered = pred_flat - pred_center
        true_centered = true_flat - true_center

        # Compute squared distances
        sq_dist = ((pred_centered - true_centered) ** 2).sum(dim=-1)

        # Apply mask and compute mean
        rmsd_loss = (sq_dist * mask_flat).sum() / (mask_flat.sum() + 1e-8)

        return rmsd_loss


class DistanceLoss(nn.Module):
    """
    Loss for pairwise distances between atoms.
    Computes distance losses for P, C4', and N atoms separately.
    """

    def __init__(self, weight_pp: float = 1.0, weight_cc: float = 1.0, weight_nn: float = 1.0):
        super(DistanceLoss, self).__init__()
        self.weight_pp = weight_pp
        self.weight_cc = weight_cc
        self.weight_nn = weight_nn

    def forward(self, pred_coords: torch.Tensor, true_coords: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_coords: (L, 3, 3) - Predicted coords for [P, C4', N]
            true_coords: (L, 3, 3) - True coords for [P, C4', N]
            mask: (L,) - Sequence mask
        """
        L = pred_coords.shape[0]

        # Create pairwise mask
        pair_mask = mask.unsqueeze(0) * mask.unsqueeze(1)  # (L, L)

        total_loss = 0.0
        weights = [self.weight_pp, self.weight_cc, self.weight_nn]

        for atom_idx, weight in enumerate(weights):
            if weight == 0:
                continue

            # Get coordinates for this atom type
            pred_atom = pred_coords[:, atom_idx, :]  # (L, 3)
            true_atom = true_coords[:, atom_idx, :]  # (L, 3)

            # Compute pairwise distances
            pred_dist = torch.cdist(pred_atom, pred_atom)  # (L, L)
            true_dist = torch.cdist(true_atom, true_atom)  # (L, L)

            # Compute squared difference
            dist_diff = (pred_dist - true_dist) ** 2

            # Apply mask and compute mean
            masked_loss = (dist_diff * pair_mask).sum() / (pair_mask.sum() + 1e-8)
            total_loss += weight * masked_loss

        return total_loss / (self.weight_pp + self.weight_cc + self.weight_nn)


class FAPELoss(nn.Module):
    """
    Frame Aligned Point Error (FAPE) Loss.
    Measures error in local coordinate frames.
    """

    def __init__(self, clamp_distance: float = 10.0, loss_unit_distance: float = 10.0):
        super(FAPELoss, self).__init__()
        self.clamp_distance = clamp_distance
        self.loss_unit_distance = loss_unit_distance

    def forward(self, pred_coords: torch.Tensor, true_coords: torch.Tensor,
                pred_rotation: torch.Tensor, pred_translation: torch.Tensor,
                true_rotation: torch.Tensor, true_translation: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_coords: (L, 3, 3) - Predicted coordinates
            true_coords: (L, 3, 3) - True coordinates
            pred_rotation: (L, 3, 3) - Predicted rotations
            pred_translation: (L, 3) - Predicted translations
            true_rotation: (L, 3, 3) - True rotations
            true_translation: (L, 3) - True translations
            mask: (L,) - Sequence mask
        """
        L = pred_coords.shape[0]

        # Transform predicted coordinates to local frames
        # For each residue i, transform all coordinates to i's frame

        # Predicted local coordinates
        pred_local = []
        true_local = []

        for i in range(L):
            if mask[i] == 0:
                continue

            # Transform to residue i's local frame
            # pred_local_i = R_i^T @ (coords - t_i)
            pred_centered = pred_coords - pred_translation[i].unsqueeze(0).unsqueeze(1)
            pred_local_i = torch.einsum('ljc,cd->ljd', pred_centered, pred_rotation[i].T)
            pred_local.append(pred_local_i)

            true_centered = true_coords - true_translation[i].unsqueeze(0).unsqueeze(1)
            true_local_i = torch.einsum('ljc,cd->ljd', true_centered, true_rotation[i].T)
            true_local.append(true_local_i)

        if len(pred_local) == 0:
            return torch.tensor(0.0, device=pred_coords.device)

        pred_local = torch.stack(pred_local)  # (N_valid, L, 3, 3)
        true_local = torch.stack(true_local)  # (N_valid, L, 3, 3)

        # Compute distances
        distances = torch.sqrt(((pred_local - true_local) ** 2).sum(dim=-1) + 1e-8)  # (N_valid, L, 3)

        # Clamp distances
        clamped_distances = torch.clamp(distances, max=self.clamp_distance)

        # Expand mask
        mask_expanded = mask.unsqueeze(0).unsqueeze(2).expand_as(clamped_distances)

        # Compute loss
        fape_loss = (clamped_distances * mask_expanded).sum() / (mask_expanded.sum() + 1e-8)
        fape_loss = fape_loss / self.loss_unit_distance

        return fape_loss


class LDDTLoss(nn.Module):
    """
    Loss for pLDDT (predicted local distance difference test) confidence prediction.
    """

    def __init__(self, bins: list = [0.5, 1.0, 2.0, 4.0]):
        super(LDDTLoss, self).__init__()
        self.bins = bins

    def forward(self, lddt_logits: torch.Tensor, distance_matrix: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lddt_logits: (L, L, 5) - Log probabilities for LDDT bins (<0.5, <1, <2, <4, >4)
            distance_matrix: (L, L) - True pairwise distances
            mask: (L,) - Sequence mask
        """
        # Compute true LDDT labels from distance matrix
        lddt_labels = self._compute_lddt_labels(distance_matrix)

        # Create pairwise mask
        pair_mask = mask.unsqueeze(0) * mask.unsqueeze(1)

        # Compute cross-entropy loss
        lddt_loss = F.cross_entropy(
            lddt_logits.reshape(-1, 5),
            lddt_labels.reshape(-1),
            reduction='none'
        )

        # Apply mask
        lddt_loss = lddt_loss.reshape(distance_matrix.shape)
        masked_loss = (lddt_loss * pair_mask).sum() / (pair_mask.sum() + 1e-8)

        return masked_loss

    def _compute_lddt_labels(self, distance_matrix: torch.Tensor) -> torch.Tensor:
        """
        Convert distance matrix to LDDT bin labels.
        Bins: <0.5, <1, <2, <4, >4 Angstroms
        """
        labels = torch.zeros_like(distance_matrix, dtype=torch.long)

        labels[distance_matrix < 0.5] = 0
        labels[(distance_matrix >= 0.5) & (distance_matrix < 1.0)] = 1
        labels[(distance_matrix >= 1.0) & (distance_matrix < 2.0)] = 2
        labels[(distance_matrix >= 2.0) & (distance_matrix < 4.0)] = 3
        labels[distance_matrix >= 4.0] = 4

        return labels


class BondLoss(nn.Module):
    """
    Loss for chemical bond constraints (bond lengths, angles).
    Ensures predicted structures satisfy basic chemical geometry.
    """

    def __init__(self):
        super(BondLoss, self).__init__()

        # Ideal bond lengths (in Angstroms) for RNA backbone
        # P-C4': ~4.0, C4'-P (next): ~3.9
        self.ideal_p_c4_distance = 4.0
        self.ideal_c4_p_next_distance = 3.9
        self.ideal_p_p_distance = 5.9  # Approximate

    def forward(self, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (L, 3, 3) - Coordinates for [P, C4', N]
            mask: (L,) - Sequence mask
        """
        L = coords.shape[0]

        if L < 2:
            return torch.tensor(0.0, device=coords.device)

        total_loss = 0.0
        count = 0

        # P-C4' bond within each residue
        p_coords = coords[:, 0, :]  # (L, 3)
        c4_coords = coords[:, 1, :]  # (L, 3)

        p_c4_dist = torch.norm(p_coords - c4_coords, dim=-1)  # (L,)
        p_c4_loss = ((p_c4_dist - self.ideal_p_c4_distance) ** 2 * mask).sum()
        total_loss += p_c4_loss
        count += mask.sum()

        # P(i) - P(i+1) backbone connectivity
        p_p_dist = torch.norm(p_coords[:-1] - p_coords[1:], dim=-1)  # (L-1,)
        p_p_loss = ((p_p_dist - self.ideal_p_p_distance) ** 2 * mask[:-1] * mask[1:]).sum()
        total_loss += p_p_loss
        count += (mask[:-1] * mask[1:]).sum()

        # C4'(i) - P(i+1) connectivity
        c4_p_dist = torch.norm(c4_coords[:-1] - p_coords[1:], dim=-1)  # (L-1,)
        c4_p_loss = ((c4_p_dist - self.ideal_c4_p_next_distance) ** 2 * mask[:-1] * mask[1:]).sum()
        total_loss += c4_p_loss
        count += (mask[:-1] * mask[1:]).sum()

        return total_loss / (count + 1e-8)


class ContactLoss(nn.Module):
    """
    Loss for contact map prediction from pair representation.
    """

    def __init__(self, contact_threshold: float = 8.0):
        super(ContactLoss, self).__init__()
        self.contact_threshold = contact_threshold

    def forward(self, pair_repr: torch.Tensor, true_contacts: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pair_repr: (L, L, D) - Pair representation
            true_contacts: (L, L) - Binary contact map
            mask: (L,) - Sequence mask
        """
        # Project pair representation to contact logits
        # Use simple mean over feature dimension as proxy
        contact_logits = pair_repr.mean(dim=-1)  # (L, L)

        # Create pairwise mask
        pair_mask = mask.unsqueeze(0) * mask.unsqueeze(1)

        # Binary cross-entropy loss
        contact_loss = F.binary_cross_entropy_with_logits(
            contact_logits,
            true_contacts.float(),
            reduction='none'
        )

        # Apply mask
        masked_loss = (contact_loss * pair_mask).sum() / (pair_mask.sum() + 1e-8)

        return masked_loss


# Additional utility functions

def compute_contact_map(coords: torch.Tensor, threshold: float = 8.0) -> torch.Tensor:
    """
    Compute contact map from coordinates.

    Args:
        coords: (L, 3, 3) - Coordinates for [P, C4', N]
        threshold: Distance threshold for contact (Angstroms)

    Returns:
        contact_map: (L, L) - Binary contact map
    """
    # Use C4' atoms for contact definition
    c4_coords = coords[:, 1, :]  # (L, 3)

    # Compute pairwise distances
    distances = torch.cdist(c4_coords, c4_coords)  # (L, L)

    # Create contact map
    contact_map = (distances < threshold).float()

    return contact_map


def compute_distance_matrix(coords: torch.Tensor, atom_idx: int = 1) -> torch.Tensor:
    """
    Compute pairwise distance matrix for a specific atom type.

    Args:
        coords: (L, 3, 3) - Coordinates for [P, C4', N]
        atom_idx: Which atom to use (0=P, 1=C4', 2=N)

    Returns:
        distance_matrix: (L, L) - Pairwise distances
    """
    atom_coords = coords[:, atom_idx, :]  # (L, 3)
    distance_matrix = torch.cdist(atom_coords, atom_coords)  # (L, L)
    return distance_matrix


def compute_rmsd(pred_coords: torch.Tensor, true_coords: torch.Tensor,
                 mask: Optional[torch.Tensor] = None) -> float:
    """
    Compute RMSD between predicted and true coordinates after optimal alignment.

    Args:
        pred_coords: (L, 3, 3) or (L*3, 3) - Predicted coordinates
        true_coords: (L, 3, 3) or (L*3, 3) - True coordinates
        mask: Optional mask

    Returns:
        rmsd: Root mean squared deviation
    """
    # Flatten if needed
    if pred_coords.dim() == 3:
        pred_flat = pred_coords.reshape(-1, 3)
        true_flat = true_coords.reshape(-1, 3)
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, 3).reshape(-1)
    else:
        pred_flat = pred_coords
        true_flat = true_coords

    if mask is None:
        mask = torch.ones(pred_flat.shape[0], device=pred_flat.device)

    # Center coordinates
    pred_center = (pred_flat * mask.unsqueeze(1)).sum(0) / (mask.sum() + 1e-8)
    true_center = (true_flat * mask.unsqueeze(1)).sum(0) / (mask.sum() + 1e-8)

    pred_centered = pred_flat - pred_center
    true_centered = true_flat - true_center

    # Compute RMSD
    sq_dist = ((pred_centered - true_centered) ** 2).sum(dim=-1)
    rmsd = torch.sqrt((sq_dist * mask).sum() / (mask.sum() + 1e-8))

    return rmsd.item()
