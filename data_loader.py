"""
Data Loader for DRfold2 Training
Handles loading PDB files, FASTA sequences, and creating training batches
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import random
from scipy.spatial.transform import Rotation


# RNA nucleotide encoding
RNA_ALPHABET = {'A': 1, 'G': 2, 'C': 3, 'U': 4, 'T': 4}
ATOM_NAMES = ['P', "C4'", 'N1', 'N9']  # Key atoms for RNA structure


class RNAStructureDataset(Dataset):
    """
    Dataset for RNA structure prediction training.
    Loads PDB files and corresponding FASTA sequences.
    """

    def __init__(self, config: Dict, split: str = 'train'):
        """
        Args:
            config: Configuration dictionary
            split: 'train', 'val', or 'test'
        """
        self.config = config
        self.split = split
        self.data_config = config['data']

        # Get data directories
        if split == 'train':
            self.pdb_dir = self.data_config['train_data_dir']
            self.fasta_dir = self.data_config['train_fasta_dir']
            self.data_list = self.data_config.get('train_list', None)
        elif split == 'val':
            self.pdb_dir = self.data_config['val_data_dir']
            self.fasta_dir = self.data_config['val_fasta_dir']
            self.data_list = self.data_config.get('val_list', None)
        else:  # test
            self.pdb_dir = self.data_config['test_data_dir']
            self.fasta_dir = self.data_config['test_fasta_dir']
            self.data_list = self.data_config.get('test_list', None)

        # Load file list
        self.samples = self._load_file_list()

        # Augmentation settings
        self.use_augmentation = self.data_config.get('use_augmentation', False) and split == 'train'
        self.augmentation_config = self.data_config.get('augmentation', {})

        # Base atom coordinates (standard geometry for each nucleotide)
        self.base_atoms_standard = self._get_standard_base_atoms()

        print(f"Loaded {len(self.samples)} samples for {split} split")

    def _load_file_list(self) -> List[str]:
        """Load list of sample IDs"""
        samples = []

        # If a list file is provided, use it
        if self.data_list and os.path.exists(self.data_list):
            with open(self.data_list, 'r') as f:
                samples = [line.strip() for line in f if line.strip()]
        else:
            # Otherwise, scan the PDB directory
            if os.path.exists(self.pdb_dir):
                for fname in os.listdir(self.pdb_dir):
                    if fname.endswith('.pdb'):
                        sample_id = fname.replace('.pdb', '')
                        samples.append(sample_id)

        return samples

    def _get_standard_base_atoms(self) -> np.ndarray:
        """
        Get standard base atom positions for RNA nucleotides.
        Returns array of shape (4, 3, 3) for [A, G, C, U] x [P, C4', N] x [x,y,z]
        """
        # These are approximate standard positions
        # P at origin, C4' along x, N in xy plane
        basenpy_standard = np.array([
            # A (Adenine - N9)
            [[-0.421, 3.776, 0.0],   # P
             [0.0, 0.0, 0.0],         # C4'
             [3.391, 0.0, 0.0]],      # N9

            # G (Guanine - N9)
            [[-0.421, 3.776, 0.0],   # P
             [0.0, 0.0, 0.0],         # C4'
             [3.391, 0.0, 0.0]],      # N9

            # C (Cytosine - N1)
            [[-0.421, 3.776, 0.0],   # P
             [0.0, 0.0, 0.0],         # C4'
             [3.391, 0.0, 0.0]],      # N1

            # U (Uracil - N1)
            [[-0.421, 3.776, 0.0],   # P
             [0.0, 0.0, 0.0],         # C4'
             [3.391, 0.0, 0.0]]       # N1
        ], dtype=np.float32)

        return basenpy_standard

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Load a single training example.

        Returns:
            Dictionary containing:
                - sequence: RNA sequence string
                - seq_encoding: Encoded sequence (L,)
                - coords: 3D coordinates (L, 3, 3)
                - mask: Sequence mask (L,)
                - rotation: Frame rotations (L, 3, 3)
                - translation: Frame translations (L, 3)
                - sample_id: Sample identifier
        """
        sample_id = self.samples[idx]

        # Load FASTA sequence
        sequence = self._load_fasta(sample_id)

        # Load PDB coordinates
        coords, mask = self._load_pdb(sample_id, sequence)

        # Encode sequence
        seq_encoding = self._encode_sequence(sequence)

        # Compute frames (rotation and translation) from coordinates
        rotation, translation = self._compute_frames(coords, mask)

        # Get base atom positions based on sequence
        base_atoms = self._get_base_atoms(sequence)

        # Apply data augmentation if enabled
        if self.use_augmentation:
            coords, rotation, translation, base_atoms = self._augment_data(
                coords, rotation, translation, base_atoms, mask
            )

        # Compute distance matrix for LDDT
        distance_matrix = self._compute_distance_matrix(coords, mask)

        # Compute contact map
        contact_map = self._compute_contact_map(coords, mask)

        return {
            'sequence': sequence,
            'seq_encoding': torch.from_numpy(seq_encoding).long(),
            'coords': torch.from_numpy(coords).float(),
            'mask': torch.from_numpy(mask).float(),
            'rotation': torch.from_numpy(rotation).float(),
            'translation': torch.from_numpy(translation).float(),
            'base_atoms': torch.from_numpy(base_atoms).float(),
            'distance_matrix': torch.from_numpy(distance_matrix).float(),
            'contact_map': torch.from_numpy(contact_map).float(),
            'sample_id': sample_id
        }

    def _load_fasta(self, sample_id: str) -> str:
        """Load sequence from FASTA file"""
        fasta_path = os.path.join(self.fasta_dir, f"{sample_id}.fasta")

        if not os.path.exists(fasta_path):
            fasta_path = os.path.join(self.fasta_dir, f"{sample_id}.fa")

        if not os.path.exists(fasta_path):
            raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

        sequence = ""
        with open(fasta_path, 'r') as f:
            for line in f:
                if not line.startswith('>'):
                    sequence += line.strip().upper()

        return sequence

    def _load_pdb(self, sample_id: str, sequence: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load coordinates from PDB file.

        Returns:
            coords: (L, 3, 3) array for [P, C4', N]
            mask: (L,) binary mask
        """
        pdb_path = os.path.join(self.pdb_dir, f"{sample_id}.pdb")

        if not os.path.exists(pdb_path):
            raise FileNotFoundError(f"PDB file not found: {pdb_path}")

        L = len(sequence)
        coords = np.zeros((L, 3, 3), dtype=np.float32)
        mask = np.zeros(L, dtype=np.float32)

        # Parse PDB file
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    atom_name = line[12:16].strip()
                    res_num = int(line[22:26].strip()) - 1  # 0-indexed

                    if res_num < 0 or res_num >= L:
                        continue

                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])

                    # Map atom to index
                    if atom_name == 'P':
                        coords[res_num, 0] = [x, y, z]
                        mask[res_num] = 1
                    elif atom_name == "C4'":
                        coords[res_num, 1] = [x, y, z]
                    elif atom_name in ['N1', 'N9']:
                        # N1 for pyrimidines (C, U), N9 for purines (A, G)
                        coords[res_num, 2] = [x, y, z]

        return coords, mask

    def _encode_sequence(self, sequence: str) -> np.ndarray:
        """Encode sequence to integer array"""
        encoding = np.zeros(len(sequence), dtype=np.int64)
        for i, nucleotide in enumerate(sequence):
            encoding[i] = RNA_ALPHABET.get(nucleotide, 0)
        return encoding

    def _get_base_atoms(self, sequence: str) -> np.ndarray:
        """Get base atom positions for sequence"""
        L = len(sequence)
        base_atoms = np.zeros((L, 3, 3), dtype=np.float32)

        for i, nucleotide in enumerate(sequence):
            if nucleotide == 'A':
                base_atoms[i] = self.base_atoms_standard[0]
            elif nucleotide == 'G':
                base_atoms[i] = self.base_atoms_standard[1]
            elif nucleotide == 'C':
                base_atoms[i] = self.base_atoms_standard[2]
            elif nucleotide in ['U', 'T']:
                base_atoms[i] = self.base_atoms_standard[3]

        return base_atoms

    def _compute_frames(self, coords: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute local coordinate frames from coordinates.

        Returns:
            rotation: (L, 3, 3) rotation matrices
            translation: (L, 3) translation vectors
        """
        L = coords.shape[0]
        rotation = np.eye(3)[np.newaxis].repeat(L, axis=0).astype(np.float32)
        translation = coords[:, 1, :].copy()  # Use C4' as translation

        # Compute rotation from three atoms (P, C4', N)
        for i in range(L):
            if mask[i] == 0:
                continue

            # Define local frame from P, C4', N atoms
            p = coords[i, 0]   # P
            c4 = coords[i, 1]  # C4'
            n = coords[i, 2]   # N

            # x-axis: P -> C4'
            x = c4 - p
            x_norm = np.linalg.norm(x)
            if x_norm > 1e-6:
                x = x / x_norm

            # z-axis: perpendicular to plane
            v2 = n - c4
            z = np.cross(x, v2)
            z_norm = np.linalg.norm(z)
            if z_norm > 1e-6:
                z = z / z_norm

            # y-axis: complete right-handed system
            y = np.cross(z, x)

            rotation[i] = np.stack([x, y, z], axis=0).T

        return rotation, translation

    def _compute_distance_matrix(self, coords: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Compute pairwise distance matrix using C4' atoms"""
        c4_coords = coords[:, 1, :]  # (L, 3)

        # Compute pairwise distances
        diff = c4_coords[:, np.newaxis, :] - c4_coords[np.newaxis, :, :]
        distances = np.sqrt((diff ** 2).sum(axis=-1))

        return distances.astype(np.float32)

    def _compute_contact_map(self, coords: np.ndarray, mask: np.ndarray,
                             threshold: float = 8.0) -> np.ndarray:
        """Compute binary contact map"""
        distance_matrix = self._compute_distance_matrix(coords, mask)
        contact_map = (distance_matrix < threshold).astype(np.float32)

        # Apply mask
        mask_2d = mask[:, np.newaxis] * mask[np.newaxis, :]
        contact_map = contact_map * mask_2d

        return contact_map

    def _augment_data(self, coords: np.ndarray, rotation: np.ndarray,
                     translation: np.ndarray, base_atoms: np.ndarray,
                     mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply data augmentation"""

        # Random 3D rotation
        if self.augmentation_config.get('random_rotation', True):
            if random.random() < self.augmentation_config.get('rotation_prob', 0.5):
                rot_matrix = Rotation.random().as_matrix().astype(np.float32)

                # Apply rotation to coordinates
                coords_flat = coords.reshape(-1, 3)
                coords_flat = coords_flat @ rot_matrix.T
                coords = coords_flat.reshape(coords.shape)

                # Apply rotation to base atoms
                base_atoms_flat = base_atoms.reshape(-1, 3)
                base_atoms_flat = base_atoms_flat @ rot_matrix.T
                base_atoms = base_atoms_flat.reshape(base_atoms.shape)

                # Apply rotation to frames
                rotation = rotation @ rot_matrix.T
                translation = translation @ rot_matrix.T

        # Add random noise
        if self.augmentation_config.get('random_noise', True):
            noise_std = self.augmentation_config.get('noise_std', 0.1)
            noise = np.random.normal(0, noise_std, coords.shape).astype(np.float32)
            coords = coords + noise * mask[:, np.newaxis, np.newaxis]

        return coords, rotation, translation, base_atoms


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for DataLoader.
    Handles variable-length sequences by padding.
    """
    # Find max sequence length in batch
    max_len = max(item['seq_encoding'].shape[0] for item in batch)

    batch_size = len(batch)

    # Initialize padded tensors
    seq_encoding = torch.zeros(batch_size, max_len, dtype=torch.long)
    coords = torch.zeros(batch_size, max_len, 3, 3, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.float32)
    rotation = torch.zeros(batch_size, max_len, 3, 3, dtype=torch.float32)
    translation = torch.zeros(batch_size, max_len, 3, dtype=torch.float32)
    base_atoms = torch.zeros(batch_size, max_len, 3, 3, dtype=torch.float32)
    distance_matrix = torch.zeros(batch_size, max_len, max_len, dtype=torch.float32)
    contact_map = torch.zeros(batch_size, max_len, max_len, dtype=torch.float32)

    sequences = []
    sample_ids = []

    # Fill in the batch
    for i, item in enumerate(batch):
        L = item['seq_encoding'].shape[0]

        seq_encoding[i, :L] = item['seq_encoding']
        coords[i, :L] = item['coords']
        mask[i, :L] = item['mask']
        rotation[i, :L] = item['rotation']
        translation[i, :L] = item['translation']
        base_atoms[i, :L] = item['base_atoms']
        distance_matrix[i, :L, :L] = item['distance_matrix']
        contact_map[i, :L, :L] = item['contact_map']

        sequences.append(item['sequence'])
        sample_ids.append(item['sample_id'])

    return {
        'sequence': sequences,
        'seq_encoding': seq_encoding,
        'coords': coords,
        'mask': mask,
        'rotation': rotation,
        'translation': translation,
        'base_atoms': base_atoms,
        'distance_matrix': distance_matrix,
        'contact_map': contact_map,
        'sample_id': sample_ids
    }


def create_data_loaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.

    Args:
        config: Configuration dictionary

    Returns:
        train_loader, val_loader, test_loader
    """
    data_config = config['data']
    training_config = config['training']

    # Create datasets
    train_dataset = RNAStructureDataset(config, split='train')
    val_dataset = RNAStructureDataset(config, split='val')
    test_dataset = RNAStructureDataset(config, split='test')

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=data_config['pin_memory'],
        prefetch_factor=data_config['prefetch_factor'] if data_config['num_workers'] > 0 else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=data_config['pin_memory'],
        prefetch_factor=data_config['prefetch_factor'] if data_config['num_workers'] > 0 else None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Use batch size 1 for testing
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader
