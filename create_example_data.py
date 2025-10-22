"""
Create example training data for testing the training pipeline
Generates synthetic RNA structures for demonstration
"""

import os
import numpy as np
from pathlib import Path

def create_example_pdb(sequence, output_path):
    """
    Create a simple PDB file with basic RNA backbone geometry
    This is for testing purposes only - not real structures
    """
    L = len(sequence)

    with open(output_path, 'w') as f:
        f.write("HEADER    RNA STRUCTURE\n")
        f.write(f"TITLE     EXAMPLE RNA STRUCTURE - {L} NUCLEOTIDES\n")

        atom_num = 1

        for i, nucleotide in enumerate(sequence):
            res_num = i + 1

            # Generate simple coordinates (not chemically accurate, just for testing)
            # Backbone forms a simple helix-like structure
            angle = i * 0.6  # radians
            radius = 10.0

            # P atom (phosphate)
            x_p = radius * np.cos(angle)
            y_p = radius * np.sin(angle)
            z_p = i * 3.0

            # C4' atom (sugar)
            x_c = (radius + 2.0) * np.cos(angle)
            y_c = (radius + 2.0) * np.sin(angle)
            z_c = i * 3.0 + 0.5

            # N1/N9 atom (base)
            x_n = (radius + 5.0) * np.cos(angle)
            y_n = (radius + 5.0) * np.sin(angle)
            z_n = i * 3.0 + 1.0

            # Write P atom
            f.write(f"ATOM  {atom_num:5d}  P    {nucleotide} A{res_num:4d}    "
                   f"{x_p:8.3f}{y_p:8.3f}{z_p:8.3f}  1.00 50.00           P\n")
            atom_num += 1

            # Write C4' atom
            f.write(f"ATOM  {atom_num:5d}  C4'  {nucleotide} A{res_num:4d}    "
                   f"{x_c:8.3f}{y_c:8.3f}{z_c:8.3f}  1.00 50.00           C\n")
            atom_num += 1

            # Write N1 (for C,U) or N9 (for A,G) atom
            n_name = "N9" if nucleotide in ['A', 'G'] else "N1"
            f.write(f"ATOM  {atom_num:5d}  {n_name}  {nucleotide} A{res_num:4d}    "
                   f"{x_n:8.3f}{y_n:8.3f}{z_n:8.3f}  1.00 50.00           N\n")
            atom_num += 1

        f.write("END\n")


def create_example_fasta(sequence, output_path, name="example"):
    """Create a FASTA file"""
    with open(output_path, 'w') as f:
        f.write(f">{name}\n")
        f.write(f"{sequence}\n")


def generate_random_rna_sequence(length, gc_content=0.5):
    """Generate a random RNA sequence"""
    nucleotides = ['A', 'U', 'G', 'C']

    # Adjust probabilities for GC content
    gc_prob = gc_content / 2
    au_prob = (1 - gc_content) / 2

    probs = [au_prob, au_prob, gc_prob, gc_prob]

    sequence = ''.join(np.random.choice(nucleotides, size=length, p=probs))
    return sequence


def main():
    print("Creating example training data...")

    # Create directories
    dirs = {
        'train': 'data/train',
        'train_fasta': 'data/train_fasta',
        'val': 'data/val',
        'val_fasta': 'data/val_fasta',
        'test': 'data/test',
        'test_fasta': 'data/test_fasta'
    }

    for dir_name, dir_path in dirs.items():
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")

    # Generate training samples
    print("\nGenerating training samples...")
    num_train = 20
    for i in range(num_train):
        # Random length between 20 and 100
        length = np.random.randint(20, 101)
        sequence = generate_random_rna_sequence(length)

        sample_id = f"train_sample_{i:03d}"

        # Create PDB
        pdb_path = f"data/train/{sample_id}.pdb"
        create_example_pdb(sequence, pdb_path)

        # Create FASTA
        fasta_path = f"data/train_fasta/{sample_id}.fasta"
        create_example_fasta(sequence, fasta_path, name=sample_id)

        print(f"  Created {sample_id} (length: {length})")

    # Generate validation samples
    print("\nGenerating validation samples...")
    num_val = 5
    for i in range(num_val):
        length = np.random.randint(20, 101)
        sequence = generate_random_rna_sequence(length)

        sample_id = f"val_sample_{i:03d}"

        pdb_path = f"data/val/{sample_id}.pdb"
        create_example_pdb(sequence, pdb_path)

        fasta_path = f"data/val_fasta/{sample_id}.fasta"
        create_example_fasta(sequence, fasta_path, name=sample_id)

        print(f"  Created {sample_id} (length: {length})")

    # Generate test samples
    print("\nGenerating test samples...")
    num_test = 3
    for i in range(num_test):
        length = np.random.randint(20, 101)
        sequence = generate_random_rna_sequence(length)

        sample_id = f"test_sample_{i:03d}"

        pdb_path = f"data/test/{sample_id}.pdb"
        create_example_pdb(sequence, pdb_path)

        fasta_path = f"data/test_fasta/{sample_id}.fasta"
        create_example_fasta(sequence, fasta_path, name=sample_id)

        print(f"  Created {sample_id} (length: {length})")

    print("\n" + "=" * 80)
    print("Example data created successfully!")
    print("=" * 80)
    print(f"\nCreated:")
    print(f"  - {num_train} training samples")
    print(f"  - {num_val} validation samples")
    print(f"  - {num_test} test samples")
    print("\nNote: These are synthetic structures for testing only.")
    print("For real training, use actual PDB structures from RCSB or similar databases.")
    print("\nYou can now test the training pipeline with:")
    print("  python test_training_setup.py")
    print("  python train.py --config train_config.yaml --device cuda")


if __name__ == '__main__':
    main()
