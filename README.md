# DRfold2: Ab initio RNA structure prediction with composite language model and denoised end-to-end learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.11%2B-red)](https://pytorch.org/)

## Overview

DRfold2 is a deep learning method for RNA structure prediction. At its core, DRfold2 utilizes the RNA Composite Language Model (RCLM), which provides enhanced full likelihood approximation capabilities to effectively capture co-evolutionary signals from unsupervised sequence data.

### Key Features

- Advanced RNA Composite Language Model (RCLM)
- End-to-end structure and geometry prediction
- Optimization protocol


## Installation

### Prerequisites

#### Minimal Requirements
- Python (tested on 3.10.4, 3.11.4, and 3.11.7)
- PyTorch (tested on 1.11, 2.01, and 2.21)
- NumPy
- SciPy
- BioPython 

#### Optional Dependencies
- OpenMM (required for structure refinement) - [Installation Guide](https://openmm.org/)

### Setup Instructions

1. Clone and navigate to the DRfold2 directory:
   ```bash
   git clone https://github.com/leeyang/DRfold2
   cd DRfold2
   ```

2. Run the installation script:
   ```bash
   bash install.sh
   ```
   This will download model weights ~1.3GB and install Arena.

## Usage

### Basic Structure Prediction

For single model prediction:
```bash
python DRfold_infer.py [input fasta file] [output_dir]
```

For multiple model prediction (up to 5 models):
```bash
python DRfold_infer.py [input fasta file] [output_dir] 1
```

### Parameters

- `[input fasta file]`: Target sequence in FASTA format
- `[output_dir]`: Directory for saving intermediate and final results
- Final predictions will be saved as `[output dir]/relax/model_*.pdb`

### Structure Refinement (Optional)

To further refine a predicted structure:
```bash
python script/refine.py [input pdb] [output pdb]
```

## Example Usage

```bash
python DRfold_infer.py test/seq.fasta test/8fza_A/ 1
```

The final results can be found in `test/8fza_A/relax/`.

**Note:** For long RNA sequences, you may want to clear intermediate results from `test/8fza_A/` to save space.

## Performance

DRfold2 has been extensively tested on non-redundant test sets with various redundancy cut-offs, consistently demonstrating superior performance in:
- 3D structure prediction
- 2D base pair modeling
- Co-evolutionary feature learning from unsupervised data

## Bug Reports and Issues

Please report any issues or bugs on our [GitHub Issues page](https://github.com/leeyang/DRfold2/issues).

## Citation

If you use DRfold2 in your research, please cite:
```bibtex
@article{li2025drfold2,
  title={Ab initio RNA structure prediction with composite language model and denoised end-to-end learning},
  author={Yang Li, Chenjie Feng, Xi Zhang, Yang Zhang.},
  journal={},
  year={2025}
}
```

## License

Copyright (c) 2025 Yang Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
