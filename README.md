## About
The repo provides a bucket of models from "[Neural Mulliken Analysis: Molecular Graphs from Density Matrices for QSPR on Raw Quantum-Chemical Data](https://doi.org/10.26434/chemrxiv-2024-k2k3l)"
paper and a minimal inference example.

For RhNet2 model implementation see the [corresponding repo](https://github.com/Shorku/rhnet2)

## Repo structure
- rhnet2.ipynb: data preprocessing, model loading and inference example (No TF-GNN needed)
- rhnet2_with_tfgnn.ipynb: data preprocessing, model loading and inference example (with TF-GNN)
- ./models/: 94 RhNet2 models for aqueous solubility prediction 
- ./data_example/dft/: examples of DFT calculations of the test set compounds
- ./data_example/tfrecords/: the same examples converted to tfgnn.GraphTensor format
