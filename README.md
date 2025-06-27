## About
This repository provides a bucket of models from 
"[Neural Mulliken Analysis: Molecular Graphs from Density Matrices for QSPR on Raw Quantum-Chemical Data](https://doi.org/10.1021/acs.jctc.5c00425)"
paper (also available as a 
[preprint](https://doi.org/10.26434/chemrxiv-2024-k2k3l-v3)). 

A notebook with minimal inference example is also available (see 
[rhnet2.ipynb](https://github.com/Shorku/SolubilityChallenge2008/blob/main/rhnet2.ipynb)).
For general RhNet2 model implementation see the 
[corresponding repository](https://github.com/Shorku/rhnet2).

**Note:** provided here, ensemble of models is 
**probably to heavy to play with**. For a lighter model trained on a larger
dataset see [another repository](https://github.com/Shorku/SolubilityChallenge2019)

## Contents
- [Repo structure](#repo-structure)
- [Model](#model)

## Repo structure
```plaintext
+-- rhnet2.ipynb                # Data preprocessing, model loading and inference example (No TF-GNN needed)
+-- rhnet2_with_tfgnn.ipynb     # Data preprocessing, model loading and inference example (with TF-GNN)
+-- data_utils.py               # Data preprocessing utilities
+-- graph_utils.py              # TF-GNN graph data preprocessing utilities
+-- models.config               # Tensorflow Model Server config
+-- rhnet2.pbtxt                # TF-GNN graph schema
+-- models                      # RhNet2SC1 models
|   +-- LOO_G8D1W1Hd0Hw0_GKANTrueWKANFalseHKANFalseL2H0.08L2W0.08_batch8_lr0.0001
|   |   +-- 50-06-6             # Single model trained on the SC-1 set. Phenobarbital (CASRN 50-06-6) excluded.
|   |   +-- 50-33-9             # Single model trained on the SC-1 set. Phenylbutazone (CASRN 50-33-9) excluded. 
|   |   +-- ...
+-- data_example                # Examples of DFT calculations of the SC-1 test set compounds
|   +-- dft                     # ORCA outputs
|   |   +-- Acebutolol.zip      # ORCA calculation for Acebutolol
|   |   +-- ...
|   +-- tfrecords               # Same data converted to TF-GNN graphs
|   |   +-- Acebutolol.tfrecord # Acebutolol density graph
```

## Model
The model provided here was trained and tested using the training and test sets from
the First Solubility Challenge (SC-1,
[link](https://doi.org/10.1021/ci800058v),
[link](https://doi.org/10.1021/ci800436c)).
For model and fitting details see the corresponding
[paper](https://doi.org/10.1021/acs.jctc.5c00425) (also available as a 
[preprint](https://doi.org/10.26434/chemrxiv-2024-k2k3l-v3)).
