# MLIPS-Fundamentals
Working notebooks for learning GNNs, MLIPs, and uncertainty quantification for catalyst screening.

## Motivation

Machine-learned interatomic potentials (MLIPs) are increasingly central to computational materials science and catalysis, but the ML stack underneath them — graph neural networks, uncertainty quantification, distribution shift — is scattered across the ML and quantum chemistry literature. This repo works through the fundamental principles and techniques used.

## Notebooks
 
| # | Notebook | Topic |
|---|----------|-------|
| 01 | `01_graph_data_structures.ipynb` | `Data` objects in PyG, `edge_index`, manual graph construction |
<!--
| 02 | `02_gcn_on_mutag.ipynb` | 2-layer GCN on MUTAG, graph classification end-to-end |
| 03 | `03_message_passing_from_scratch.ipynb` | Re-implement GCN via `MessagePassing` base class |
| 04 | `04_qm9_data_exploration.ipynb` | Load QM9, inspect node/edge/target tensors, train-val-test split |
| 05 | `05_molecular_gnn_qm9.ipynb` | Distance-based edge features, global pooling, train on dipole moment |
| 06 | `06_schnet_concepts.ipynb` | Read-along notes on SchNet; compare to notebook 05 |
| 07 | `07_deep_ensembles.ipynb` | Train 5 models with different seeds, aggregate predictions, plot std vs. error |
| 08 | `08_reliability_diagrams.ipynb` | Prediction intervals, empirical coverage, reliability diagram |
| 09 | `09_temperature_scaling.ipynb` | Post-hoc calibration via scalar temperature on validation set |
| 10 | `10_bias_correction.ipynb` | Constant and linear bias correction under distribution shift; RMSE vs. n-labels curve |
-->

## Setup
 
```bash
git clone https://github.com/zbschwab/mlip-fundamentals
cd mlip-fundamentals
pip install torch torch-geometric
```
 
Or use Google Colab — each notebook includes a setup cell. <!-- QM9 training runs in ~10 minutes on a free T4 GPU. -->
 
## References
#### 01
- https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html
<!-- 
- Schütt et al. (2017) — *SchNet: A continuous-filter convolutional neural network for modeling quantum interactions*
- Lakshminarayanan et al. (2017) — *Simple and scalable predictive uncertainty estimation using deep ensembles*
- Guo et al. (2017) — *On calibration of modern neural networks*
- Tran et al. (2020) — *Methods for comparing uncertainty quantifications for material property predictions*
- Open Catalyst Project — https://opencatalystproject.org
-->
 
