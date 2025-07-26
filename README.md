# Spiking-ResNet Implementation for BP prediction from PPG signal.

This repository contains the implementation of a spiking version of ResNet‑18 with snnTorch. For the residual connections, I took inspiration from the paper "Rethinking residual connection in training large-scale spiking neural networks". The residual blocks follow the "PA‑B" configuration mentioned in the paper, whose advantage is robustness to overfitting and delicate perturbations, with sparse updates. The implementation of the residual block is in `src/snn_ppg/models/block.py`. To test the other configurations, just change the order of operations in the `forward` method. The model also includes the "Densely Additive Connections" mentioned in the paper. Their implementation can be found in `src/snn_ppg/models/dablock.py`.  
You can tweak the number of layers manually by changing the model configuration in `src/snn_ppg/models/spiking_resnet.py`.

## Data
In this code, I work with the data proposed in the benchmark for BP prediction through PPG signal mentioned in the paper "A benchmark for machine-learning based non-invasive blood pressure estimation using photoplethysmogram".
You can download data by following their [repository](https://github.com/inventec-ai-center/bp-benchmark) instructions; in particular, the code comes with preprocessing functions that work with the "bcg_dataset". All the functions for data preprocessing are located inside the folder `src/snn_ppg/data`.

## Train & Test
Train and Test entry points are still under development; I will upload a full guide on how to run the scripts when the repository is finished. The model can be tested by running the notebook **Spiking_ResNet.ipynb** on Google Colab; make sure to upload the files in the correct Drive folder, following the code instructions.

## References
1. Yudong Li, Yunlin Lei, Xu Yang, Rethinking residual connection in training large-scale spiking neural networks, Neurocomputing, Volume 616, 2025, 128950, ISSN 0925-2312, https://doi.org/10.1016/j.neucom.2024.128950.
2. González, S., Hsieh, WT. & Chen, T.PC. A benchmark for machine-learning based non-invasive blood pressure estimation using photoplethysmogram. Sci Data 10, 149 (2023). https://doi.org/10.1038/s41597-023-02020-6
