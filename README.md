# Spiking-ResNet Implementation for BP prediction from PPG signal.
This repository contains the implementation of Spiking version of the ResNet 18 with snnTorch. For the residual connections i took inspiration from the paper "Rethinking residual connection in training large-scale spiking neural
networks". The residual blocks follow the  "PA-B" configuration mentioned in the paper, which advantage is that robust to overfitting and delicate perturbations, but with sparse updates. The implementation of the residual block in is "src/snn_ppg/models/block.py". To test the other configurations, just change the order of operations in the forward method. The model also includes the "Densively Additive Connections" mentioned in the paper. Their implementation can be found in "src/snn_ppg/models/dablock.py" 
You can tweak the number of layers manually by changing the model configuration in "src/snn_ppg/models/spiking_resnet.py".
## Data
In this code, I work with the data proposed in the benchmark for bp prediction thorugh PPG signal mentioned in the paper "A benchmark for machine-learning based non-invasive blood pressure estimation using photoplethysmogram".
You can download data by following their [repository](https://github.com/inventec-ai-center/bp-benchmark) instructions, in particular the code comes with preprocessing functions that work with the "bcg_dataset". All the function for data preprocessing are located inside the folde "src/snn_ppg/data".
## Train & Test
Train and Test entry points are still under work, I will upload a full guide on how to run the scripts when the repository is finished. The model can be tested by running the noteboook Spiking_ResNet.ipynb on google colab, make sure to upload the files in the correct Drive folder, by following the code instructions.
## References
1. Yudong Li, Yunlin Lei, Xu Yang, Rethinking residual connection in training large-scale spiking neural networks, Neurocomputing, Volume 616, 2025, 128950, ISSN 0925-2312, https://doi.org/10.1016/j.neucom.2024.128950.
2. González, S., Hsieh, WT. & Chen, T.PC. A benchmark for machine-learning based non-invasive blood pressure estimation using photoplethysmogram. Sci Data 10, 149 (2023). https://doi.org/10.1038/s41597-023-02020-6
