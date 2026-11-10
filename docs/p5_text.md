# P5 Deliverables: Methods §3.4 and Results §4.2

## Methods §3.4: M2 - 1D-CNN

For the deep learning method, we implemented a One-Dimensional Convolutional Neural Network (1D-CNN). The model treats the 140 numerical network flow features as a one-dimensional sequence. It uses two convolutional layers with 32 and 64 filters to learn local patterns between features. Batch normalisation and ReLU activation are applied after each convolutional layer. Adaptive average pooling is then used to reduce the feature dimensions. Finally, a fully connected layer produces one attack score for each network flow.

To support a fair comparison with M1, we used the same dataset split and class imbalance handling. The model was trained for 40 epochs using the Adam optimiser, with a learning rate of 0.001 and a batch size of 256. We used a positive class weight of 4.9299 to reduce the effect of class imbalance. Five random seeds (42–46) were used to measure the consistency of the results. Mixed precision training was also enabled to improve GPU efficiency.

The experiments were performed on an NVIDIA GeForce RTX 4060 Laptop GPU. The average training time was approximately 1,929 seconds per seed. This shows that M2 required more training time and computing resources than a lightweight traditional model. We also measured CPU inference performance to understand its potential running cost. These measurements provide evidence for comparing both model performance and computational cost.

## Results §4.2: M2 Results and Comparison

M2 achieved a mean ROC-AUC of 0.9809 ± 0.0040 and a mean Average Precision (AP) of 0.9172 ± 0.0233 across five random seeds. Its mean accuracy was 0.9211 ± 0.0126, while its mean Macro-F1 was 0.8478 ± 0.0292. At a false positive rate (FPR) of 1%, the mean true positive rate (TPR) was 0.6115 ± 0.0700. At an FPR of 5%, it increased to 0.7951 ± 0.0986.

Figure X: Training loss curve of M2.

The training loss curve shows how the model's loss changed during training. The evaluation results also show that performance varied between random seeds, especially at low false positive rates. This suggests that the model's detection performance at strict operating points was less consistent than its overall ranking performance.

Compared with M1 (GBDT), which achieved an AP of 0.9997, M2 had a lower AP of 0.9172. This indicates that M1 was better at identifying attack traffic in this dataset. One possible reason is that network flow features are tabular data and do not have a natural order or spatial relationship. Therefore, the local filters used by a CNN may not always capture the most useful patterns. In contrast, tree-based models can split features based on their values and may be more suitable for this type of data.

In terms of computational cost, M2 required an average of approximately 32.15 minutes to train each seed. Its CPU inference throughput was approximately 23,666 flows per second. Overall, M2 achieved good detection performance, but it did not outperform M1. This suggests that a deep learning model is not always the best choice for network intrusion detection, especially when the input consists of structured flow features.