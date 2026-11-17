# P6 Deliverables: Methods §3.5 and Results §4.3

## Methods §3.5: M3 - Autoencoder

We implemented an autoencoder to detect network flows whose feature patterns deviate from the negative-class training population. For an input vector $x\in\mathbb{R}^{140}$, the network produces a reconstruction $\hat{x}=f_\theta(x)$ and an anomaly score

$$
s_\theta(x)=\frac{1}{140}\sum_{j=1}^{140}(x_j-\hat{x}_j)^2.
$$

Larger scores indicate greater attack suspicion. Inputs use the shared original/stratified/main dataset and saved preprocessing parameters. Only training rows with `y_binary=0` enter model fitting, following the declared binary-label mapping, including any attempted flows assigned to the negative class. <!-- Source: methods/m3/run.py; src/model_loader.py -->

Eight predefined configurations were evaluated with seeds 42–46, giving 40 fits. Candidates combined hidden widths of 64–16 or 128–32, learning rates of 0.001 or 0.0003, and weight decay of zero or 0.0001. The symmetric fully connected networks therefore had widths 140–64–16–64–140 or 140–128–32–128–140, with ReLU hidden activations and a linear output. Adam used batches of 512, a maximum of 40 epochs, and early stopping after five epochs without a negative-class validation MSE improvement exceeding $10^{-6}$. Each fit retained its best checkpoint. Configuration selection minimized five-seed mean negative-class validation MSE, with candidate identifier as the tie-breaker. <!-- Source: methods/m3/config.json; methods/m3/run.py; results/m3_summary/search.json -->

Each fit used 1,466,743 negative-class training flows. For each selected checkpoint, thresholds $\tau_q$ were the 95th and 99th percentiles of reconstruction errors from 302,963 negative-class validation flows, using linear quantile interpolation. The decision rule was $\hat{y}_q=\mathbf{1}[s_\theta(x)\geq\tau_q]$, with $q\in\{0.95,0.99\}$. Both thresholds were frozen before test evaluation. Model selection, early stopping and threshold calibration used only negative-class validation samples. <!-- Source: docs/m3_threshold.md; results/m3_summary/search.json -->

Dense-layer inference scales with the sum of adjacent layer-width products; training additionally scales with sample count and epochs. CPU inference used the shared benchmark with a batch size of 512, five warm-up calls, 30 timed repetitions and a thread limit of one. Isolation Forest and One-Class SVM were auxiliary comparisons, each fitted once to the same seed-42 subsample of 10,000 negative-class training flows and calibrated on all negative-class validation flows. Their training scope differs from the full-data autoencoder. Reconstruction quality alone provides limited evidence of attack separation or generalization to unseen attack families. <!-- Source: methods/m3/config.json; methods/m3/run.py; results/m3_summary/summary.json -->

## Results §4.3: M3 Results and Comparison

On the original dataset, configuration A06, with hidden widths 128–32, learning rate 0.0003 and zero weight decay, achieved the lowest mean negative-class validation MSE of 0.005856. Its training runs stopped after 18–21 epochs; evaluation used each run's best validation checkpoint. On 346,544 test flows, including 64,038 attacks, the five-seed mean AP was 0.7427 ± 0.0062 and ROC-AUC was 0.9040 ± 0.0031. All reported dispersions are sample standard deviations across five seeds (ddof=1). AP and ROC-AUC are identical between threshold versions because the underlying anomaly scores are unchanged. <!-- Source: results/m3_summary/search.json; results/m3_summary/summary.json -->

At the 95th-percentile threshold, mean attack recall was 61.44% ± 1.31 percentage points and FPR was 4.84% ± 0.17 percentage points. Raising the threshold to the 99th percentile reduced FPR to 0.83% ± 0.14 percentage points, while recall fell to 52.27% ± 1.80 percentage points. Precision increased from 74.20% to 93.47%, and macro-F1 increased from 0.8028 to 0.8074. These rates were measured on Test using validation-calibrated thresholds. <!-- Source: results/m3_summary/summary.json -->

Figure X: Test reconstruction-error distributions for original-data autoencoder A06, seed 42. Both panels show log(1 + reconstruction MSE). The left panel covers the full range with a logarithmic density axis, and the right panel shows the central range with tails omitted. Vertical lines mark q95 and q99 thresholds calibrated on negative-class validation flows. <!-- Source: figures/fig_m3_error_dist_v2.png; methods/m3/plot_error_distribution.py -->

Detection varied substantially by attack family. At the 95th-percentile threshold, mean recall was 94.26% for DoS Slowhttptest and 90.91% for DoS Hulk, but 10.32% for botnet, 4.43% for portscan and 0% for SQL injection. Heartbleed and infiltration each reached 100%, but their test supports were only two and five flows, limiting interpretation. <!-- Source: results/m3_summary/summary.json -->

Auxiliary AP values were 0.5828 for Isolation Forest and 0.5046 for One-Class SVM, with one fit per method. Selected autoencoder fits took 347.54 ± 36.62 seconds on average, including validation-based checkpoint selection. Mean amortized CPU inference cost was 0.001746 milliseconds per flow under the batch benchmark. Low recall for several attack families shows uneven detection coverage at the calibrated thresholds. Cross-method comparisons and corrected-data sensitivity are reported in Section 4.4. <!-- Source: results/m3_summary/summary.json -->
