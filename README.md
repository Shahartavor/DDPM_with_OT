# DDPM with OT 
Measuring and improving the distribution of diffusion-model generated data using optimal-transport

# How to run:


# Final Results
In the [Passive Phase](#Test-OT-per-epoch), we used a simple CNN for feature extraction to project images into feature space.\
For evaluation, 1,000 feature vectors were sampled from both the training and test sets (matching the MNIST test set size).\
At each of the 35 epochs, we generated 1,000 synthetic samples from the checkpointed DDPM model trained above (with MSE noise prediction).\
We then computed three distances in feature space:
1. OT between generated and training samples
2. OT between generated and test samples
3. OT between training and test samples (baseline)

Key findings:

1. OT(gen, train) and OT(gen, test) are relatively close, suggesting generalization.
2. OT(gen, test) remains much larger than OT(train, test), indicating limited sample quality.
3. Both OT(gen, train) and OT(gen, test) decrease over epochs, showing convergence toward the real distribution.
4. MMD and FID follow similar trends, confirming OT’s consistency with standard metrics.

<img width="1208" height="314" alt="ot_per_epoch_passive" src="https://github.com/user-attachments/assets/cc0c2352-27e0-4380-a5f2-2eb4c78d3cf3" />

### Best result:
Retraining with higher weight on MSE loss then OT loss, and with blur=10:
<img width="1639" height="399" alt="retrain_mse_high_blur10_ot" src="https://github.com/user-attachments/assets/642c0f7a-612e-4e92-8cc9-3a1ba9f2ab3d" />
<img width="1521" height="447" alt="retrain_mse_high_blur10_grid" src="https://github.com/user-attachments/assets/83666e0b-d444-4eca-b225-d29f8c92a4be" />

