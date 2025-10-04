# DDPM with OT 
Measuring and improving the distribution of diffusion-model generated data using optimal-transport

# Active Phase Results
In the Passive Phase (in title: Test OT per epoch), we used a simple CNN for feature extraction to project images into feature space.\
For evaluation, 1,000 feature vectors were sampled from both the training and test sets (matching the MNIST test set size).\
At each of the 35 epochs, we generated 1,000 synthetic samples from the checkpointed DDPM model trained above (with MSE noise prediction).\
We then computed three distances in feature space:
1. OT between generated and training samples
2. OT between generated and test samples
3. OT between training and test samples (baseline)

_Key findings:_

1. OT(gen, train) and OT(gen, test) are relatively close, suggesting generalization.
2. OT(gen, test) remains much larger than OT(train, test), indicating limited sample quality.
3. Both OT(gen, train) and OT(gen, test) decrease over epochs, showing convergence toward the real distribution.
4. MMD and FID follow similar trends, confirming OT’s consistency with standard metrics.

<img width="1208" height="314" alt="ot_per_epoch_passive" src="https://github.com/user-attachments/assets/cc0c2352-27e0-4380-a5f2-2eb4c78d3cf3" />

_Choosing blur_
The OT loss includes a **blur** parameter (default 0.05) that controls entropy: higher values make distances smoother but less precise, while lower values stay closer to the original data
We used blur=10, as it provided a favorable trade-off between coverage, fidelity, and overfitting in the Passive Phase, while preserving computational speed.
<img width="1698" height="371" alt="blur=10" src="https://github.com/user-attachments/assets/691beb51-0900-4194-847e-6b0a71528c5d" />

1. *Fine-tuning current model with higher weight on MSE loss then OT loss, and with blur=10:*
<img width="1617" height="405" alt="finetune_mse_high_blur10_ot" src="https://github.com/user-attachments/assets/33a5099a-4dd8-4fca-851b-995266d894d9" />
<img width="1446" height="451" alt="finetune_mse_high_blur10_grid" src="https://github.com/user-attachments/assets/074c2c27-75df-4f94-a81a-c2f22b38f7c7" />


3. *Retraining with higher weight on MSE loss then OT loss, and with blur=10:*
<img width="1639" height="399" alt="retrain_mse_high_blur10_ot" src="https://github.com/user-attachments/assets/642c0f7a-612e-4e92-8cc9-3a1ba9f2ab3d" />
<img width="1521" height="447" alt="retrain_mse_high_blur10_grid" src="https://github.com/user-attachments/assets/83666e0b-d444-4eca-b225-d29f8c92a4be" />

4. *Retraining with higher weight on MSE loss then OT loss, and with blur=0.05:*
<img width="1633" height="395" alt="retrain_mse_high_blur0 05_ot" src="https://github.com/user-attachments/assets/a5d59c6d-3ab0-45e1-bc9c-4ab46a980f2f" />
<img width="1475" height="456" alt="retrain_mse_high_blur0 05_grid" src="https://github.com/user-attachments/assets/d708c31a-6b02-46fd-b128-4f59db502217" />

5. *Retraining with equal weights and with blur=10:*
<img width="1633" height="395" alt="retrain_equal_blur10_ot" src="https://github.com/user-attachments/assets/c68547df-468b-488d-9775-9b06644ad7c1" />
<img width="1422" height="452" alt="retrain_equal_blur10_grid" src="https://github.com/user-attachments/assets/2fdd979e-aa86-445a-953b-e0ca526b0118" />
