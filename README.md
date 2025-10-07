# DDPM Training with Optimal Transport Loss
Measuring and improving the distribution of diffusion-model generated data using optimal-transport.

## Introduction
This Python notebook implements a Denoising Diffusion Probabilistic Model (DDPM) for generating MNIST-like images and evaluates/improves the generated data distribution using Optimal Transport (OT) metrics in feature space.

The project is divided into two phases:
- **Passive Phase**: Train a standard DDPM with MSE loss, extract features using a pre-trained CNN, generate samples, and measure distributional distances (OT, FID, MMD) to assess quality, coverage, fidelity, and overfitting.
- **Active Phase**: Resume or retrain the DDPM by adding a differentiable OT loss term to the MSE, aiming to align generated features closer to the real distribution.

Motivation: Diffusion models may face mode collapse or low diversity; Optimal Transport improves alignment by capturing geometric distribution differences.

### How to run:
Open in Colab, install dependencies (cell 2), and execute sequentially. Training takes ~5-10 min/epoch on GPU. Customize epochs, blur, and runs in main section.

## Passive Phase
In the Passive Phase, we train a DDPM using standard MSE loss. A simple CNN (trained to ~99% accuracy on MNIST classification) is used as a feature extractor to project images into a 128-dimensional feature space.
At each of the 35 epochs, we generate 10,000 synthetic samples from the checkpointed DDPM model and compare them against 10,000 feature vectors from the training and test sets.
We compute three distances in feature space:

1. OT (Sinkhorn divergence) between generated and training/test samples.
2. FID (Fréchet Inception Distance) for distributional similarity.
3. MMD (Maximum Mean Discrepancy) for kernel-based comparison.
4. Baseline: OT/FID/MMD between training and test sets.


Key Results and Findings

- **Convergence**: OT(gen, train/test), FID, and MMD decrease over the first 15-20 epochs, indicating the model improves in capturing the real distribution. However, progress plateaus, leaving a persistent gap.
- **Generalization**: OT(gen, train) ≈ OT(gen, test), showing no overfitting—the model generalizes to unseen test data.
- **Sample Quality**: OT(gen, test) remains ~6x larger than OT(train, test) (e.g., ~633 vs. ~103 at epoch 35), highlighting limited fidelity and diversity in generated samples.
- **Consistency Across Metrics**: FID and MMD trends mirror OT, validating OT as a reliable metric for diffusion evaluation.

<img width="1208" height="314" alt="ot_per_epoch_passive" src="https://github.com/user-attachments/assets/cc0c2352-27e0-4380-a5f2-2eb4c78d3cf3" />

### Coverage, Fidelity, Overfitting Analysis

- **Coverage (Recall)**: Gradually add digits to test subset; OT decreases as coverage improves.
- **Fidelity (Precision)**: Add noise to test; OT increases with noise std, measuring sharpness loss.
- **Overfitting**: Mix train into half-test; OT to train decreases, to held-out test increases with mix fraction.

### Choosing Blur for OT
The OT loss includes a blur parameter (default 0.05) that controls entropy: higher values make distances smoother but less precise, while lower values stay closer to the original data.
We used blur=10, as it provided a favorable trade-off between coverage, fidelity, and overfitting in the Passive Phase, while preserving computational speed.
<img width="1698" height="371" alt="blur=10" src="https://github.com/user-attachments/assets/691beb51-0900-4194-847e-6b0a71528c5d" />

## Active Phase

In the Active Phase, we add a differentiable OT term to the MSE loss during training, using GeomLoss (`Sinkhorn, p=2, debias=True`). Dynamic weighting balances MSE/OT (`ot_weight = loss_eps / loss_ot`).
Supports resuming from passive checkpoints or retraining. Logs losses/weights to WandB. Generates/evaluates per epoch.

### 1. Resume with Higher MSE Weight (OT Weight Lower) and Blur=10

* Starts from passive epoch 35, trains to 50.
* **Results**: OT distances decrease slightly compared to Passive, but the gap to baseline remains large (~500+). Generated grids show improved sharpness, but diversity is limited.

<img width="1617" height="405" alt="finetune_mse_high_blur10_ot" src="https://github.com/user-attachments/assets/33a5099a-4dd8-4fca-851b-995266d894d9" />
<img width="1446" height="451" alt="finetune_mse_high_blur10_grid" src="https://github.com/user-attachments/assets/074c2c27-75df-4f94-a81a-c2f22b38f7c7" />

### 2. Retraining with Higher MSE Weight and Blur=10:
* From scratch.
* **Results**: Better convergence than fine-tuning; OT drops faster early on but plateaus. Grids exhibit more varied digits, suggesting improved coverage.

<img width="1639" height="399" alt="retrain_mse_high_blur10_ot" src="https://github.com/user-attachments/assets/642c0f7a-612e-4e92-8cc9-3a1ba9f2ab3d" />
<img width="1521" height="447" alt="retrain_mse_high_blur10_grid" src="https://github.com/user-attachments/assets/83666e0b-d444-4eca-b225-d29f8c92a4be" />

### 3. Retraining with Higher MSE Weight and Blur=0.05:
* From scratch, lower blur for precision.
* **Results**: Lower blur leads to sharper OT reductions, but higher instability (spikes in loss). Grids are crisper, but some artifacts appear.

<img width="1633" height="395" alt="retrain_mse_high_blur0 05_ot" src="https://github.com/user-attachments/assets/a5d59c6d-3ab0-45e1-bc9c-4ab46a980f2f" />
<img width="1475" height="456" alt="retrain_mse_high_blur0 05_grid" src="https://github.com/user-attachments/assets/d708c31a-6b02-46fd-b128-4f59db502217" />

### 4. Retraining with Equal Weights and Blur=10:
* From scratch, balanced MSE/OT.
* **Results**: Balanced weights yield the best OT alignment (~200-300 gap), with grids showing high fidelity and diversity. However, training is slower due to stronger OT pull.

<img width="1633" height="395" alt="retrain_equal_blur10_ot" src="https://github.com/user-attachments/assets/c68547df-468b-488d-9775-9b06644ad7c1" />
<img width="1422" height="452" alt="retrain_equal_blur10_grid" src="https://github.com/user-attachments/assets/2fdd979e-aa86-445a-953b-e0ca526b0118" />

## Conclusion
Passive Phase shows DDPM convergence but gaps; Active Phase with OT reduces them, with retraining and balanced weights performing best. Dynamic weighting helps stability.

Future: Adaptive blur, complex datasets, class-conditioning.
