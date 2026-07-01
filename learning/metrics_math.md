# Mathematics of Model Evaluation and Metrics

This document details the mathematical foundations of the evaluation pipelines 
found in `metrics.py`, `train_imu_ae.py`, and `train_eeg_mapping.py`.

---

## 1. Loss Functions (Optimization Objectives)

### Phase 1: IMU Autoencoder Loss (`train_imu_ae.py`)
The Phase 1 autoencoder is optimized using a combination of a weighted Mean Squared Error (MSE) and, if using a Variational Autoencoder (VAE), the Kullback-Leibler Divergence (KLD).

* **Target vs. Auxiliary MSE:** The IMU features are split into "target" channels (e.g., velocity) and "auxiliary" channels. The reconstruction loss strongly favors the target features.
    Loss_recon = (w_target * MSE_target) + (w_aux * MSE_aux)
    Where w_target = 0.9 and w_aux = 0.1.

* **KL Divergence (VAE only):** Regularizes the latent space to approximate a standard normal distribution. 
    KLD = -0.5 * mean(1 + log(sigma^2) - mu^2 - sigma^2)
    *Note: The implementation clips the KLD to a range of [-1000, 1000] for stability.*

* **Total Loss:** Total_Loss = Loss_recon + (kld_weight * KLD)
    The `kld_weight` is linearly annealed from 0 to 0.001 between epochs 20 and 60.

### Phase 2: EEG Mapping Loss (`train_eeg_mapping.py`)
The Phase 2 network maps EEG signals to the frozen latent space of the Phase 1 IMU autoencoder.
* **Latent MSE:** The sole optimization objective is the MSE between the predicted latent vector `z_pred` and the target latent vector `z_target` (which is the `mu` output if Phase 1 was a VAE).
    Loss = mean((z_pred - z_target)^2)

---

## 2. Linear Metrics (Kinematics)
For kinematic channels (velocities, accelerations), standard regression metrics are calculated in `metrics.py`. These are computed either in the scaled model space or back-projected to physical units (degrees per second, dps).

* **MSE (Mean Squared Error):** MSE = mean((y_pred - y_true)^2)

* **RMSE (Root Mean Squared Error):** Gives the error in the same physical units as the target.
    RMSE = sqrt(MSE)

* **MAE (Mean Absolute Error):** Less sensitive to outliers than RMSE.
    MAE = mean(|y_pred - y_true|)

* **R² (Coefficient of Determination):** The headline metric. It measures the proportion of variance in the true signal explained by the model. It is scale-invariant (the R² of scaled data equals the R² of physical data).
    R² = 1 - (SS_res / SS_tot)
    Where:
    SS_res (Residual Sum of Squares) = sum((y_true - y_pred)^2)
    SS_tot (Total Sum of Squares) = sum((y_true - mean(y_true))^2)

* **NRMSE (Normalized RMSE):** Provides a scale-free error metric by dividing RMSE by the standard deviation of the true signal.
    NRMSE = RMSE / std(y_true)

* **Pearson Correlation (r):** Measures linear correlation between prediction and target.
    corr = cov(y_pred, y_true) / (std(y_pred) * std(y_true))

---

## 3. Circular Metrics (Angles)
For orientation angles (degrees), linear arithmetic fails because 359° and 1° are only 2° apart, not 358° apart. The metrics account for this using trigonometric wrapping.

* **Angle Difference Wrapping:**
    diff_rad = (y_pred_deg - y_true_deg) * (pi / 180)
    wrapped_error = arctan2(sin(diff_rad), cos(diff_rad)) * (180 / pi)

* **Circular RMSE & MAE:** Calculated identically to standard RMSE and MAE, but applied to the `wrapped_error` array rather than the raw difference.

* **Cosine Similarity:** The target skill metric for angles. Bounded between [-1, 1], where 1 represents perfect alignment.
    Cos_Sim = mean(cos(wrapped_error_in_radians))

---

## 4. Curve Construction & Dashboard Tracking

### Phase 1 Curves (`train_imu_ae.py`)
Two distinct plots are generated during Phase 1 training to monitor convergence:
1.  **Reconstruction Trajectories:** * **X-axis:** Epochs (1 to 100).
    * **Y-axis:** Scaled MSE.
    * **Lines:** Train Target MSE, Val Target MSE, Train Aux MSE, Val Aux MSE.
2.  **KLD & Annealing (VAE only):**
    * **X-axis:** Epochs.
    * **Primary Y-axis:** KLD value (Train and Val).
    * **Secondary Y-axis:** Annealing weight (`kld_w`), visually showing the linear ramp from 0.0 to 0.001 between epochs 20 and 60.

### Phase 2 Curves (`train_eeg_mapping.py`)
The Phase 2 graph tracks the proxy training signal against the actual physical objective:
1.  **Latent Optimization vs. Physical Reality:**
    * **X-axis:** Epochs.
    * **Primary Y-axis (Latent MSE):** The actual loss being optimized via backpropagation (Train and Val).
    * **Secondary Y-axis (End-to-End Target R²):** The validation R² calculated by decoding the predicted latent vector back to physical IMU space. This is tracked but *not* backpropagated, serving as a sanity check that latent convergence corresponds to actual physical decoding improvement.