import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. IMU ENCODERS (Conv1D & LSTM)
# ==========================================

class IMUEncoderConv1D(nn.Module):
    def __init__(self, in_channels, latent_dim, sequence_length, latent_type="ae", hidden_dim=64):
        super().__init__()
        self.latent_type = latent_type
        
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU()
        )
        
        self.flatten = nn.Flatten()
        
        # The dummy pass solves two tasks at once:
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, sequence_length)
            dummy_output = self.feature_extractor(dummy_input)
            
            # 1. It finds the exact flat feature size for the linear layer
            flat_dim = dummy_output.numel()
            
            # 2. It captures the exact final timeline length, handling any rounding down
            self.base_len = dummy_output.shape[2] 
        
        if self.latent_type == "vae":
            self.fc_mu = nn.Linear(flat_dim, latent_dim)
            self.fc_logvar = nn.Linear(flat_dim, latent_dim)
        else:
            self.fc_z = nn.Linear(flat_dim, latent_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, T, C) -> (B, C, T)
        features = self.flatten(self.feature_extractor(x))
        if self.latent_type == "vae":
            return self.fc_mu(features), self.fc_logvar(features)
        return self.fc_z(features)


class IMUEncoderLSTM(nn.Module):
    def __init__(self, in_channels, latent_dim, latent_type="ae", hidden_dim=128, num_layers=2):
        super().__init__()
        self.latent_type = latent_type
        
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        
        if self.latent_type == "vae":
            self.fc_mu = nn.Linear(hidden_dim, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        else:
            self.fc_z = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # Input shape: (B, T, C)
        out, (h_n, c_n) = self.lstm(x)
        # Take the final hidden state of the last layer
        final_hidden = h_n[-1] 
        
        if self.latent_type == "vae":
            return self.fc_mu(final_hidden), self.fc_logvar(final_hidden)
        return self.fc_z(final_hidden)


# ==========================================
# 2. IMU DECODERS (Symmetric Transposed Conv1D & Conditional LSTM)
# ==========================================

class IMUDecoderConv1D(nn.Module):
    def __init__(self, latent_dim, out_channels, target_len, base_len, hidden_dim=64):
        super().__init__()
        self.target_len = target_len
        self.base_len = base_len  # Fed directly from the encoder's calculation
        self.base_channels = hidden_dim * 4
        
        self.fc = nn.Linear(latent_dim, self.base_channels * self.base_len)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(self.base_channels, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, out_channels, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, self.base_channels, self.base_len)
        out = self.decoder(x)
        
        # SAFETY NET: If integer rounding left us short or long by a few frames,
        # linearly interpolate the result to match target_len exactly.
        if out.shape[2] != self.target_len:
            out = F.interpolate(out, size=self.target_len, mode='linear', align_corners=False)
            
        return out.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)

class IMUDecoderLSTM(nn.Module):
    """
    CORRECTED: Eliminated static sequence repetition. Implements Conditional State 
    Initialization where the latent space directly configures the initial hidden (h0) 
    and cell (c0) states of a dynamic system canvas.
    """
    def __init__(self, latent_dim, out_channels, target_len, hidden_dim=128, num_layers=2):
        super().__init__()
        self.target_len = target_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_channels = out_channels
        
        # Projections to initialize the dynamical hidden states from the static latent space
        self.fc_h = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.fc_c = nn.Linear(latent_dim, hidden_dim * num_layers)
        
        # The network receives sequentially independent drive inputs (or zeros) 
        # and shifts states via the seeded h0/c0 matrices
        self.lstm = nn.LSTM(
            input_size=out_channels, 
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc_out = nn.Linear(hidden_dim, out_channels)

    def forward(self, z):
        batch_size = z.size(0)
        
        # Project and restructure flat representations to multilayer recurrent vectors
        h_0 = self.fc_h(z).view(self.num_layers, batch_size, self.hidden_dim)
        c_0 = self.fc_c(z).view(self.num_layers, batch_size, self.hidden_dim)
        
        # Create an unconstrained sequence timeline canvas for state changes to transition over
        inputs = torch.zeros(batch_size, self.target_len, self.out_channels, device=z.device)
        
        # Recurrent calculation driven purely by configuration matrices
        out, _ = self.lstm(inputs, (h_0, c_0))
        return self.fc_out(out)


# ==========================================
# 3. UNIFIED AUTOENCODER WRAPPER (PHASE 1)
# ==========================================

class IMUAutoencoder(nn.Module):
    def __init__(self, in_channels, latent_dim, target_len, model_type="conv1d", latent_type="ae", hidden_dim=64):
        super().__init__()
        self.latent_type = latent_type
        
        if model_type == "conv1d":
            # 1. Build the encoder first
            self.encoder = IMUEncoderConv1D(in_channels, latent_dim, target_len, latent_type, hidden_dim)
            # 2. Pass the encoder's dynamically discovered base_len straight into the decoder
            self.decoder = IMUDecoderConv1D(latent_dim, in_channels, target_len, self.encoder.base_len, hidden_dim)
        elif model_type == "lstm":
            self.encoder = IMUEncoderLSTM(in_channels, latent_dim, latent_type, hidden_dim=hidden_dim*2)
            self.decoder = IMUDecoderLSTM(latent_dim, in_channels, target_len, hidden_dim=hidden_dim*2)
        else:
            raise ValueError("model_type must be 'conv1d' or 'lstm'")

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        if self.latent_type == "vae":
            mu, logvar = self.encoder(x)
            z = self.reparameterize(mu, logvar)
            recon = self.decoder(z)
            return recon, mu, logvar
        elif self.latent_type == "ae":
            z = self.encoder(x)
            recon = self.decoder(z)
            return recon, z
        else:
            raise ValueError("latent_type must be 'ae' or 'vae'")
        
# ==========================================
# 4. CUSTOM LOSS FUNCTION FOR AEs
# ==========================================

class TargetFeatureLoss(nn.Module):
    def __init__(self, target_features_idx: list, target_weight: float, auxiliary_weight: float):
        super().__init__()
        self.target_features_idx = target_features_idx
        self.target_weight = target_weight
        self.auxiliary_weight = auxiliary_weight

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        target_idx = set(self.target_features_idx)
        auxiliary_idx = [i for i in range(input.shape[2]) if i not in target_idx]

        if auxiliary_idx:
            mse_loss_aux = F.mse_loss(input[:, :, auxiliary_idx], target[:, :, auxiliary_idx])
        else:
            mse_loss_aux = input.new_tensor(0.0)

        mse_loss_target = F.mse_loss(
            input[:, :, self.target_features_idx],
            target[:, :, self.target_features_idx],
        )
        return self.auxiliary_weight * mse_loss_aux + self.target_weight * mse_loss_target

# ==========================================
# 5. EEG MAPPING NETWORK (PHASE 2)
# ==========================================

class EEGMappingNetwork(nn.Module):
    """
    Maps strict, high-frequency EEG data sequences to the shared IMU latent space.
    """
    def __init__(self, in_channels, latent_dim, hidden_dim, sequence_length, model_type="conv1d"):
        super().__init__()
        self.model_type = model_type
        
        if model_type == "conv1d":
            self.feature_extractor = nn.Sequential(
                nn.Conv1d(in_channels, hidden_dim, kernel_size=7, stride=4, padding=3),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=5, stride=4, padding=2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU()
            )
            
            self.flatten = nn.Flatten()
            
            # Dynamically compute flat dimension for full strict EEG sequence lengths
            with torch.no_grad():
                dummy_input = torch.zeros(1, in_channels, sequence_length)
                dummy_output = self.feature_extractor(dummy_input)
                flat_dim = dummy_output.numel()
                
            self.fc = nn.Linear(flat_dim, latent_dim)
            
        elif model_type == "lstm":
            self.lstm = nn.LSTM(in_channels, hidden_dim, num_layers=2, batch_first=True)
            self.fc = nn.Linear(hidden_dim, latent_dim)
            
    def forward(self, x):
        if self.model_type == "conv1d":
            x = x.permute(0, 2, 1) # Turn (B, T, C) into (B, C, T)
            features = self.flatten(self.feature_extractor(x))
            return self.fc(features)
        else:
            _, (h_n, _) = self.lstm(x)
            return self.fc(h_n[-1])
        

# ==========================================
# 6. DIRECT EEG -> IMU MAPPING NETWORK
# ==========================================

class EEGToMovementNetwork(nn.Module):
    def __init__(self, eeg_channels, imu_channels, eeg_len=768, imu_len=180, hidden_dim=32):
        """
        Direct End-to-End Mapping Network: EEG -> IMU Movement
        
        Args:
            eeg_channels (int): Number of input EEG channels (voltages + power bands)
            imu_channels (int): Total number of output IMU channels
            eeg_len (int): Temporal window length of EEG (e.g., 768 samples)
            imu_len (int): Temporal window length of IMU (e.g., 180 samples)
            hidden_dim (int): Base hidden channel dimension
        """
        super().__init__()
        self.eeg_len = eeg_len
        self.imu_len = imu_len
        self.imu_channels = imu_channels
        
        # 1D Convolutional blocks to extract features across the EEG sequence.
        # Expects: (Batch, Channels, Length) -> we permute input in forward loop
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(eeg_channels, hidden_dim, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
            
            # Compress the timeline cleanly down to a manageable token count
            nn.AdaptiveAvgPool1d(16) 
        )
        
        # Maps compressed spatio-temporal EEG features directly to flat IMU representation
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim * 4 * 16, hidden_dim * 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 8, imu_len * imu_channels)
        )

    def forward(self, eeg):
        # Input shape from TimeSeriesDataset: (Batch, Sequence_Len, Channels)
        # Permute to fit Conv1D expectations: (Batch, Channels, Sequence_Len)
        x = eeg.permute(0, 2, 1)
        
        # Extract features
        x = self.feature_extractor(x)
        
        # Flatten temporal token features
        x = x.view(x.size(0), -1)
        
        # Project to flat IMU shape
        x = self.projection_head(x)
        
        # Reshape back to match destination IMU space: (Batch, IMU_Len, IMU_Channels)
        x = x.view(-1, self.imu_len, self.imu_channels)
        return x