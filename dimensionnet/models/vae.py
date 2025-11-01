"""
Variational Autoencoder (VAE) for Dimension Detection

Implements a VAE that learns to detect intrinsic dimensionality
in high-dimensional data by finding optimal latent representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class Encoder(nn.Module):
    """
    VAE Encoder Network

    Maps high-dimensional input to latent distribution parameters (μ, σ).
    """

    def __init__(self, input_dim, latent_dim, hidden_dims=[256, 128, 64]):
        super().__init__()

        # Build encoder layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Latent distribution parameters
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """
    VAE Decoder Network

    Reconstructs high-dimensional data from latent representation.
    """

    def __init__(self, latent_dim, output_dim, hidden_dims=[64, 128, 256]):
        super().__init__()

        layers = []
        prev_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        return self.decoder(z)


class VAE(nn.Module):
    """
    Variational Autoencoder for Dimensionality Detection

    Args:
        input_dim (int): Input data dimensionality
        latent_dim (int): Latent space dimensionality
        hidden_dims (list): Hidden layer dimensions
        beta (float): Weight for KL divergence (β-VAE)
    """

    def __init__(self, input_dim, latent_dim, hidden_dims=[256, 128, 64], beta=1.0):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder = Encoder(input_dim, latent_dim, hidden_dims)
        self.decoder = Decoder(latent_dim, input_dim, hidden_dims[::-1])

        # Track training history
        self.history = {
            'train_loss': [],
            'recon_loss': [],
            'kl_loss': [],
            'val_loss': []
        }

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = μ + σ⊙ε where ε ~ N(0,I)

        Enables backpropagation through random sampling.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """Forward pass through VAE"""
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        """
        VAE Loss = Reconstruction Loss + β * KL Divergence

        KL(q(z|x) || p(z)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss /= x.size(0)

        # Total loss
        total_loss = recon_loss + self.beta * kl_loss

        return total_loss, recon_loss, kl_loss

    def fit(self, X, epochs=100, batch_size=128, lr=1e-3,
            validation_split=0.1, device='cuda', verbose=True):
        """
        Train VAE on data

        Args:
            X (np.ndarray): Training data [N, input_dim]
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            lr (float): Learning rate
            validation_split (float): Fraction for validation
            device (str): 'cuda' or 'cpu'
            verbose (bool): Print progress

        Returns:
            dict: Training history
        """
        # Setup device
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.to(device)

        # Convert to tensor
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)

        # Train/val split
        n_samples = X.size(0)
        n_val = int(n_samples * validation_split)
        indices = torch.randperm(n_samples)

        X_train = X[indices[n_val:]]
        X_val = X[indices[:n_val]]

        # Data loader
        train_dataset = torch.utils.data.TensorDataset(X_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        # Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Training loop
        for epoch in range(epochs):
            self.train()
            train_loss = 0
            train_recon = 0
            train_kl = 0

            progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}') if verbose else train_loader

            for (batch_x,) in progress:
                batch_x = batch_x.to(device)

                optimizer.zero_grad()
                recon_x, mu, logvar = self(batch_x)
                loss, recon_loss, kl_loss = self.loss_function(recon_x, batch_x, mu, logvar)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_recon += recon_loss.item()
                train_kl += kl_loss.item()

                if verbose:
                    progress.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'recon': f'{recon_loss.item():.4f}',
                        'kl': f'{kl_loss.item():.4f}'
                    })

            # Average losses
            n_batches = len(train_loader)
            train_loss /= n_batches
            train_recon /= n_batches
            train_kl /= n_batches

            # Validation
            self.eval()
            with torch.no_grad():
                X_val_gpu = X_val.to(device)
                recon_val, mu_val, logvar_val = self(X_val_gpu)
                val_loss, _, _ = self.loss_function(recon_val, X_val_gpu, mu_val, logvar_val)

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['recon_loss'].append(train_recon)
            self.history['kl_loss'].append(train_kl)
            self.history['val_loss'].append(val_loss.item())

            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss.item():.4f}')

        return self.history

    def encode(self, X, device='cuda'):
        """
        Encode data to latent space

        Args:
            X (np.ndarray or torch.Tensor): Input data

        Returns:
            np.ndarray: Latent representations
        """
        self.eval()
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.to(device)

        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)

        X = X.to(device)

        with torch.no_grad():
            mu, _ = self.encoder(X)

        return mu.cpu().numpy()

    def decode(self, z, device='cuda'):
        """
        Decode latent representations

        Args:
            z (np.ndarray or torch.Tensor): Latent codes

        Returns:
            np.ndarray: Reconstructed data
        """
        self.eval()
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.to(device)

        if isinstance(z, np.ndarray):
            z = torch.FloatTensor(z)

        z = z.to(device)

        with torch.no_grad():
            recon = self.decoder(z)

        return recon.cpu().numpy()

    def estimate_intrinsic_dimension(self, X=None, method='reconstruction_error'):
        """
        Estimate intrinsic dimensionality of data

        Uses reconstruction error as proxy for information content.
        Intrinsic dim ≈ latent dim where reconstruction error plateaus.

        Args:
            X (np.ndarray): Data (if not provided, uses training data)
            method (str): Estimation method

        Returns:
            int: Estimated intrinsic dimension
        """
        if method == 'reconstruction_error':
            # Use elbow in reconstruction error curve
            recon_errors = self.history['recon_loss']

            # Find elbow using second derivative
            if len(recon_errors) > 10:
                diffs = np.diff(recon_errors)
                second_diffs = np.diff(diffs)
                elbow_idx = np.argmax(second_diffs) + 1

                # Map to dimension (rough estimate)
                estimated_dim = min(self.latent_dim, max(1, int(self.latent_dim * (1 - elbow_idx / len(recon_errors)))))
                return estimated_dim
            else:
                return self.latent_dim

        else:
            return self.latent_dim

    def get_num_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test VAE
    print("Testing VAE...")

    # Create synthetic data (5D manifold in 100D space)
    n_samples = 1000
    intrinsic_dim = 5
    ambient_dim = 100

    # Generate 5D data
    X_intrinsic = np.random.randn(n_samples, intrinsic_dim)

    # Embed in 100D with random projection
    projection = np.random.randn(intrinsic_dim, ambient_dim)
    X_ambient = X_intrinsic @ projection

    # Add noise
    X_ambient += 0.1 * np.random.randn(n_samples, ambient_dim)

    # Train VAE
    vae = VAE(input_dim=ambient_dim, latent_dim=10)
    print(f"Parameters: {vae.get_num_parameters():,}")

    history = vae.fit(X_ambient, epochs=20, batch_size=64, device='cpu', verbose=True)

    # Estimate dimension
    estimated_dim = vae.estimate_intrinsic_dimension()
    print(f"\nTrue intrinsic dimension: {intrinsic_dim}")
    print(f"Estimated dimension: {estimated_dim}")

    # Test encoding/decoding
    z = vae.encode(X_ambient[:10], device='cpu')
    recon = vae.decode(z, device='cpu')

    print(f"\nEncoded shape: {z.shape}")
    print(f"Reconstruction error: {np.mean((X_ambient[:10] - recon)**2):.6f}")
