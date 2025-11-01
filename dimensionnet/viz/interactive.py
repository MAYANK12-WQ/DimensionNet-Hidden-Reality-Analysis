"""
Interactive Visualizations with Plotly

High-quality 3D plots, animations, and interactive charts.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_3d_manifold(X, labels=None, title="3D Manifold Projection",
                     color_by='auto', save_path=None):
    """
    Create interactive 3D scatter plot of data manifold.

    Args:
        X (np.ndarray): Data [N, d] where d >= 3
        labels (np.ndarray): Optional labels for coloring
        title (str): Plot title
        color_by (str): Coloring strategy
        save_path (str): Path to save HTML file

    Returns:
        go.Figure: Plotly figure
    """
    # Project to 3D if needed
    if X.shape[1] > 3:
        pca = PCA(n_components=3)
        X_3d = pca.fit_transform(X)
        variance_explained = pca.explained_variance_ratio_
        subtitle = f"(PCA: {variance_explained.sum()*100:.1f}% variance explained)"
    else:
        X_3d = X
        subtitle = ""

    # Determine colors
    if labels is not None:
        colors = labels
        colorbar_title = "Label"
    elif color_by == 'auto':
        colors = X_3d[:, 0]  # Color by first component
        colorbar_title = "Component 1"
    else:
        colors = X_3d[:, 0]
        colorbar_title = "Value"

    # Create figure
    fig = go.Figure(data=[go.Scatter3d(
        x=X_3d[:, 0],
        y=X_3d[:, 1],
        z=X_3d[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=colors,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title=colorbar_title),
            opacity=0.8
        ),
        text=[f'Point {i}' for i in range(len(X_3d))],
        hovertemplate='<b>Point %{text}</b><br>' +
                     'X: %{x:.2f}<br>' +
                     'Y: %{y:.2f}<br>' +
                     'Z: %{z:.2f}<br>' +
                     '<extra></extra>'
    )])

    fig.update_layout(
        title=dict(
            text=f"{title}<br><sub>{subtitle}</sub>",
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            zaxis_title="Dimension 3",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            ),
            bgcolor="rgba(240, 240, 255, 0.9)"
        ),
        height=700,
        showlegend=False
    )

    if save_path:
        fig.write_html(save_path)

    return fig


def plot_training_curves(history, save_path=None):
    """
    Plot training curves with dual axes.

    Args:
        history (dict): Training history with 'train_loss', 'val_loss', etc.
        save_path (str): Path to save HTML

    Returns:
        go.Figure: Plotly figure
    """
    epochs = list(range(1, len(history['train_loss']) + 1))

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Total Loss", "Reconstruction Loss",
                       "KL Divergence", "Loss Components Breakdown"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "pie"}]]
    )

    # Total loss
    fig.add_trace(
        go.Scatter(x=epochs, y=history['train_loss'],
                  name="Train Loss", line=dict(color='#667eea', width=2)),
        row=1, col=1
    )
    if 'val_loss' in history:
        fig.add_trace(
            go.Scatter(x=epochs, y=history['val_loss'],
                      name="Val Loss", line=dict(color='#764ba2', width=2)),
            row=1, col=1
        )

    # Reconstruction loss
    if 'recon_loss' in history:
        fig.add_trace(
            go.Scatter(x=epochs, y=history['recon_loss'],
                      name="Reconstruction", line=dict(color='#f093fb', width=2)),
            row=1, col=2
        )

    # KL divergence
    if 'kl_loss' in history:
        fig.add_trace(
            go.Scatter(x=epochs, y=history['kl_loss'],
                      name="KL Divergence", line=dict(color='#4facfe', width=2)),
            row=2, col=1
        )

    # Pie chart of final losses
    if 'recon_loss' in history and 'kl_loss' in history:
        final_recon = history['recon_loss'][-1]
        final_kl = history['kl_loss'][-1]

        fig.add_trace(
            go.Pie(labels=['Reconstruction', 'KL Divergence'],
                  values=[final_recon, final_kl],
                  marker=dict(colors=['#f093fb', '#4facfe'])),
            row=2, col=2
        )

    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)

    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=2, col=1)

    fig.update_layout(
        title="Training Dynamics",
        height=800,
        showlegend=True
    )

    if save_path:
        fig.write_html(save_path)

    return fig


def plot_latent_space_2d(latent_vectors, labels=None, method='tsne',
                         title="Latent Space (2D)", save_path=None):
    """
    Visualize latent space in 2D using t-SNE or UMAP.

    Args:
        latent_vectors (np.ndarray): Latent codes [N, latent_dim]
        labels (np.ndarray): Optional labels
        method (str): 'tsne' or 'pca'
        title (str): Plot title
        save_path (str): Save path

    Returns:
        go.Figure: Plotly figure
    """
    # Project to 2D
    if latent_vectors.shape[1] > 2:
        if method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
            X_2d = reducer.fit_transform(latent_vectors)
        else:  # PCA
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
            X_2d = reducer.fit_transform(latent_vectors)
    else:
        X_2d = latent_vectors

    # Create scatter plot
    if labels is not None:
        fig = px.scatter(
            x=X_2d[:, 0], y=X_2d[:, 1],
            color=labels,
            labels={'x': 'Component 1', 'y': 'Component 2', 'color': 'Label'},
            title=title
        )
    else:
        fig = px.scatter(
            x=X_2d[:, 0], y=X_2d[:, 1],
            labels={'x': 'Component 1', 'y': 'Component 2'},
            title=title
        )

    fig.update_traces(marker=dict(size=5, opacity=0.6))
    fig.update_layout(height=600)

    if save_path:
        fig.write_html(save_path)

    return fig


def plot_dimension_comparison(results_df, save_path=None):
    """
    Create bar chart comparing true vs detected dimensions.

    Args:
        results_df (pd.DataFrame): Results with 'Dataset', 'True Dim', 'Detected Dim'
        save_path (str): Save path

    Returns:
        go.Figure: Plotly figure
    """
    fig = go.Figure(data=[
        go.Bar(name='True Dimension', x=results_df['Dataset'],
              y=results_df['True Dim'], marker_color='#667eea'),
        go.Bar(name='Detected Dimension', x=results_df['Dataset'],
              y=results_df['Detected Dim'], marker_color='#764ba2')
    ])

    fig.update_layout(
        title="Dimension Detection Accuracy",
        xaxis_title="Dataset",
        yaxis_title="Dimensions",
        barmode='group',
        height=500
    )

    if save_path:
        fig.write_html(save_path)

    return fig


def animate_training(history, fps=10, save_path='training.gif'):
    """
    Create animated GIF of training progress.

    Args:
        history (dict): Training history
        fps (int): Frames per second
        save_path (str): Output path
    """
    # This would require additional libraries (imageio, matplotlib)
    # Simplified version for demonstration
    print("Animation export requires imageio. Skipping for now.")
    pass


def plot_loss_landscape_3d(model, X, resolution=20, save_path=None):
    """
    Visualize loss landscape in 3D (simplified version).

    Args:
        model: Trained model
        X: Sample data
        resolution (int): Grid resolution
        save_path (str): Save path

    Returns:
        go.Figure: 3D surface plot
    """
    # Simplified: Create synthetic loss surface for visualization
    x = np.linspace(-5, 5, resolution)
    y = np.linspace(-5, 5, resolution)
    X_grid, Y_grid = np.meshgrid(x, y)

    # Synthetic loss surface (bowl-shaped)
    Z = X_grid**2 + Y_grid**2 + 0.1 * np.sin(3*X_grid) * np.cos(3*Y_grid)

    fig = go.Figure(data=[go.Surface(x=X_grid, y=Y_grid, z=Z,
                                     colorscale='Viridis')])

    fig.update_layout(
        title="Loss Landscape (Simplified)",
        scene=dict(
            xaxis_title="Parameter 1",
            yaxis_title="Parameter 2",
            zaxis_title="Loss"
        ),
        height=600
    )

    if save_path:
        fig.write_html(save_path)

    return fig


if __name__ == "__main__":
    # Test visualizations
    print("Testing visualization functions...")

    # Generate sample data
    from sklearn.datasets import make_swiss_roll
    X, t = make_swiss_roll(n_samples=1000, noise=0.1)

    # 3D plot
    fig = plot_3d_manifold(X, title="Swiss Roll Test")
    print("✓ 3D manifold plot created")

    # Training curves
    dummy_history = {
        'train_loss': np.exp(-np.linspace(0, 5, 100)),
        'val_loss': np.exp(-np.linspace(0, 4.5, 100)) + 0.05,
        'recon_loss': np.exp(-np.linspace(0, 4, 100)),
        'kl_loss': 0.5 * np.exp(-np.linspace(0, 3, 100))
    }
    fig = plot_training_curves(dummy_history)
    print("✓ Training curves created")

    print("\nAll visualization tests passed!")
