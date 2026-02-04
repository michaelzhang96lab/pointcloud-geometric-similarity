"""
Visualize the synthetic surface point clouds
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def load_csv(filepath):
    """Load point cloud from CSV"""
    data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    return data

def visualize_surfaces():
    """Create comparison visualization of the three clouds"""
    
    # Load data
    cloud_a = load_csv("synthetic_surfaces/cloud_A_high_density_texture1.csv")
    cloud_b = load_csv("synthetic_surfaces/cloud_B_low_density_texture1.csv")
    cloud_c = load_csv("synthetic_surfaces/cloud_C_low_density_texture2.csv")
    
    # Create figure with 2 rows
    fig = plt.figure(figsize=(15, 10))
    
    # Row 1: 3D views (subsampled proportionally for visibility)
    # Show 10% of each cloud to maintain relative density appearance
    sample_ratio = 0.1
    
    # Cloud A (subsample proportionally)
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    n_show_a = int(len(cloud_a) * sample_ratio)
    idx_a = np.random.choice(len(cloud_a), n_show_a, replace=False)
    ax1.scatter(cloud_a[idx_a, 0], cloud_a[idx_a, 1], cloud_a[idx_a, 2] * 1000, 
                c=cloud_a[idx_a, 2] * 1000, cmap='viridis', s=0.3, alpha=0.6)
    ax1.set_title(f'Cloud A: HIGH Density, Texture 1\n({len(cloud_a):,} points, showing {n_show_a:,})')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (μm)')
    
    # Cloud B (subsample proportionally)
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    n_show_b = int(len(cloud_b) * sample_ratio)
    idx_b = np.random.choice(len(cloud_b), n_show_b, replace=False)
    ax2.scatter(cloud_b[idx_b, 0], cloud_b[idx_b, 1], cloud_b[idx_b, 2] * 1000,
                c=cloud_b[idx_b, 2] * 1000, cmap='viridis', s=0.5, alpha=0.6)
    ax2.set_title(f'Cloud B: LOW Density, Texture 1\n({len(cloud_b):,} points, showing {n_show_b:,}) - SAME texture as A')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_zlabel('Z (μm)')
    
    # Cloud C (subsample proportionally)
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    n_show_c = int(len(cloud_c) * sample_ratio)
    idx_c = np.random.choice(len(cloud_c), n_show_c, replace=False)
    ax3.scatter(cloud_c[idx_c, 0], cloud_c[idx_c, 1], cloud_c[idx_c, 2] * 1000,
                c=cloud_c[idx_c, 2] * 1000, cmap='plasma', s=0.5, alpha=0.6)
    ax3.set_title(f'Cloud C: LOW Density, Texture 2\n({len(cloud_c):,} points, showing {n_show_c:,}) - DIFFERENT texture')
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Y (mm)')
    ax3.set_zlabel('Z (μm)')
    
    # Row 2: Height maps (2D top-down view showing texture)
    grid_size = 100
    
    def points_to_heightmap(points, grid_size):
        """Convert scattered points to gridded height map"""
        x_bins = np.linspace(points[:, 0].min(), points[:, 0].max(), grid_size + 1)
        y_bins = np.linspace(points[:, 1].min(), points[:, 1].max(), grid_size + 1)
        
        heightmap = np.full((grid_size, grid_size), np.nan)
        counts = np.zeros((grid_size, grid_size))
        
        x_idx = np.digitize(points[:, 0], x_bins) - 1
        y_idx = np.digitize(points[:, 1], y_bins) - 1
        
        x_idx = np.clip(x_idx, 0, grid_size - 1)
        y_idx = np.clip(y_idx, 0, grid_size - 1)
        
        for i in range(len(points)):
            xi, yi = x_idx[i], y_idx[i]
            if np.isnan(heightmap[yi, xi]):
                heightmap[yi, xi] = points[i, 2]
                counts[yi, xi] = 1
            else:
                heightmap[yi, xi] += points[i, 2]
                counts[yi, xi] += 1
        
        mask = counts > 0
        heightmap[mask] /= counts[mask]
        
        return heightmap
    
    # Height maps
    ax4 = fig.add_subplot(2, 3, 4)
    hm_a = points_to_heightmap(cloud_a, 200)
    im4 = ax4.imshow(hm_a * 1000, cmap='viridis', extent=[0, 1, 0, 1], origin='lower')
    ax4.set_title('Cloud A: Height Map\n(50 μm correlation length)')
    ax4.set_xlabel('X (mm)')
    ax4.set_ylabel('Y (mm)')
    plt.colorbar(im4, ax=ax4, label='Height (μm)')
    
    ax5 = fig.add_subplot(2, 3, 5)
    hm_b = points_to_heightmap(cloud_b, grid_size)
    im5 = ax5.imshow(hm_b * 1000, cmap='viridis', extent=[0, 1, 0, 1], origin='lower')
    ax5.set_title('Cloud B: Height Map\n(Same texture as A, sparser)')
    ax5.set_xlabel('X (mm)')
    ax5.set_ylabel('Y (mm)')
    plt.colorbar(im5, ax=ax5, label='Height (μm)')
    
    ax6 = fig.add_subplot(2, 3, 6)
    hm_c = points_to_heightmap(cloud_c, grid_size)
    im6 = ax6.imshow(hm_c * 1000, cmap='plasma', extent=[0, 1, 0, 1], origin='lower')
    ax6.set_title('Cloud C: Height Map\n(20 μm correlation - FINER texture)')
    ax6.set_xlabel('X (mm)')
    ax6.set_ylabel('Y (mm)')
    plt.colorbar(im6, ax=ax6, label='Height (μm)')
    
    plt.tight_layout()
    plt.savefig('synthetic_surfaces/visualization.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to synthetic_surfaces/visualization.png")
    
    # Comparison figure
    fig2, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    def compute_1d_psd(heightmap):
        hm = heightmap.copy()
        hm[np.isnan(hm)] = 0
        fft = np.fft.fft2(hm)
        psd_2d = np.abs(fft) ** 2
        
        ny, nx = psd_2d.shape
        y, x = np.ogrid[:ny, :nx]
        center_y, center_x = ny // 2, nx // 2
        y = y - center_y
        x = x - center_x
        r = np.sqrt(x**2 + y**2).astype(int)
        
        psd_2d_shifted = np.fft.fftshift(psd_2d)
        max_r = min(center_y, center_x)
        radial_psd = np.zeros(max_r)
        
        for i in range(max_r):
            mask = (r == i)
            if mask.sum() > 0:
                radial_psd[i] = psd_2d_shifted[mask].mean()
        
        return radial_psd
    
    # Height distribution
    axes[0].hist(cloud_a[:, 2] * 1000, bins=50, alpha=0.7, label='Cloud A', density=True)
    axes[0].hist(cloud_b[:, 2] * 1000, bins=50, alpha=0.7, label='Cloud B', density=True)
    axes[0].hist(cloud_c[:, 2] * 1000, bins=50, alpha=0.7, label='Cloud C', density=True)
    axes[0].set_xlabel('Height (μm)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Height Distribution\n(A and B similar, C slightly different)')
    axes[0].legend()
    
    # PSD comparison
    psd_a = compute_1d_psd(hm_a)
    psd_b = compute_1d_psd(hm_b)
    psd_c = compute_1d_psd(hm_c)
    
    freq = np.arange(len(psd_a))
    axes[1].semilogy(freq[1:30], psd_a[1:30], 'b-', label='Cloud A (Texture 1)', linewidth=2)
    axes[1].semilogy(freq[1:30], psd_b[1:30], 'g--', label='Cloud B (Texture 1)', linewidth=2)
    axes[1].semilogy(freq[1:30], psd_c[1:30], 'r-', label='Cloud C (Texture 2)', linewidth=2)
    axes[1].set_xlabel('Spatial Frequency')
    axes[1].set_ylabel('Power Spectral Density')
    axes[1].set_title('Texture Comparison via PSD\n(A≈B, but A≠C)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Stats
    stats_text = f"""Statistical Summary:

Cloud A (High density, Texture 1):
  Points: {len(cloud_a):,}
  Sa: {np.mean(np.abs(cloud_a[:, 2] - cloud_a[:, 2].mean())) * 1000:.3f} μm
  Sq: {np.std(cloud_a[:, 2]) * 1000:.3f} μm

Cloud B (Low density, Texture 1):
  Points: {len(cloud_b):,}
  Sa: {np.mean(np.abs(cloud_b[:, 2] - cloud_b[:, 2].mean())) * 1000:.3f} μm
  Sq: {np.std(cloud_b[:, 2]) * 1000:.3f} μm

Cloud C (Low density, Texture 2):
  Points: {len(cloud_c):,}
  Sa: {np.mean(np.abs(cloud_c[:, 2] - cloud_c[:, 2].mean())) * 1000:.3f} μm
  Sq: {np.std(cloud_c[:, 2]) * 1000:.3f} μm

Key insight:
  A and B have SAME texture (same PSD shape)
  but DIFFERENT densities.
  
  C has DIFFERENT texture (different PSD)
  but SIMILAR density to B.
"""
    axes[2].text(0.05, 0.95, stats_text, transform=axes[2].transAxes, 
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[2].axis('off')
    axes[2].set_title('Summary Statistics')
    
    plt.tight_layout()
    plt.savefig('synthetic_surfaces/comparison.png', dpi=150, bbox_inches='tight')
    print("Saved comparison to synthetic_surfaces/comparison.png")
    
    plt.close('all')

if __name__ == "__main__":
    visualize_surfaces()
