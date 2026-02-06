"""
Synthetic Engineered Surface Point Cloud Generator

Generates point clouds representing engineered surface textures with:
- Controllable surface roughness (Sa - arithmetical mean height)
- Controllable correlation length (texture feature size)
- Controllable point density
- Optional directional lay (anisotropic surfaces)

Surface model: Gaussian random field with exponential autocorrelation
This is a standard model in surface metrology (ISO 25178 framework)

Output: PLY files (widely supported) and optional CSV

Author: Zhongyi Michael Zhang
Purpose: Testing geometric similarity algorithms on realistic surface data
"""

import numpy as np
from typing import Tuple, Optional
import os


def generate_gaussian_random_surface(
    size_x: float,
    size_y: float,
    n_points_x: int,
    n_points_y: int,
    sa_roughness: float,
    correlation_length_x: float,
    correlation_length_y: Optional[float] = None,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a Gaussian random rough surface using FFT-based spectral method.
    
    Parameters:
    -----------
    size_x, size_y : float
        Physical dimensions of the surface (e.g., in mm or μm)
    n_points_x, n_points_y : int
        Number of grid points in each direction
    sa_roughness : float
        Target Sa (arithmetical mean height) roughness
    correlation_length_x : float
        Correlation length in x direction (controls feature size)
    correlation_length_y : float, optional
        Correlation length in y direction. If None, equals correlation_length_x (isotropic)
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    X, Y, Z : np.ndarray
        2D arrays of coordinates
    """
    if seed is not None:
        np.random.seed(seed)
    
    if correlation_length_y is None:
        correlation_length_y = correlation_length_x
    
    # Create spatial grid
    x = np.linspace(0, size_x, n_points_x)
    y = np.linspace(0, size_y, n_points_y)
    X, Y = np.meshgrid(x, y)
    
    # Frequency grid
    dx = size_x / (n_points_x - 1)
    dy = size_y / (n_points_y - 1)
    
    freq_x = np.fft.fftfreq(n_points_x, dx)
    freq_y = np.fft.fftfreq(n_points_y, dy)
    Fx, Fy = np.meshgrid(freq_x, freq_y)
    
    # Power spectral density for exponential autocorrelation
    # PSD(f) = (2π * σ² * lcx * lcy) / (1 + (2π*fx*lcx)² + (2π*fy*lcy)²)^1.5
    # This gives exponential ACF: R(τx,τy) = σ² * exp(-|τx|/lcx - |τy|/lcy)
    
    denominator = (1 + (2 * np.pi * Fx * correlation_length_x)**2 + 
                   (2 * np.pi * Fy * correlation_length_y)**2)**1.5
    
    # Avoid division by zero at DC
    denominator[denominator == 0] = 1e-10
    
    psd = (2 * np.pi * correlation_length_x * correlation_length_y) / denominator
    
    # Generate random phase
    random_phase = np.exp(2j * np.pi * np.random.random((n_points_y, n_points_x)))
    
    # Generate surface via inverse FFT
    amplitude = np.sqrt(psd)
    Z_fft = amplitude * random_phase
    Z = np.real(np.fft.ifft2(Z_fft))
    
    # Normalize to target Sa roughness
    # Sa = mean(|Z - mean(Z)|)
    Z = Z - np.mean(Z)
    current_sa = np.mean(np.abs(Z))
    if current_sa > 0:
        Z = Z * (sa_roughness / current_sa)
    
    return X, Y, Z


def grid_to_pointcloud(
    X: np.ndarray, 
    Y: np.ndarray, 
    Z: np.ndarray,
    target_density: Optional[float] = None,
    target_n_points: Optional[int] = None,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Convert grid surface to point cloud with specified density.
    
    Parameters:
    -----------
    X, Y, Z : np.ndarray
        2D arrays from generate_gaussian_random_surface
    target_density : float, optional
        Points per unit area. If None, uses all grid points.
    target_n_points : int, optional
        Exact number of points desired. Overrides target_density.
    seed : int, optional
        Random seed for reproducible subsampling
        
    Returns:
    --------
    points : np.ndarray
        Nx3 array of (x, y, z) coordinates
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Flatten to point cloud
    all_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    n_total = len(all_points)
    
    if target_n_points is not None:
        n_sample = min(target_n_points, n_total)
    elif target_density is not None:
        area = (X.max() - X.min()) * (Y.max() - Y.min())
        n_sample = min(int(target_density * area), n_total)
    else:
        return all_points
    
    # Random subsample
    indices = np.random.choice(n_total, size=n_sample, replace=False)
    return all_points[indices]


def add_measurement_noise(
    points: np.ndarray,
    noise_std: float,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Add Gaussian measurement noise to point cloud (simulates sensor noise).
    
    Parameters:
    -----------
    points : np.ndarray
        Nx3 array of points
    noise_std : float
        Standard deviation of noise in z direction
    seed : int, optional
        Random seed
        
    Returns:
    --------
    noisy_points : np.ndarray
        Points with added noise
    """
    if seed is not None:
        np.random.seed(seed)
    
    noisy = points.copy()
    noisy[:, 2] += np.random.normal(0, noise_std, len(points))
    return noisy


def save_pointcloud_ply(points: np.ndarray, filepath: str):
    """Save point cloud to PLY format."""
    n_points = len(points)
    
    header = f"""ply
format ascii 1.0
element vertex {n_points}
property float x
property float y
property float z
end_header
"""
    
    with open(filepath, 'w') as f:
        f.write(header)
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    
    print(f"Saved {n_points} points to {filepath}")


def save_pointcloud_csv(points: np.ndarray, filepath: str):
    """Save point cloud to CSV format."""
    np.savetxt(filepath, points, delimiter=',', header='x,y,z', comments='', fmt='%.6f')
    print(f"Saved {len(points)} points to {filepath}")


def generate_test_set():
    """
    Generate the three point clouds as specified:
    - Cloud A: High density surface texture (texture 1)
    - Cloud B: Low density (~1/10 of A), same texture as A
    - Cloud C: Low density (similar to B), DIFFERENT texture
    
    Surface parameters chosen to represent typical machined surfaces:
    - Size: 1mm x 1mm patch
    - Sa roughness: ~0.8 μm (typical for ground surface)
    - Correlation length: ~50 μm (typical feature size)
    """
    
    # Common parameters
    size_x = 1.0  # mm
    size_y = 1.0  # mm
    sa_roughness = 0.0008  # mm (= 0.8 μm, typical ground surface)
    
    # Texture 1 parameters (for clouds A and B)
    correlation_length_1 = 0.05  # mm (= 50 μm)
    texture_seed_1 = 42
    
    # Texture 2 parameters (for cloud C) - different correlation length = different texture
    correlation_length_2 = 0.02  # mm (= 20 μm) - finer texture
    texture_seed_2 = 123
    
    # Density parameters
    high_density_grid = 500  # 500x500 = 250,000 points for high density
    low_density_points = 25000  # ~1/10 of high density
    
    print("=" * 60)
    print("GENERATING SYNTHETIC ENGINEERED SURFACE POINT CLOUDS")
    print("=" * 60)
    print(f"\nSurface parameters:")
    print(f"  - Patch size: {size_x} x {size_y} mm")
    print(f"  - Sa roughness: {sa_roughness * 1000:.1f} μm")
    print(f"  - Texture 1 correlation length: {correlation_length_1 * 1000:.0f} μm")
    print(f"  - Texture 2 correlation length: {correlation_length_2 * 1000:.0f} μm")
    print()
    
    # Generate high-resolution grid for texture 1
    print("Generating texture 1 (high-resolution grid)...")
    X1, Y1, Z1 = generate_gaussian_random_surface(
        size_x, size_y,
        high_density_grid, high_density_grid,
        sa_roughness,
        correlation_length_1,
        seed=texture_seed_1
    )
    
    # Cloud A: High density, texture 1
    print("Creating Cloud A (high density, texture 1)...")
    cloud_a = grid_to_pointcloud(X1, Y1, Z1)  # All points
    cloud_a = add_measurement_noise(cloud_a, noise_std=sa_roughness * 0.05, seed=100)
    
    # Cloud B: Low density, texture 1 (subsampled from same surface)
    print("Creating Cloud B (low density, texture 1)...")
    cloud_b = grid_to_pointcloud(X1, Y1, Z1, target_n_points=low_density_points, seed=200)
    cloud_b = add_measurement_noise(cloud_b, noise_std=sa_roughness * 0.05, seed=101)
    
    # Generate texture 2
    print("Generating texture 2...")
    X2, Y2, Z2 = generate_gaussian_random_surface(
        size_x, size_y,
        high_density_grid, high_density_grid,
        sa_roughness,
        correlation_length_2,
        seed=texture_seed_2
    )
    
    # Cloud C: Low density, texture 2
    print("Creating Cloud C (low density, texture 2)...")
    cloud_c = grid_to_pointcloud(X2, Y2, Z2, target_n_points=low_density_points, seed=300)
    cloud_c = add_measurement_noise(cloud_c, noise_std=sa_roughness * 0.05, seed=102)
    
    # Create output directory
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save all clouds
    print("\nSaving point clouds...")
    
    save_pointcloud_ply(cloud_a, os.path.join(output_dir, "cloud_A_high_density_texture1.ply"))
    save_pointcloud_ply(cloud_b, os.path.join(output_dir, "cloud_B_low_density_texture1.ply"))
    save_pointcloud_ply(cloud_c, os.path.join(output_dir, "cloud_C_low_density_texture2.ply"))
    
    # Also save CSV versions
    save_pointcloud_csv(cloud_a, os.path.join(output_dir, "cloud_A_high_density_texture1.csv"))
    save_pointcloud_csv(cloud_b, os.path.join(output_dir, "cloud_B_low_density_texture1.csv"))
    save_pointcloud_csv(cloud_c, os.path.join(output_dir, "cloud_C_low_density_texture2.csv"))
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nCloud A (high density, texture 1):")
    print(f"  - Points: {len(cloud_a):,}")
    print(f"  - Density: {len(cloud_a) / (size_x * size_y):,.0f} points/mm²")
    print(f"  - Z range: [{cloud_a[:, 2].min()*1000:.2f}, {cloud_a[:, 2].max()*1000:.2f}] μm")
    
    print(f"\nCloud B (low density, texture 1 - SAME as A):")
    print(f"  - Points: {len(cloud_b):,}")
    print(f"  - Density: {len(cloud_b) / (size_x * size_y):,.0f} points/mm²")
    print(f"  - Z range: [{cloud_b[:, 2].min()*1000:.2f}, {cloud_b[:, 2].max()*1000:.2f}] μm")
    print(f"  - Density ratio A/B: {len(cloud_a) / len(cloud_b):.1f}x")
    
    print(f"\nCloud C (low density, texture 2 - DIFFERENT from A/B):")
    print(f"  - Points: {len(cloud_c):,}")
    print(f"  - Density: {len(cloud_c) / (size_x * size_y):,.0f} points/mm²")
    print(f"  - Z range: [{cloud_c[:, 2].min()*1000:.2f}, {cloud_c[:, 2].max()*1000:.2f}] μm")
    
    print(f"\nTexture difference:")
    print(f"  - Texture 1 feature size: ~{correlation_length_1 * 1000:.0f} μm")
    print(f"  - Texture 2 feature size: ~{correlation_length_2 * 1000:.0f} μm")
    print(f"  - (Texture 2 has finer/smaller features)")
    
    print(f"\nFiles saved to: {os.path.abspath(output_dir)}/")
    
    return cloud_a, cloud_b, cloud_c


def generate_custom_surface(
    output_name: str,
    size_x: float = 1.0,
    size_y: float = 1.0,
    sa_roughness: float = 0.0008,
    correlation_length: float = 0.05,
    n_points: int = 50000,
    anisotropy_ratio: float = 1.0,
    seed: int = None
):
    """
    Generate a custom surface with specified parameters.
    
    Parameters:
    -----------
    output_name : str
        Base name for output files
    size_x, size_y : float
        Surface dimensions in mm
    sa_roughness : float
        Sa roughness in mm (e.g., 0.0008 = 0.8 μm)
    correlation_length : float
        Feature size in mm (e.g., 0.05 = 50 μm)
    n_points : int
        Number of points in output
    anisotropy_ratio : float
        Ratio of correlation_length_y / correlation_length_x
        1.0 = isotropic, >1 = elongated in x direction (e.g., turning marks)
    seed : int
        Random seed for reproducibility
    """
    
    # Generate at high resolution first
    grid_size = max(500, int(np.sqrt(n_points) * 2))
    
    X, Y, Z = generate_gaussian_random_surface(
        size_x, size_y,
        grid_size, grid_size,
        sa_roughness,
        correlation_length,
        correlation_length * anisotropy_ratio,
        seed=seed
    )
    
    points = grid_to_pointcloud(X, Y, Z, target_n_points=n_points, seed=seed)
    points = add_measurement_noise(points, noise_std=sa_roughness * 0.05, seed=seed)
    
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    
    save_pointcloud_ply(points, os.path.join(output_dir, f"{output_name}.ply"))
    save_pointcloud_csv(points, os.path.join(output_dir, f"{output_name}.csv"))
    
    return points


if __name__ == "__main__":
    # Generate the standard test set (A, B, C)
    cloud_a, cloud_b, cloud_c = generate_test_set()
    
    print("\n" + "=" * 60)
    print("USAGE EXAMPLES")
    print("=" * 60)
    print("""
# To generate custom surfaces, use generate_custom_surface():

from surface_pointcloud_generator import generate_custom_surface

# Isotropic fine surface (e.g., polished)
generate_custom_surface(
    "polished_surface",
    sa_roughness=0.0002,      # 0.2 μm
    correlation_length=0.01,  # 10 μm features
    n_points=100000
)

# Anisotropic surface (e.g., turned surface with lay direction)
generate_custom_surface(
    "turned_surface",
    sa_roughness=0.002,       # 2 μm
    correlation_length=0.03,  # 30 μm in x
    anisotropy_ratio=5.0,     # 150 μm in y (elongated marks)
    n_points=50000
)

# Coarse ground surface
generate_custom_surface(
    "ground_coarse",
    sa_roughness=0.001,       # 1 μm
    correlation_length=0.08,  # 80 μm features
    n_points=30000
)
""")
