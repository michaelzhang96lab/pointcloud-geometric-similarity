"""
Geometric Similarity Comparison for 3D Surface Texture Point Clouds

This module compares point clouds based on intrinsic geometric properties
that are POSE-INVARIANT (rotation and translation do not affect results).

Methods implemented:
1. CURVATURE DISTRIBUTIONS (Primary) - Compare distributions of principal curvatures
2. FPFH (Validation) - Fast Point Feature Histograms from computer vision
3. D2 SHAPE DISTRIBUTION (Baseline) - Pairwise distance distributions

Author: Zhongyi Michael Zhang
Purpose: Demonstrate pose-invariant geometric comparison for surface textures.

Explanatory Note: 
A part of the data fusion pipeline I proposed in my PhD thesis is to compare two 3D point clouds,
both of which shows the surface textures of a small area on an engineered product but are in drastically 
different point densities, and assess their geometric similarity. The original method in my thesis is voxelising
the 3D space occupied by both point clouds, and then calculating the percentage of points in each point cloud falling
into every voxel; the next step is comparing the percentage of points in each point cloud in each voxel, then 
counting the number of voxels having similar percentages of the two point clouds. This method is prone to noise
and the imperfect alignment of the two point clouds. In this script, I attempt three other methods which are
tested with industrial scenarios and mathematically heavier. These three methods should provide a more comprehensive
and robust assessment of the geometric similarity between two 3D point clouds in different point densities.

"""

import numpy as np
from scipy import stats
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from typing import Tuple, Dict, List, Optional
import warnings


# =============================================================================
# POINT CLOUD I/O
# =============================================================================

def load_ply(filepath: str) -> np.ndarray:
    """
    Load point cloud from PLY file.
    
    Parameters:
    -----------
    filepath : str
        Path to PLY file
        
    Returns:
    --------
    points : np.ndarray
        Nx3 array of (x, y, z) coordinates
    """
    points = []
    header_ended = False
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not header_ended:
                if line == 'end_header':
                    header_ended = True
                continue
            
            parts = line.split()
            if len(parts) >= 3:
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
    
    return np.array(points)


def load_csv(filepath: str) -> np.ndarray:
    """Load point cloud from CSV file."""
    return np.loadtxt(filepath, delimiter=',', skiprows=1)


# =============================================================================
# NORMAL ESTIMATION
# =============================================================================

def estimate_normals(points: np.ndarray, k_neighbours: int = 30) -> np.ndarray:
    """
    Estimate surface normals using PCA on local neighbourhoods.
    
    The normal at each point is the eigenvector corresponding to the
    smallest eigenvalue of the covariance matrix of its k nearest neighbours.
    
    Parameters:
    -----------
    points : np.ndarray
        Nx3 array of points
    k_neighbours : int
        Number of neighbours for local PCA
        
    Returns:
    --------
    normals : np.ndarray
        Nx3 array of unit normal vectors
    """
    n_points = len(points)
    normals = np.zeros((n_points, 3))
    
    # Build KD-tree for efficient neighbour queries
    tree = cKDTree(points)
    
    for i in range(n_points):
        # Find k nearest neighbours
        _, idx = tree.query(points[i], k=k_neighbours)
        neighbours = points[idx]
        
        # Center the neighbourhood
        centroid = neighbours.mean(axis=0)
        centered = neighbours - centroid
        
        # PCA via covariance matrix
        cov = np.dot(centered.T, centered) / (k_neighbours - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Normal is eigenvector with smallest eigenvalue
        # (direction of least variance = perpendicular to surface)
        normals[i] = eigenvectors[:, 0]
    
    # Consistent orientation: flip normals to point "up" (positive z on average)
    # For surface textures, this is reasonable
    if np.mean(normals[:, 2]) < 0:
        normals = -normals
    
    return normals


# =============================================================================
# METHOD 1: CURVATURE DISTRIBUTIONS (PRIMARY)
# =============================================================================

def estimate_curvatures(
    points: np.ndarray, 
    normals: np.ndarray,
    k_neighbours: int = 30
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate principal curvatures at each point using quadric surface fitting.
    
    At each point, we fit a local quadratic surface z = ax² + bxy + cy² + dx + ey + f
    in a coordinate system aligned with the normal. The principal curvatures
    are derived from the Hessian of this quadratic.
    
    Parameters:
    -----------
    points : np.ndarray
        Nx3 array of points
    normals : np.ndarray
        Nx3 array of unit normals
    k_neighbours : int
        Number of neighbours for local surface fitting
        
    Returns:
    --------
    k1 : np.ndarray
        Maximum principal curvature at each point
    k2 : np.ndarray
        Minimum principal curvature at each point
    H : np.ndarray
        Mean curvature = (k1 + k2) / 2
    K : np.ndarray
        Gaussian curvature = k1 * k2
    """
    n_points = len(points)
    k1 = np.zeros(n_points)
    k2 = np.zeros(n_points)
    
    tree = cKDTree(points)
    
    for i in range(n_points):
        # Find neighbours
        _, idx = tree.query(points[i], k=k_neighbours)
        neighbours = points[idx]
        
        # Local coordinate system: normal is z-axis
        n = normals[i]
        
        # Find two orthogonal vectors in the tangent plane
        if abs(n[0]) < abs(n[1]):
            t1 = np.array([0, -n[2], n[1]])
        else:
            t1 = np.array([-n[2], 0, n[0]])
        t1 = t1 / np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        
        # Rotation matrix: global -> local
        R = np.vstack([t1, t2, n])
        
        # Transform neighbours to local coordinates
        centered = neighbours - points[i]
        local = np.dot(centered, R.T)
        
        # Fit quadric: z = ax² + bxy + cy² + dx + ey + f
        # In matrix form: [x², xy, y², x, y, 1] @ [a,b,c,d,e,f].T = z
        x, y, z = local[:, 0], local[:, 1], local[:, 2]
        
        A = np.column_stack([x**2, x*y, y**2, x, y, np.ones_like(x)])
        
        try:
            # Least squares fit
            coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
            a, b, c = coeffs[0], coeffs[1], coeffs[2]
            
            # Principal curvatures from the shape operator
            # For z = ax² + bxy + cy², the curvatures are eigenvalues of:
            # [[2a, b], [b, 2c]]
            shape_operator = np.array([[2*a, b], [b, 2*c]])
            curvatures = np.linalg.eigvalsh(shape_operator)
            
            k1[i] = curvatures[1]  # max
            k2[i] = curvatures[0]  # min
            
        except np.linalg.LinAlgError:
            k1[i] = 0
            k2[i] = 0
    
    H = (k1 + k2) / 2  # Mean curvature
    K = k1 * k2        # Gaussian curvature
    
    return k1, k2, H, K


def curvature_histogram(
    curvatures: np.ndarray, 
    n_bins: int = 50,
    range_percentile: float = 99
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute normalized histogram of curvature values.
    
    Parameters:
    -----------
    curvatures : np.ndarray
        Array of curvature values
    n_bins : int
        Number of histogram bins
    range_percentile : float
        Use this percentile to set histogram range (removes outliers)
        
    Returns:
    --------
    hist : np.ndarray
        Normalized histogram (sums to 1)
    bin_centers : np.ndarray
        Center values of each bin
    """
    # Robust range estimation
    low = np.percentile(curvatures, 100 - range_percentile)
    high = np.percentile(curvatures, range_percentile)
    
    hist, bin_edges = np.histogram(curvatures, bins=n_bins, range=(low, high), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Normalize to sum to 1 (probability distribution)
    hist = hist / hist.sum() if hist.sum() > 0 else hist
    
    return hist, bin_centers


def compare_distributions(hist1: np.ndarray, hist2: np.ndarray) -> Dict[str, float]:
    """
    Compare two histograms using multiple metrics.
    
    Parameters:
    -----------
    hist1, hist2 : np.ndarray
        Normalized histograms (must have same shape)
        
    Returns:
    --------
    metrics : dict
        Dictionary containing various similarity/distance metrics
    """
    # Ensure valid probability distributions
    h1 = hist1 / hist1.sum() if hist1.sum() > 0 else hist1 + 1e-10
    h2 = hist2 / hist2.sum() if hist2.sum() > 0 else hist2 + 1e-10
    
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    h1 = h1 + eps
    h2 = h2 + eps
    h1 = h1 / h1.sum()
    h2 = h2 / h2.sum()
    
    metrics = {}
    
    # 1. Histogram Intersection (higher = more similar, max = 1)
    metrics['intersection'] = np.sum(np.minimum(h1, h2))
    
    # 2. Bhattacharyya Coefficient (higher = more similar, max = 1)
    metrics['bhattacharyya'] = np.sum(np.sqrt(h1 * h2))
    
    # 3. Chi-squared distance (lower = more similar)
    metrics['chi_squared'] = 0.5 * np.sum((h1 - h2)**2 / (h1 + h2 + eps))
    
    # 4. Jensen-Shannon Divergence (lower = more similar, range [0, ln(2)])
    m = (h1 + h2) / 2
    kl1 = np.sum(h1 * np.log(h1 / m))
    kl2 = np.sum(h2 * np.log(h2 / m))
    metrics['jensen_shannon'] = (kl1 + kl2) / 2
    
    # 5. Earth Mover's Distance / Wasserstein-1 (lower = more similar)
    # For 1D histograms, this is the L1 distance between CDFs
    cdf1 = np.cumsum(h1)
    cdf2 = np.cumsum(h2)
    metrics['emd'] = np.sum(np.abs(cdf1 - cdf2)) / len(h1)
    
    # 6. Correlation (higher = more similar, range [-1, 1])
    metrics['correlation'] = np.corrcoef(h1, h2)[0, 1]
    
    return metrics


def curvature_similarity(
    points1: np.ndarray, 
    points2: np.ndarray,
    k_neighbours: int = 30,
    n_bins: int = 50
) -> Dict[str, any]:
    """
    Compare two point clouds using curvature distributions.
    
    Parameters:
    -----------
    points1, points2 : np.ndarray
        Point clouds to compare
    k_neighbours : int
        Neighbours for normal/curvature estimation
    n_bins : int
        Histogram bins
        
    Returns:
    --------
    result : dict
        Contains curvature arrays, histograms, and comparison metrics
    """
    print("  Estimating normals for cloud 1...")
    normals1 = estimate_normals(points1, k_neighbours)
    print("  Estimating normals for cloud 2...")
    normals2 = estimate_normals(points2, k_neighbours)
    
    print("  Computing curvatures for cloud 1...")
    k1_1, k2_1, H1, K1 = estimate_curvatures(points1, normals1, k_neighbours)
    print("  Computing curvatures for cloud 2...")
    k1_2, k2_2, H2, K2 = estimate_curvatures(points2, normals2, k_neighbours)
    
    # Build histograms for mean curvature (most informative for textures)
    # Use common range for fair comparison
    all_H = np.concatenate([H1, H2])
    h_low = np.percentile(all_H, 1)
    h_high = np.percentile(all_H, 99)
    
    hist_H1, _ = np.histogram(H1, bins=n_bins, range=(h_low, h_high), density=True)
    hist_H2, _ = np.histogram(H2, bins=n_bins, range=(h_low, h_high), density=True)
    
    # Normalize
    hist_H1 = hist_H1 / hist_H1.sum() if hist_H1.sum() > 0 else hist_H1
    hist_H2 = hist_H2 / hist_H2.sum() if hist_H2.sum() > 0 else hist_H2
    
    # Compare
    metrics = compare_distributions(hist_H1, hist_H2)
    
    # Also compare Gaussian curvature
    all_K = np.concatenate([K1, K2])
    k_low = np.percentile(all_K, 1)
    k_high = np.percentile(all_K, 99)
    
    hist_K1, _ = np.histogram(K1, bins=n_bins, range=(k_low, k_high), density=True)
    hist_K2, _ = np.histogram(K2, bins=n_bins, range=(k_low, k_high), density=True)
    hist_K1 = hist_K1 / hist_K1.sum() if hist_K1.sum() > 0 else hist_K1
    hist_K2 = hist_K2 / hist_K2.sum() if hist_K2.sum() > 0 else hist_K2
    
    metrics_K = compare_distributions(hist_K1, hist_K2)
    
    return {
        'mean_curvature': {
            'cloud1': {'values': H1, 'histogram': hist_H1},
            'cloud2': {'values': H2, 'histogram': hist_H2},
            'metrics': metrics
        },
        'gaussian_curvature': {
            'cloud1': {'values': K1, 'histogram': hist_K1},
            'cloud2': {'values': K2, 'histogram': hist_K2},
            'metrics': metrics_K
        }
    }


# =============================================================================
# METHOD 2: D2 SHAPE DISTRIBUTION (BASELINE)
# =============================================================================

def d2_shape_distribution(
    points: np.ndarray, 
    n_samples: int = 10000,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Compute D2 shape distribution: histogram of pairwise distances.
    
    Parameters:
    -----------
    points : np.ndarray
        Nx3 point cloud
    n_samples : int
        Number of point pairs to sample
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    distances : np.ndarray
        Array of sampled pairwise distances
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_points = len(points)
    
    # Random sampling of pairs
    idx1 = np.random.randint(0, n_points, n_samples)
    idx2 = np.random.randint(0, n_points, n_samples)
    
    # Compute distances
    distances = np.linalg.norm(points[idx1] - points[idx2], axis=1)
    
    # Remove self-pairs (distance = 0)
    distances = distances[distances > 0]
    
    return distances


def d2_similarity(
    points1: np.ndarray, 
    points2: np.ndarray,
    n_samples: int = 50000,
    n_bins: int = 100
) -> Dict[str, any]:
    """
    Compare two point clouds using D2 shape distributions.
    
    Parameters:
    -----------
    points1, points2 : np.ndarray
        Point clouds to compare
    n_samples : int
        Number of point pairs to sample from each cloud
    n_bins : int
        Histogram bins
        
    Returns:
    --------
    result : dict
        Contains distributions and comparison metrics
    """
    print("  Computing D2 distribution for cloud 1...")
    d2_1 = d2_shape_distribution(points1, n_samples, seed=42)
    print("  Computing D2 distribution for cloud 2...")
    d2_2 = d2_shape_distribution(points2, n_samples, seed=43)
    
    # Common range
    all_d = np.concatenate([d2_1, d2_2])
    d_low, d_high = np.percentile(all_d, 1), np.percentile(all_d, 99)
    
    hist1, bin_edges = np.histogram(d2_1, bins=n_bins, range=(d_low, d_high), density=True)
    hist2, _ = np.histogram(d2_2, bins=n_bins, range=(d_low, d_high), density=True)
    
    hist1 = hist1 / hist1.sum() if hist1.sum() > 0 else hist1
    hist2 = hist2 / hist2.sum() if hist2.sum() > 0 else hist2
    
    metrics = compare_distributions(hist1, hist2)
    
    return {
        'cloud1': {'distances': d2_1, 'histogram': hist1},
        'cloud2': {'distances': d2_2, 'histogram': hist2},
        'bin_edges': bin_edges,
        'metrics': metrics
    }


# =============================================================================
# METHOD 3: FPFH (VALIDATION)
# =============================================================================

def compute_fpfh(
    points: np.ndarray, 
    normals: np.ndarray,
    radius: float,
    n_bins: int = 11
) -> np.ndarray:
    """
    Compute Fast Point Feature Histograms (simplified implementation).
    
    FPFH encodes the angular relationships between normals in a local
    neighbourhood. For each point pair (p, q) with normals (n_p, n_q),
    we compute three angles:
    - α: angle between n_p and (q-p)
    - φ: angle between n_q and (q-p)  
    - θ: angle between n_p and n_q
    
    These are binned into histograms and aggregated.
    
    Parameters:
    -----------
    points : np.ndarray
        Nx3 point cloud
    normals : np.ndarray
        Nx3 unit normals
    radius : float
        Neighbourhood radius
    n_bins : int
        Bins per angular dimension (total histogram size = 3 * n_bins)
        
    Returns:
    --------
    fpfh : np.ndarray
        Nx(3*n_bins) array of FPFH descriptors
    """
    n_points = len(points)
    fpfh = np.zeros((n_points, 3 * n_bins))
    
    tree = cKDTree(points)
    
    for i in range(n_points):
        # Find neighbours within radius
        idx = tree.query_ball_point(points[i], radius)
        
        if len(idx) < 2:
            continue
        
        # Remove self
        idx = [j for j in idx if j != i]
        
        if len(idx) == 0:
            continue
        
        alphas, phis, thetas = [], [], []
        
        for j in idx:
            # Vector from p to q
            d = points[j] - points[i]
            d_norm = np.linalg.norm(d)
            if d_norm < 1e-10:
                continue
            d = d / d_norm
            
            n_p = normals[i]
            n_q = normals[j]
            
            # Compute angles
            # α: angle between n_p and d
            alpha = np.arctan2(np.linalg.norm(np.cross(n_p, d)), np.dot(n_p, d))
            
            # φ: angle between n_q and d
            phi = np.arctan2(np.linalg.norm(np.cross(n_q, d)), np.dot(n_q, d))
            
            # θ: angle between n_p and n_q
            theta = np.arctan2(np.linalg.norm(np.cross(n_p, n_q)), np.dot(n_p, n_q))
            
            alphas.append(alpha)
            phis.append(phi)
            thetas.append(theta)
        
        if len(alphas) == 0:
            continue
        
        # Build histograms
        hist_alpha, _ = np.histogram(alphas, bins=n_bins, range=(0, np.pi))
        hist_phi, _ = np.histogram(phis, bins=n_bins, range=(0, np.pi))
        hist_theta, _ = np.histogram(thetas, bins=n_bins, range=(0, np.pi))
        
        # Concatenate
        fpfh[i, :n_bins] = hist_alpha
        fpfh[i, n_bins:2*n_bins] = hist_phi
        fpfh[i, 2*n_bins:] = hist_theta
    
    # Normalize each descriptor
    row_sums = fpfh.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    fpfh = fpfh / row_sums
    
    return fpfh


def fpfh_similarity(
    points1: np.ndarray, 
    points2: np.ndarray,
    radius: Optional[float] = None,
    n_bins: int = 11
) -> Dict[str, any]:
    """
    Compare two point clouds using FPFH descriptors.
    
    We compute FPFH for each point, then compare the distributions
    of descriptors between clouds.
    
    Parameters:
    -----------
    points1, points2 : np.ndarray
        Point clouds to compare
    radius : float, optional
        Neighbourhood radius. If None, auto-computed.
    n_bins : int
        Bins per angular dimension
        
    Returns:
    --------
    result : dict
        Contains FPFH arrays and comparison metrics
    """
    # Auto-compute radius based on point density
    if radius is None:
        # Estimate based on average nearest-neighbour distance
        tree1 = cKDTree(points1)
        dists, _ = tree1.query(points1, k=2)
        avg_spacing1 = np.mean(dists[:, 1])
        
        tree2 = cKDTree(points2)
        dists, _ = tree2.query(points2, k=2)
        avg_spacing2 = np.mean(dists[:, 1])
        
        radius = max(avg_spacing1, avg_spacing2) * 5
        print(f"  Auto-computed FPFH radius: {radius:.6f}")
    
    print("  Estimating normals for cloud 1...")
    normals1 = estimate_normals(points1, k_neighbours=30)
    print("  Estimating normals for cloud 2...")
    normals2 = estimate_normals(points2, k_neighbours=30)
    
    print("  Computing FPFH for cloud 1...")
    fpfh1 = compute_fpfh(points1, normals1, radius, n_bins)
    print("  Computing FPFH for cloud 2...")
    fpfh2 = compute_fpfh(points2, normals2, radius, n_bins)
    
    # Global descriptor: average FPFH
    global_fpfh1 = fpfh1.mean(axis=0)
    global_fpfh2 = fpfh2.mean(axis=0)
    
    # Normalize
    global_fpfh1 = global_fpfh1 / global_fpfh1.sum() if global_fpfh1.sum() > 0 else global_fpfh1
    global_fpfh2 = global_fpfh2 / global_fpfh2.sum() if global_fpfh2.sum() > 0 else global_fpfh2
    
    metrics = compare_distributions(global_fpfh1, global_fpfh2)
    
    return {
        'cloud1': {'fpfh': fpfh1, 'global': global_fpfh1},
        'cloud2': {'fpfh': fpfh2, 'global': global_fpfh2},
        'radius': radius,
        'metrics': metrics
    }


# =============================================================================
# MAIN COMPARISON FUNCTION
# =============================================================================

def compare_point_clouds(
    points1: np.ndarray,
    points2: np.ndarray,
    name1: str = "Cloud 1",
    name2: str = "Cloud 2",
    methods: List[str] = ['curvature', 'd2', 'fpfh']
) -> Dict[str, any]:
    """
    Compare two point clouds using multiple pose-invariant methods.
    
    Parameters:
    -----------
    points1, points2 : np.ndarray
        Point clouds to compare
    name1, name2 : str
        Names for display
    methods : list
        Which methods to use: 'curvature', 'd2', 'fpfh'
        
    Returns:
    --------
    results : dict
        Results from each method
    """
    print(f"\n{'='*60}")
    print(f"COMPARING: {name1} vs {name2}")
    print(f"{'='*60}")
    print(f"  Cloud 1: {len(points1):,} points")
    print(f"  Cloud 2: {len(points2):,} points")
    
    results = {}
    
    if 'curvature' in methods:
        print(f"\n[METHOD 1: CURVATURE DISTRIBUTIONS]")
        results['curvature'] = curvature_similarity(points1, points2)
        
    if 'd2' in methods:
        print(f"\n[METHOD 2: D2 SHAPE DISTRIBUTION]")
        results['d2'] = d2_similarity(points1, points2)
        
    if 'fpfh' in methods:
        print(f"\n[METHOD 3: FPFH]")
        results['fpfh'] = fpfh_similarity(points1, points2)
    
    return results


def print_comparison_summary(results: Dict[str, any], name1: str, name2: str):
    """Print a formatted summary of comparison results."""
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: {name1} vs {name2}")
    print(f"{'='*60}")
    
    if 'curvature' in results:
        print("\n[CURVATURE DISTRIBUTIONS]")
        m = results['curvature']['mean_curvature']['metrics']
        print(f"  Mean Curvature:")
        print(f"    - Histogram Intersection: {m['intersection']:.4f} (1.0 = identical)")
        print(f"    - Bhattacharyya Coeff:    {m['bhattacharyya']:.4f} (1.0 = identical)")
        print(f"    - Jensen-Shannon Div:     {m['jensen_shannon']:.4f} (0.0 = identical)")
        print(f"    - Earth Mover's Distance: {m['emd']:.4f} (0.0 = identical)")
        
        m = results['curvature']['gaussian_curvature']['metrics']
        print(f"  Gaussian Curvature:")
        print(f"    - Histogram Intersection: {m['intersection']:.4f}")
        print(f"    - Bhattacharyya Coeff:    {m['bhattacharyya']:.4f}")
    
    if 'd2' in results:
        print("\n[D2 SHAPE DISTRIBUTION]")
        m = results['d2']['metrics']
        print(f"    - Histogram Intersection: {m['intersection']:.4f}")
        print(f"    - Bhattacharyya Coeff:    {m['bhattacharyya']:.4f}")
        print(f"    - Jensen-Shannon Div:     {m['jensen_shannon']:.4f}")
    
    if 'fpfh' in results:
        print("\n[FPFH]")
        m = results['fpfh']['metrics']
        print(f"    - Histogram Intersection: {m['intersection']:.4f}")
        print(f"    - Bhattacharyya Coeff:    {m['bhattacharyya']:.4f}")
        print(f"    - Jensen-Shannon Div:     {m['jensen_shannon']:.4f}")


# =============================================================================
# TEST ON SYNTHETIC DATA
# =============================================================================

def run_test():
    """Test the comparison methods on synthetic surface data."""
    
    import os
    
    # Path to synthetic surfaces
    data_dir = "synthetic_surfaces"
    
    print("="*60)
    print("GEOMETRIC SIMILARITY COMPARISON TEST")
    print("="*60)
    
    # Load point clouds
    print("\nLoading point clouds...")
    cloud_a = load_ply(os.path.join(data_dir, "cloud_A_high_density_texture1.ply"))
    cloud_b = load_ply(os.path.join(data_dir, "cloud_B_low_density_texture1.ply"))
    cloud_c = load_ply(os.path.join(data_dir, "cloud_C_low_density_texture2.ply"))
    
    print(f"  Cloud A: {len(cloud_a):,} points (high density, texture 1)")
    print(f"  Cloud B: {len(cloud_b):,} points (low density, texture 1)")
    print(f"  Cloud C: {len(cloud_c):,} points (low density, texture 2)")
    
    # Expected results:
    # A vs B: HIGH similarity (same texture, different density)
    # A vs C: LOW similarity (different texture)
    # B vs C: LOW similarity (different texture, same density)
    
    print("\n" + "="*60)
    print("EXPECTED RESULTS:")
    print("  A vs B: HIGH similarity (same texture)")
    print("  A vs C: LOW similarity (different texture)")
    print("  B vs C: LOW similarity (different texture)")
    print("="*60)
    
    # For faster testing, subsample cloud A
    print("\nSubsampling Cloud A for faster computation...")
    np.random.seed(42)
    idx_a = np.random.choice(len(cloud_a), size=25000, replace=False)
    cloud_a_sub = cloud_a[idx_a]
    print(f"  Cloud A subsampled: {len(cloud_a_sub):,} points")
    
    # Run comparisons
    results_ab = compare_point_clouds(
        cloud_a_sub, cloud_b, 
        "Cloud A (texture 1)", "Cloud B (texture 1)"
    )
    print_comparison_summary(results_ab, "A", "B")
    
    results_ac = compare_point_clouds(
        cloud_a_sub, cloud_c,
        "Cloud A (texture 1)", "Cloud C (texture 2)"
    )
    print_comparison_summary(results_ac, "A", "C")
    
    results_bc = compare_point_clouds(
        cloud_b, cloud_c,
        "Cloud B (texture 1)", "Cloud C (texture 2)"
    )
    print_comparison_summary(results_bc, "B", "C")
    
    # Final verdict
    print("\n" + "="*60)
    print("FINAL VERDICT")
    print("="*60)
    
    # Use mean curvature Bhattacharyya coefficient as primary metric
    sim_ab = results_ab['curvature']['mean_curvature']['metrics']['bhattacharyya']
    sim_ac = results_ac['curvature']['mean_curvature']['metrics']['bhattacharyya']
    sim_bc = results_bc['curvature']['mean_curvature']['metrics']['bhattacharyya']
    
    print(f"\nCurvature-based similarity (Bhattacharyya, higher = more similar):")
    print(f"  A vs B: {sim_ab:.4f}  {'✓ HIGH' if sim_ab > 0.9 else '? UNEXPECTED'}")
    print(f"  A vs C: {sim_ac:.4f}  {'✓ LOW' if sim_ac < sim_ab else '? UNEXPECTED'}")
    print(f"  B vs C: {sim_bc:.4f}  {'✓ LOW' if sim_bc < sim_ab else '? UNEXPECTED'}")
    
    if sim_ab > sim_ac and sim_ab > sim_bc:
        print("\n✓ SUCCESS: Same-texture pairs score higher than different-texture pairs.")
    else:
        print("\n✗ UNEXPECTED: Results do not match expectations. May need parameter tuning.")
    
    return results_ab, results_ac, results_bc


if __name__ == "__main__":
    results_ab, results_ac, results_bc = run_test()
