"""
Visualization for Geometric Similarity Comparison Results

Creates publication-quality figures showing:
1. Curvature distributions for each cloud
2. Comparison of all three methods
3. Summary bar chart
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# Import from our module
from geometric_similarity import ( # type: ignore
    load_ply, estimate_normals, estimate_curvatures,
    d2_shape_distribution, curvature_histogram
)


def create_comparison_visualization():
    """Create comprehensive visualization of the comparison results."""
    
    # Load data
    data_dir = "synthetic_surfaces"
    print("Loading point clouds...")
    cloud_a = load_ply(os.path.join(data_dir, "cloud_A_high_density_texture1.ply"))
    cloud_b = load_ply(os.path.join(data_dir, "cloud_B_low_density_texture1.ply"))
    cloud_c = load_ply(os.path.join(data_dir, "cloud_C_low_density_texture2.ply"))
    
    # Subsample A for computation
    np.random.seed(42)
    idx_a = np.random.choice(len(cloud_a), size=25000, replace=False)
    cloud_a_sub = cloud_a[idx_a]
    
    print("Computing normals...")
    normals_a = estimate_normals(cloud_a_sub, k_neighbours=30)
    normals_b = estimate_normals(cloud_b, k_neighbours=30)
    normals_c = estimate_normals(cloud_c, k_neighbours=30)
    
    print("Computing curvatures...")
    k1_a, k2_a, H_a, K_a = estimate_curvatures(cloud_a_sub, normals_a, k_neighbours=30)
    k1_b, k2_b, H_b, K_b = estimate_curvatures(cloud_b, normals_b, k_neighbours=30)
    k1_c, k2_c, H_c, K_c = estimate_curvatures(cloud_c, normals_c, k_neighbours=30)
    
    # =========================================================================
    # Figure 1: Curvature Distributions
    # =========================================================================
    
    fig1, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    # Common range for fair comparison
    all_H = np.concatenate([H_a, H_b, H_c])
    h_range = (np.percentile(all_H, 1), np.percentile(all_H, 99))
    
    all_K = np.concatenate([K_a, K_b, K_c])
    k_range = (np.percentile(all_K, 1), np.percentile(all_K, 99))
    
    n_bins = 50
    
    # Row 1: Mean Curvature
    axes[0, 0].hist(H_a, bins=n_bins, range=h_range, density=True, alpha=0.7, 
                    color='blue', edgecolor='black', linewidth=0.5)
    axes[0, 0].set_title('Cloud A: Mean Curvature\n(High density, Texture 1)', fontsize=11)
    axes[0, 0].set_xlabel('Mean Curvature H (mm⁻¹)')
    axes[0, 0].set_ylabel('Probability Density')
    axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.5)
    
    axes[0, 1].hist(H_b, bins=n_bins, range=h_range, density=True, alpha=0.7,
                    color='green', edgecolor='black', linewidth=0.5)
    axes[0, 1].set_title('Cloud B: Mean Curvature\n(Low density, Texture 1)', fontsize=11)
    axes[0, 1].set_xlabel('Mean Curvature H (mm⁻¹)')
    axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.5)
    
    axes[0, 2].hist(H_c, bins=n_bins, range=h_range, density=True, alpha=0.7,
                    color='orange', edgecolor='black', linewidth=0.5)
    axes[0, 2].set_title('Cloud C: Mean Curvature\n(Low density, Texture 2)', fontsize=11)
    axes[0, 2].set_xlabel('Mean Curvature H (mm⁻¹)')
    axes[0, 2].axvline(0, color='red', linestyle='--', alpha=0.5)
    
    # Row 2: Gaussian Curvature
    axes[1, 0].hist(K_a, bins=n_bins, range=k_range, density=True, alpha=0.7,
                    color='blue', edgecolor='black', linewidth=0.5)
    axes[1, 0].set_title('Cloud A: Gaussian Curvature', fontsize=11)
    axes[1, 0].set_xlabel('Gaussian Curvature K (mm⁻²)')
    axes[1, 0].set_ylabel('Probability Density')
    axes[1, 0].axvline(0, color='red', linestyle='--', alpha=0.5)
    
    axes[1, 1].hist(K_b, bins=n_bins, range=k_range, density=True, alpha=0.7,
                    color='green', edgecolor='black', linewidth=0.5)
    axes[1, 1].set_title('Cloud B: Gaussian Curvature', fontsize=11)
    axes[1, 1].set_xlabel('Gaussian Curvature K (mm⁻²)')
    axes[1, 1].axvline(0, color='red', linestyle='--', alpha=0.5)
    
    axes[1, 2].hist(K_c, bins=n_bins, range=k_range, density=True, alpha=0.7,
                    color='orange', edgecolor='black', linewidth=0.5)
    axes[1, 2].set_title('Cloud C: Gaussian Curvature', fontsize=11)
    axes[1, 2].set_xlabel('Gaussian Curvature K (mm⁻²)')
    axes[1, 2].axvline(0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('synthetic_surfaces/curvature_distributions.png', dpi=150, bbox_inches='tight')
    print("Saved: curvature_distributions.png")
    plt.close()
    
    # =========================================================================
    # Figure 2: Overlay Comparison
    # =========================================================================
    
    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Mean curvature overlay
    axes[0].hist(H_a, bins=n_bins, range=h_range, density=True, alpha=0.5,
                 color='blue', label='A (Texture 1)')
    axes[0].hist(H_b, bins=n_bins, range=h_range, density=True, alpha=0.5,
                 color='green', label='B (Texture 1)')
    axes[0].hist(H_c, bins=n_bins, range=h_range, density=True, alpha=0.5,
                 color='orange', label='C (Texture 2)')
    axes[0].set_title('Mean Curvature Distribution Comparison\n(A and B should overlap; C should differ)', fontsize=11)
    axes[0].set_xlabel('Mean Curvature H (mm⁻¹)')
    axes[0].set_ylabel('Probability Density')
    axes[0].legend()
    axes[0].axvline(0, color='red', linestyle='--', alpha=0.3)
    
    # Gaussian curvature overlay
    axes[1].hist(K_a, bins=n_bins, range=k_range, density=True, alpha=0.5,
                 color='blue', label='A (Texture 1)')
    axes[1].hist(K_b, bins=n_bins, range=k_range, density=True, alpha=0.5,
                 color='green', label='B (Texture 1)')
    axes[1].hist(K_c, bins=n_bins, range=k_range, density=True, alpha=0.5,
                 color='orange', label='C (Texture 2)')
    axes[1].set_title('Gaussian Curvature Distribution Comparison\n(A and B should overlap; C should differ)', fontsize=11)
    axes[1].set_xlabel('Gaussian Curvature K (mm⁻²)')
    axes[1].set_ylabel('Probability Density')
    axes[1].legend()
    axes[1].axvline(0, color='red', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('synthetic_surfaces/curvature_comparison_overlay.png', dpi=150, bbox_inches='tight')
    print("Saved: curvature_comparison_overlay.png")
    plt.close()
    
    # =========================================================================
    # Figure 3: Summary Bar Chart
    # =========================================================================
    
    # These are the results from running the comparison
    results = {
        'A vs B': {
            'curvature_H': 0.9996,
            'curvature_K': 0.9995,
            'd2': 0.9995,
            'fpfh': 1.0000,
            'expected': 'HIGH'
        },
        'A vs C': {
            'curvature_H': 0.9809,
            'curvature_K': 0.9642,
            'd2': 0.9995,
            'fpfh': 0.9940,
            'expected': 'LOW'
        },
        'B vs C': {
            'curvature_H': 0.9817,
            'curvature_K': 0.9671,
            'd2': 0.9995,
            'fpfh': 0.9941,
            'expected': 'LOW'
        }
    }
    
    fig3, ax = plt.subplots(figsize=(10, 6))
    
    comparisons = list(results.keys())
    x = np.arange(len(comparisons))
    width = 0.2
    
    # Extract data
    curv_H = [results[c]['curvature_H'] for c in comparisons]
    curv_K = [results[c]['curvature_K'] for c in comparisons]
    d2_vals = [results[c]['d2'] for c in comparisons]
    fpfh_vals = [results[c]['fpfh'] for c in comparisons]
    
    # Plot bars
    bars1 = ax.bar(x - 1.5*width, curv_H, width, label='Mean Curvature', color='#2ecc71')
    bars2 = ax.bar(x - 0.5*width, curv_K, width, label='Gaussian Curvature', color='#27ae60')
    bars3 = ax.bar(x + 0.5*width, d2_vals, width, label='D2 (baseline)', color='#95a5a6')
    bars4 = ax.bar(x + 1.5*width, fpfh_vals, width, label='FPFH', color='#3498db')
    
    # Formatting
    ax.set_ylabel('Bhattacharyya Coefficient\n(higher = more similar)', fontsize=11)
    ax.set_title('Geometric Similarity Comparison Results\n(Comparing Same-Texture vs Different-Texture Pairs)', fontsize=12, pad=30)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n({results[c]['expected']} expected)" for c in comparisons])
    ax.legend(loc='lower right')
    ax.set_ylim(0.9, 1.02)  # Increased upper limit to give more room for labels
    
    # Add value labels
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, rotation=90)
    
    # Add horizontal line showing threshold
    ax.axhline(y=0.99, color='red', linestyle='--', alpha=0.5, label='Discrimination threshold')
    
    plt.tight_layout()
    plt.savefig('synthetic_surfaces/similarity_summary.png', dpi=150, bbox_inches='tight')
    print("Saved: similarity_summary.png")
    plt.close()
    
    # =========================================================================
    # Figure 4: Method Comparison Table (as figure)
    # =========================================================================
    
    fig4, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    
    # Create table data
    table_data = [
        ['Comparison', 'Mean Curv.', 'Gauss. Curv.', 'D2', 'FPFH', 'Expected', 'Verdict'],
        ['A vs B\n(same texture)', '0.9996', '0.9995', '0.9995', '1.0000', 'HIGH', '✓'],
        ['A vs C\n(diff. texture)', '0.9809', '0.9642', '0.9995', '0.9940', 'LOW', '✓'],
        ['B vs C\n(diff. texture)', '0.9817', '0.9671', '0.9995', '0.9941', 'LOW', '✓'],
    ]
    
    # Create table
    table = ax.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc='center',
        cellLoc='center',
        colColours=['#f0f0f0']*7
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Color code cells
    for i in range(1, 4):
        # Mean curvature column - highlight discrimination
        if i == 1:
            table[(i, 1)].set_facecolor('#d4edda')  # Green for high
        else:
            table[(i, 1)].set_facecolor('#f8d7da')  # Red for low
        
        # Gaussian curvature
        if i == 1:
            table[(i, 2)].set_facecolor('#d4edda')
        else:
            table[(i, 2)].set_facecolor('#f8d7da')
        
        # D2 - all same (failure)
        table[(i, 3)].set_facecolor('#fff3cd')  # Yellow for "no discrimination"
        
        # FPFH
        if i == 1:
            table[(i, 4)].set_facecolor('#d4edda')
        else:
            table[(i, 4)].set_facecolor('#cce5ff')  # Light blue for slight difference
        
        # Verdict
        table[(i, 6)].set_facecolor('#d4edda')
    
    ax.set_title('Similarity Scores (Bhattacharyya Coefficient)\n\n', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('synthetic_surfaces/results_table.png', dpi=150, bbox_inches='tight')
    print("Saved: results_table.png")
    plt.close()
    
    # =========================================================================
    # Figure 5: Visual explanation of why curvature works
    # =========================================================================
    
    fig5, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Show curvature mapped onto point cloud (top view)
    def scatter_curvature(ax, points, curvatures, title, vmin, vmax):
        sc = ax.scatter(points[:, 0], points[:, 1], c=curvatures, 
                       cmap='RdBu_r', s=1, alpha=0.7, vmin=vmin, vmax=vmax)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        return sc
    
    # Common color scale
    vmin = np.percentile(all_H, 5)
    vmax = np.percentile(all_H, 95)
    
    sc = scatter_curvature(axes[0], cloud_a_sub, H_a, 
                          'Cloud A: Mean Curvature\n(Texture 1, λ=50μm)', vmin, vmax)
    scatter_curvature(axes[1], cloud_b, H_b,
                     'Cloud B: Mean Curvature\n(Texture 1, λ=50μm)', vmin, vmax)
    scatter_curvature(axes[2], cloud_c, H_c,
                     'Cloud C: Mean Curvature\n(Texture 2, λ=20μm)', vmin, vmax)
    
    # Add colorbar
    fig5.colorbar(sc, ax=axes, label='Mean Curvature H (mm⁻¹)', shrink=0.8)
    
    plt.tight_layout()
    plt.savefig('synthetic_surfaces/curvature_maps.png', dpi=150, bbox_inches='tight')
    print("Saved: curvature_maps.png")
    plt.close()
    
    print("\nAll visualizations complete!")


if __name__ == "__main__":
    create_comparison_visualization()
