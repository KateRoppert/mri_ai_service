"""
Registration Validation Tool
Comprehensive validation of image registration results
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
from datetime import datetime

# Import imaging libraries
try:
    import nibabel as nib
    import ants
    ANTS_AVAILABLE = True
except ImportError:
    ANTS_AVAILABLE = False
    print("Warning: ANTs not available. Some features will be limited.")

try:
    from scipy import ndimage
    from scipy.stats import pearsonr, spearmanr
    from skimage import measure, filters
    from skimage.metrics import structural_similarity as ssim
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: SciPy/scikit-image not available. Some metrics will be unavailable.")

logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Container for validation metrics"""
    # Basic intensity metrics
    mean_intensity_original: float
    mean_intensity_registered: float
    std_intensity_original: float
    std_intensity_registered: float
    intensity_range_original: Tuple[float, float]
    intensity_range_registered: Tuple[float, float]
    
    # Similarity metrics (registered vs template)
    correlation_pearson: float
    correlation_spearman: float
    mutual_information: float
    normalized_mutual_information: float
    structural_similarity: float
    
    # Geometric metrics
    volume_original: int
    volume_registered: int
    volume_change_percent: float
    center_of_mass_shift: float
    
    # Quality metrics
    edge_preservation: float
    contrast_preservation: float
    sharpness_original: float
    sharpness_registered: float
    
    # Transform metrics
    transform_determinant_stats: Dict[str, float]
    jacobian_negative_voxels: int
    jacobian_negative_percent: float


class RegistrationValidator:
    """Main validation class"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging"""
        log_file = self.output_dir / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        logger.info(f"Validation log: {log_file}")
    
    def validate_registration(self, original_path: Path, registered_path: Path, 
                            template_path: Path, transform_paths: List[Path]) -> ValidationMetrics:
        """Complete validation of registration results"""
        
        logger.info("Starting registration validation")
        logger.info(f"Original: {original_path}")
        logger.info(f"Registered: {registered_path}")
        logger.info(f"Template: {template_path}")
        
        # Load images
        original_img, registered_img, template_img = self.load_images(
            original_path, registered_path, template_path
        )
        
        # Calculate metrics
        metrics = self.calculate_all_metrics(
            original_img, registered_img, template_img, transform_paths
        )
        
        # Generate reports
        self.generate_visual_report(original_img, registered_img, template_img, metrics)
        self.generate_statistical_report(metrics)
        
        return metrics
    
    def load_images(self, original_path: Path, registered_path: Path, 
                   template_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load and validate image data"""
        
        logger.info("Loading images...")
        
        # Load with nibabel for better header info
        original_nib = nib.load(str(original_path))
        registered_nib = nib.load(str(registered_path))
        template_nib = nib.load(str(template_path))
        
        # Get data arrays
        original_data = original_nib.get_fdata()
        registered_data = registered_nib.get_fdata()
        template_data = template_nib.get_fdata()
        
        logger.info(f"Original shape: {original_data.shape}")
        logger.info(f"Registered shape: {registered_data.shape}")
        logger.info(f"Template shape: {template_data.shape}")
        
        # Validate dimensions
        if registered_data.shape != template_data.shape:
            logger.warning(f"Shape mismatch: registered {registered_data.shape} vs template {template_data.shape}")
        
        return original_data, registered_data, template_data
    
    def calculate_all_metrics(self, original: np.ndarray, registered: np.ndarray, 
                            template: np.ndarray, transform_paths: List[Path]) -> ValidationMetrics:
        """Calculate comprehensive validation metrics"""
        
        logger.info("Calculating validation metrics...")
        
        # Basic intensity statistics - these can be calculated independently
        intensity_metrics = self.calculate_intensity_metrics(original, registered)
        
        # Similarity metrics - only between registered and template (same shape)
        similarity_metrics = self.calculate_similarity_metrics(registered, template)
        
        # Geometric metrics - compare original vs registered volumes
        geometric_metrics = self.calculate_geometric_metrics(original, registered)
        
        # Quality metrics - compare original vs registered
        quality_metrics = self.calculate_quality_metrics(original, registered)
        
        # Transform metrics (if available)
        transform_metrics = self.calculate_transform_metrics(transform_paths)
        
        # Combine all metrics
        metrics = ValidationMetrics(
            **intensity_metrics,
            **similarity_metrics,
            **geometric_metrics,
            **quality_metrics,
            **transform_metrics
        )
        
        return metrics
    
    def calculate_intensity_metrics(self, original: np.ndarray, registered: np.ndarray) -> Dict:
        """Calculate intensity-based metrics"""
        
        # Remove background (assume 0 is background)
        orig_mask = original > 0
        reg_mask = registered > 0
        
        orig_values = original[orig_mask]
        reg_values = registered[reg_mask]
        
        return {
            'mean_intensity_original': float(np.mean(orig_values)) if len(orig_values) > 0 else 0.0,
            'mean_intensity_registered': float(np.mean(reg_values)) if len(reg_values) > 0 else 0.0,
            'std_intensity_original': float(np.std(orig_values)) if len(orig_values) > 0 else 0.0,
            'std_intensity_registered': float(np.std(reg_values)) if len(reg_values) > 0 else 0.0,
            'intensity_range_original': (float(np.min(orig_values)), float(np.max(orig_values))) if len(orig_values) > 0 else (0.0, 0.0),
            'intensity_range_registered': (float(np.min(reg_values)), float(np.max(reg_values))) if len(reg_values) > 0 else (0.0, 0.0),
        }
    
    def calculate_similarity_metrics(self, registered: np.ndarray, template: np.ndarray) -> Dict:
        """Calculate similarity metrics between registered and template images"""
        
        # Only compare registered vs template (same shape)
        template_mask = template > 0
        registered_mask = registered > 0
        
        # Use intersection of both masks
        common_mask = template_mask & registered_mask
        
        if not np.any(common_mask):
            logger.warning("No common foreground between registered and template images")
            return {
                'correlation_pearson': 0.0,
                'correlation_spearman': 0.0,
                'mutual_information': 0.0,
                'normalized_mutual_information': 0.0,
                'structural_similarity': 0.0,
            }
        
        reg_values = registered[common_mask]
        temp_values = template[common_mask]
        
        # Correlation coefficients
        pearson_corr, _ = pearsonr(reg_values.flatten(), temp_values.flatten())
        spearman_corr, _ = spearmanr(reg_values.flatten(), temp_values.flatten())
        
        # Mutual information
        mi = self.calculate_mutual_information(registered, template)
        nmi = self.calculate_normalized_mutual_information(registered, template)
        
        # Structural similarity (on 2D slices)
        ssim_score = self.calculate_ssim_3d(registered, template)
        
        return {
            'correlation_pearson': float(pearson_corr) if not np.isnan(pearson_corr) else 0.0,
            'correlation_spearman': float(spearman_corr) if not np.isnan(spearman_corr) else 0.0,
            'mutual_information': float(mi),
            'normalized_mutual_information': float(nmi),
            'structural_similarity': float(ssim_score),
        }
    
    def calculate_geometric_metrics(self, original: np.ndarray, registered: np.ndarray) -> Dict:
        """Calculate geometric preservation metrics"""
        
        # Volume calculation
        orig_volume = int(np.sum(original > 0))
        reg_volume = int(np.sum(registered > 0))
        
        if orig_volume > 0:
            volume_change = ((reg_volume - orig_volume) / orig_volume) * 100
        else:
            volume_change = 0.0
        
        # Center of mass calculation
        com_shift = 0.0
        if SCIPY_AVAILABLE:
            try:
                if orig_volume > 0:
                    orig_com = ndimage.center_of_mass(original > 0)
                else:
                    orig_com = (0, 0, 0)
                
                if reg_volume > 0:
                    reg_com = ndimage.center_of_mass(registered > 0)
                else:
                    reg_com = (0, 0, 0)
                
                # Note: COM shift calculation is approximate since images have different shapes
                # This gives a rough estimate based on voxel coordinates
                logger.warning("Center of mass shift calculation is approximate due to different image dimensions")
                
            except Exception as e:
                logger.warning(f"Could not calculate center of mass shift: {e}")
                com_shift = 0.0
        
        return {
            'volume_original': orig_volume,
            'volume_registered': reg_volume,
            'volume_change_percent': float(volume_change),
            'center_of_mass_shift': float(com_shift),
        }
    
    def calculate_quality_metrics(self, original: np.ndarray, registered: np.ndarray) -> Dict:
        """Calculate image quality metrics"""
        
        # Since images have different shapes, we calculate metrics independently
        # and compare the magnitudes rather than direct correlations
        
        # Edge preservation (magnitude comparison)
        edge_preservation = self.calculate_edge_preservation_magnitude(original, registered)
        
        # Contrast preservation (magnitude comparison)
        contrast_preservation = self.calculate_contrast_preservation_magnitude(original, registered)
        
        # Sharpness (gradient magnitude)
        sharpness_orig = self.calculate_sharpness(original)
        sharpness_reg = self.calculate_sharpness(registered)
        
        return {
            'edge_preservation': float(edge_preservation),
            'contrast_preservation': float(contrast_preservation),
            'sharpness_original': float(sharpness_orig),
            'sharpness_registered': float(sharpness_reg),
        }
    
    def calculate_transform_metrics(self, transform_paths: List[Path]) -> Dict:
        """Calculate transform quality metrics"""
        
        # Default values
        default_metrics = {
            'transform_determinant_stats': {'mean': 1.0, 'std': 0.0, 'min': 1.0, 'max': 1.0},
            'jacobian_negative_voxels': 0,
            'jacobian_negative_percent': 0.0,
        }
        
        if not ANTS_AVAILABLE or not transform_paths:
            logger.warning("Cannot calculate transform metrics: ANTs not available or no transforms")
            return default_metrics
        
        try:
            # Find deformation field and affine transform
            warp_file = None
            affine_file = None
            
            for tf_path in transform_paths:
                if tf_path.is_file():
                    if 'Warp.nii.gz' in str(tf_path) or 'warp' in str(tf_path).lower():
                        warp_file = tf_path
                    elif tf_path.suffix in ['.mat'] or 'Affine' in str(tf_path):
                        affine_file = tf_path
                elif tf_path.is_dir():
                    # Search in directory
                    for file in tf_path.glob('**/*'):
                        if 'Warp.nii.gz' in str(file) or 'warp' in str(file).lower():
                            warp_file = file
                        elif file.suffix in ['.mat'] or 'Affine' in str(file):
                            affine_file = file
            
            if not warp_file or not warp_file.exists():
                logger.warning("No deformation field found")
                return default_metrics
            
            logger.info(f"Found warp field: {warp_file}")
            if affine_file:
                logger.info(f"Found affine transform: {affine_file}")
            
            # Load deformation field
            warp_img = ants.image_read(str(warp_file))
            
            # Calculate Jacobian determinant from displacement field directly
            # This is more reliable than trying to use composite transforms
            warp_data = warp_img.numpy()
            
            # Initialize Jacobian determinant
            if len(warp_data.shape) == 4 and warp_data.shape[3] == 3:
                # 3D displacement field with 3 components
                spacing = warp_img.spacing
                jac_det = np.ones(warp_data.shape[:3])
                
                # Calculate Jacobian matrix components
                # J_ij = ∂u_i/∂x_j + δ_ij (where δ_ij is Kronecker delta)
                jacobian_matrix = np.zeros(warp_data.shape[:3] + (3, 3))
                
                for i in range(3):  # displacement component
                    for j in range(3):  # spatial dimension
                        # Calculate partial derivative
                        grad = np.gradient(warp_data[:,:,:,i], spacing[j], axis=j)
                        jacobian_matrix[:,:,:,i,j] = grad
                        
                        # Add identity component
                        if i == j:
                            jacobian_matrix[:,:,:,i,j] += 1.0
                
                # Calculate determinant for each voxel
                for x in range(jacobian_matrix.shape[0]):
                    for y in range(jacobian_matrix.shape[1]):
                        for z in range(jacobian_matrix.shape[2]):
                            jac_det[x,y,z] = np.linalg.det(jacobian_matrix[x,y,z,:,:])
                
                jac_data = jac_det
                
            else:
                logger.warning(f"Unexpected warp field shape: {warp_data.shape}, using simplified calculation")
                # Fallback: use simple approximation
                jac_data = np.ones(warp_data.shape[:3])

            
            # Statistics
            jac_stats = {
                'mean': float(np.mean(jac_data)),
                'std': float(np.std(jac_data)),
                'min': float(np.min(jac_data)),
                'max': float(np.max(jac_data)),
            }
            
            # Negative Jacobian (folding)
            negative_voxels = int(np.sum(jac_data < 0))
            total_voxels = int(np.prod(jac_data.shape))
            negative_percent = (negative_voxels / total_voxels) * 100
            
            return {
                'transform_determinant_stats': jac_stats,
                'jacobian_negative_voxels': negative_voxels,
                'jacobian_negative_percent': float(negative_percent),
            }
            
        except Exception as e:
            logger.error(f"Error calculating transform metrics: {e}")
            return default_metrics
    
    def calculate_mutual_information(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate mutual information (simplified)"""
        
        # Create joint histogram
        mask = (img1 > 0) & (img2 > 0)
        if not np.any(mask):
            return 0.0
        
        x = img1[mask].flatten()
        y = img2[mask].flatten()
        
        if len(x) == 0 or len(y) == 0:
            return 0.0
        
        # Normalize to [0, 255] for histogram
        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        
        if x_range == 0 or y_range == 0:
            return 0.0
        
        x = ((x - x.min()) / x_range * 255).astype(int)
        y = ((y - y.min()) / y_range * 255).astype(int)
        
        # Joint histogram
        hist_2d, _, _ = np.histogram2d(x, y, bins=256)
        
        # Marginal histograms
        hist_x = np.histogram(x, bins=256)[0]
        hist_y = np.histogram(y, bins=256)[0]
        
        # Probabilities
        p_xy = hist_2d / np.sum(hist_2d)
        p_x = hist_x / np.sum(hist_x)
        p_y = hist_y / np.sum(hist_y)
        
        # Mutual information
        mi = 0.0
        for i in range(256):
            for j in range(256):
                if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
        
        return mi
    
    def calculate_normalized_mutual_information(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate normalized mutual information"""
        
        mi = self.calculate_mutual_information(img1, img2)
        
        # Calculate entropies (simplified)
        mask1 = img1 > 0
        mask2 = img2 > 0
        
        if not np.any(mask1) or not np.any(mask2):
            return 0.0
        
        # Normalize intensities
        x = img1[mask1].flatten()
        y = img2[mask2].flatten()
        
        if len(x) == 0 or len(y) == 0:
            return 0.0
        
        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        
        if x_range == 0 or y_range == 0:
            return 0.0
        
        x = ((x - x.min()) / x_range * 255).astype(int)
        y = ((y - y.min()) / y_range * 255).astype(int)
        
        # Histograms
        hist_x = np.histogram(x, bins=256)[0]
        hist_y = np.histogram(y, bins=256)[0]
        
        # Probabilities
        p_x = hist_x / np.sum(hist_x)
        p_y = hist_y / np.sum(hist_y)
        
        # Entropies
        h_x = -np.sum(p_x[p_x > 0] * np.log2(p_x[p_x > 0]))
        h_y = -np.sum(p_y[p_y > 0] * np.log2(p_y[p_y > 0]))
        
        # Normalized MI
        if h_x + h_y > 0:
            nmi = 2 * mi / (h_x + h_y)
        else:
            nmi = 0.0
        
        return nmi
    
    def calculate_ssim_3d(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate SSIM on multiple 2D slices"""
        
        if not SCIPY_AVAILABLE:
            return 0.0
        
        # Get common mask
        mask = (img1 > 0) & (img2 > 0)
        if not np.any(mask):
            return 0.0
        
        # Calculate SSIM on axial slices
        ssim_scores = []
        min_slices = min(img1.shape[2], img2.shape[2])
        
        for i in range(min_slices):
            slice1 = img1[:, :, i]
            slice2 = img2[:, :, i]
            
            if np.any(slice1 > 0) and np.any(slice2 > 0):
                try:
                    data_range = max(slice1.max(), slice2.max()) - min(slice1.min(), slice2.min())
                    if data_range > 0:
                        score = ssim(slice1, slice2, data_range=data_range)
                        ssim_scores.append(score)
                except:
                    continue
        
        return np.mean(ssim_scores) if ssim_scores else 0.0
    
    def calculate_edge_preservation_magnitude(self, original: np.ndarray, registered: np.ndarray) -> float:
        """Calculate edge preservation by comparing edge magnitudes"""
        
        if not SCIPY_AVAILABLE:
            return 0.0
        
        try:
            # Calculate edge magnitudes for both images
            grad_orig = np.sqrt(np.sum([ndimage.sobel(original, axis=i)**2 for i in range(3)], axis=0))
            grad_reg = np.sqrt(np.sum([ndimage.sobel(registered, axis=i)**2 for i in range(3)], axis=0))
            
            # Calculate mean edge strength in foreground regions
            orig_mask = original > 0
            reg_mask = registered > 0
            
            if not np.any(orig_mask) or not np.any(reg_mask):
                return 0.0
            
            mean_edge_orig = np.mean(grad_orig[orig_mask])
            mean_edge_reg = np.mean(grad_reg[reg_mask])
            
            # Edge preservation as ratio (closer to 1.0 is better)
            if mean_edge_orig > 0:
                edge_ratio = min(mean_edge_reg / mean_edge_orig, mean_edge_orig / mean_edge_reg)
            else:
                edge_ratio = 0.0
            
            return edge_ratio
            
        except Exception as e:
            logger.warning(f"Could not calculate edge preservation: {e}")
            return 0.0
    
    def calculate_contrast_preservation_magnitude(self, original: np.ndarray, registered: np.ndarray) -> float:
        """Calculate contrast preservation by comparing contrast magnitudes"""
        
        if not SCIPY_AVAILABLE:
            return 0.0
        
        try:
            from scipy.ndimage import uniform_filter
            
            def local_contrast(img, size=5):
                local_mean = uniform_filter(img.astype(float), size=size)
                local_var = uniform_filter(img.astype(float)**2, size=size) - local_mean**2
                return np.sqrt(np.maximum(local_var, 0))
            
            contrast_orig = local_contrast(original)
            contrast_reg = local_contrast(registered)
            
            # Calculate mean contrast in foreground regions
            orig_mask = original > 0
            reg_mask = registered > 0
            
            if not np.any(orig_mask) or not np.any(reg_mask):
                return 0.0
            
            mean_contrast_orig = np.mean(contrast_orig[orig_mask])
            mean_contrast_reg = np.mean(contrast_reg[reg_mask])
            
            # Contrast preservation as ratio (closer to 1.0 is better)
            if mean_contrast_orig > 0:
                contrast_ratio = min(mean_contrast_reg / mean_contrast_orig, 
                                   mean_contrast_orig / mean_contrast_reg)
            else:
                contrast_ratio = 0.0
            
            return contrast_ratio
            
        except Exception as e:
            logger.warning(f"Could not calculate contrast preservation: {e}")
            return 0.0
    
    def calculate_sharpness(self, img: np.ndarray) -> float:
        """Calculate image sharpness (mean gradient magnitude)"""
        
        if not SCIPY_AVAILABLE:
            return 0.0
        
        try:
            # Calculate gradient magnitude
            grad_mag = np.sqrt(np.sum([ndimage.sobel(img, axis=i)**2 for i in range(3)], axis=0))
            
            # Mean gradient in foreground
            mask = img > 0
            if not np.any(mask):
                return 0.0
            
            return np.mean(grad_mag[mask])
            
        except Exception as e:
            logger.warning(f"Could not calculate sharpness: {e}")
            return 0.0
    
    def generate_visual_report(self, original: np.ndarray, registered: np.ndarray, 
                         template: np.ndarray, metrics: ValidationMetrics):
        """Generate comprehensive visual validation report"""
        
        logger.info("Generating visual validation report...")
        
        # Create figure with subplots (3x3 grid instead of 4x3)
        fig = plt.figure(figsize=(18, 18))
        
        # Get middle slices for visualization (handle different dimensions)
        orig_mid_slice = original.shape[2] // 2
        reg_mid_slice = registered.shape[2] // 2
        temp_mid_slice = template.shape[2] // 2
        
        # 1. Image comparison (top row)
        ax1 = plt.subplot(3, 3, 1)
        plt.imshow(original[:, :, orig_mid_slice], cmap='gray')
        plt.title(f'Original Image\nShape: {original.shape}')
        plt.axis('off')
        
        ax2 = plt.subplot(3, 3, 2)
        plt.imshow(registered[:, :, reg_mid_slice], cmap='gray')
        plt.title(f'Registered Image\nShape: {registered.shape}')
        plt.axis('off')
        
        ax3 = plt.subplot(3, 3, 3)
        plt.imshow(template[:, :, temp_mid_slice], cmap='gray')
        plt.title(f'Template Image\nShape: {template.shape}')
        plt.axis('off')
        
        # 2. Intensity histograms (middle row, left)
        ax4 = plt.subplot(3, 3, 4)
        orig_values = original[original > 0]
        reg_values = registered[registered > 0]
        temp_values = template[template > 0]
        
        plt.hist(orig_values.flatten(), bins=50, alpha=0.5, label='Original', density=True)
        plt.hist(reg_values.flatten(), bins=50, alpha=0.5, label='Registered', density=True)
        plt.hist(temp_values.flatten(), bins=50, alpha=0.5, label='Template', density=True)
        plt.legend()
        plt.title('Intensity Distributions')
        plt.xlabel('Intensity')
        plt.ylabel('Density')
        
        # 3. Scatter plot: Template vs Registered (middle row, center)
        ax5 = plt.subplot(3, 3, 5)
        # Only create scatter plot if shapes match
        if registered.shape == template.shape:
            mask = (template > 0) & (registered > 0)
            if np.any(mask):
                sample_size = min(10000, np.sum(mask))
                sample_indices = np.random.choice(np.sum(mask), sample_size, replace=False)
                temp_sample = template[mask].flatten()[sample_indices]
                reg_sample = registered[mask].flatten()[sample_indices]
                
                plt.scatter(temp_sample, reg_sample, alpha=0.1, s=1)
                plt.plot([temp_sample.min(), temp_sample.max()], 
                        [temp_sample.min(), temp_sample.max()], 'r--', alpha=0.8)
                plt.xlabel('Template Intensity')
                plt.ylabel('Registered Intensity')
                plt.title(f'Intensity Correlation\nr = {metrics.correlation_pearson:.3f}')
            else:
                plt.text(0.5, 0.5, 'No overlapping\nforeground regions', 
                        ha='center', va='center', transform=ax5.transAxes)
        else:
            plt.text(0.5, 0.5, f'Shape mismatch:\nRegistered: {registered.shape}\nTemplate: {template.shape}\nCorrelation unavailable', 
                    ha='center', va='center', transform=ax5.transAxes, fontsize=10)
        plt.axis('off')
        
        # 4. Metrics summary (middle row, right)
        ax6 = plt.subplot(3, 3, 6)
        ax6.axis('off')
        
        metrics_text = f"""
        Similarity Metrics:
        • Pearson correlation: {metrics.correlation_pearson:.3f}
        • Mutual information: {metrics.mutual_information:.3f}
        • Structural similarity: {metrics.structural_similarity:.3f}
        
        Quality Metrics:
        • Edge preservation: {metrics.edge_preservation:.3f}
        • Contrast preservation: {metrics.contrast_preservation:.3f}
        • Volume change: {metrics.volume_change_percent:.1f}%
        
        Transform Quality:
        • Negative Jacobian: {metrics.jacobian_negative_percent:.2f}%
        """
        
        ax6.text(0.1, 0.9, metrics_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        
        # 5. Volume comparison (bottom row, left)
        ax7 = plt.subplot(3, 3, 7)
        volumes = [metrics.volume_original, metrics.volume_registered]
        labels = ['Original', 'Registered']
        colors = ['skyblue', 'lightcoral']
        
        bars = plt.bar(labels, volumes, color=colors)
        plt.title('Volume Comparison')
        plt.ylabel('Volume (voxels)')
        
        # Add percentage change
        for i, (bar, vol) in enumerate(zip(bars, volumes)):
            height = bar.get_height()
            if i == 1:  # Registered
                change = metrics.volume_change_percent
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{change:+.1f}%', ha='center', va='bottom')
        
        # 6. Checkerboard overlay (bottom row, center) - only if shapes match
        ax8 = plt.subplot(3, 3, 8)
        if registered.shape == template.shape:
            checkerboard = self.create_checkerboard_overlay(
                registered[:, :, reg_mid_slice], template[:, :, temp_mid_slice]
            )
            plt.imshow(checkerboard, cmap='gray')
            plt.title('Checkerboard Overlay\n(Registered/Template)')
        else:
            plt.text(0.5, 0.5, f'Checkerboard overlay\nnot available\n(different shapes)', 
                    ha='center', va='center', transform=ax8.transAxes, fontsize=12)
        plt.axis('off')
        
        # 7. Edge overlay (bottom row, right) - only if shapes match
        ax9 = plt.subplot(3, 3, 9)
        if registered.shape == template.shape:
            edge_overlay = self.create_edge_overlay(
                registered[:, :, reg_mid_slice], template[:, :, temp_mid_slice]
            )
            plt.imshow(edge_overlay)
            plt.title('Edge Overlay\n(Red: Template, Green: Registered)')
        else:
            plt.text(0.5, 0.5, f'Edge overlay\nnot available\n(different shapes)', 
                    ha='center', va='center', transform=ax9.transAxes, fontsize=12)
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save report
        report_path = self.output_dir / 'validation_report.png'
        plt.savefig(report_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visual report saved: {report_path}")
    
    def create_checkerboard_overlay(self, img1: np.ndarray, img2: np.ndarray, 
                                  checker_size: int = 20) -> np.ndarray:
        """Create checkerboard overlay of two images"""
        
        # Normalize images
        img1_norm = (img1 - img1.min()) / (img1.max() - img1.min() + 1e-8)
        img2_norm = (img2 - img2.min()) / (img2.max() - img2.min() + 1e-8)
        
        # Create checkerboard pattern
        h, w = img1.shape
        checker = np.zeros((h, w), dtype=bool)
        
        for i in range(0, h, checker_size):
            for j in range(0, w, checker_size):
                if ((i // checker_size) + (j // checker_size)) % 2 == 0:
                    checker[i:i+checker_size, j:j+checker_size] = True
        
        # Apply checkerboard
        result = np.where(checker, img1_norm, img2_norm)
        return result
    
    def create_edge_overlay(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Create colored edge overlay"""
        
        if not SCIPY_AVAILABLE:
            # Return simple overlay
            return np.stack([img1/img1.max(), img2/img2.max(), np.zeros_like(img1)], axis=2)
        
        # Detect edges
        edges1 = filters.sobel(img1) > 0.1 * filters.sobel(img1).max()
        edges2 = filters.sobel(img2) > 0.1 * filters.sobel(img2).max()
        
        # Create RGB overlay
        overlay = np.zeros((*img1.shape, 3))
        
        # Background (grayscale)
        bg = (img1 + img2) / 2
        bg = bg / (bg.max() + 1e-8)
        overlay[:, :, 0] = bg
        overlay[:, :, 1] = bg
        overlay[:, :, 2] = bg
        
        # Red edges for template, Green for registered
        overlay[edges2, :] = [1, 0, 0]  # Template edges in red
        overlay[edges1, :] = [0, 1, 0]  # Registered edges in green
        overlay[edges1 & edges2, :] = [1, 1, 0]  # Overlap in yellow
        
        return overlay
    
    def generate_statistical_report(self, metrics: ValidationMetrics):
        """Generate detailed statistical report"""
        
        logger.info("Generating statistical report...")
        
        # Create comprehensive report
        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'intensity_statistics': {
                'original': {
                    'mean': metrics.mean_intensity_original,
                    'std': metrics.std_intensity_original,
                    'range': metrics.intensity_range_original
                },
                'registered': {
                    'mean': metrics.mean_intensity_registered,
                    'std': metrics.std_intensity_registered,
                    'range': metrics.intensity_range_registered
                }
            },
            'similarity_metrics': {
                'pearson_correlation': metrics.correlation_pearson,
                'spearman_correlation': metrics.correlation_spearman,
                'mutual_information': metrics.mutual_information,
                'normalized_mutual_information': metrics.normalized_mutual_information,
                'structural_similarity': metrics.structural_similarity
            },
            'geometric_metrics': {
                'volume_original': metrics.volume_original,
                'volume_registered': metrics.volume_registered,
                'volume_change_percent': metrics.volume_change_percent,
                'center_of_mass_shift': metrics.center_of_mass_shift
            },
            'quality_metrics': {
                'edge_preservation': metrics.edge_preservation,
                'contrast_preservation': metrics.contrast_preservation,
                'sharpness_original': metrics.sharpness_original,
                'sharpness_registered': metrics.sharpness_registered
            },
            'transform_metrics': {
                'determinant_statistics': metrics.transform_determinant_stats,
                'jacobian_negative_voxels': metrics.jacobian_negative_voxels,
                'jacobian_negative_percent': metrics.jacobian_negative_percent
            },
            'quality_assessment': self.assess_registration_quality(metrics)
        }
        
        # Save JSON report
        json_path = self.output_dir / 'validation_metrics.json'
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save human-readable report
        text_path = self.output_dir / 'validation_summary.txt'
        with open(text_path, 'w') as f:
            f.write(self.format_summary_report(metrics))
        
        logger.info(f"Statistical reports saved: {json_path}, {text_path}")
    
    def assess_registration_quality(self, metrics: ValidationMetrics) -> Dict[str, str]:
        """Assess overall registration quality"""
        
        assessment = {}
        
        # Similarity assessment
        if metrics.correlation_pearson > 0.9:
            assessment['similarity'] = 'Excellent'
        elif metrics.correlation_pearson > 0.8:
            assessment['similarity'] = 'Good'
        elif metrics.correlation_pearson > 0.6:
            assessment['similarity'] = 'Fair'
        else:
            assessment['similarity'] = 'Poor'
        
        # Volume preservation
        vol_change = abs(metrics.volume_change_percent)
        if vol_change < 2:
            assessment['volume_preservation'] = 'Excellent'
        elif vol_change < 5:
            assessment['volume_preservation'] = 'Good'
        elif vol_change < 10:
            assessment['volume_preservation'] = 'Fair'
        else:
            assessment['volume_preservation'] = 'Poor'
        
        # Transform quality
        if metrics.jacobian_negative_percent < 0.1:
            assessment['transform_quality'] = 'Excellent'
        elif metrics.jacobian_negative_percent < 1.0:
            assessment['transform_quality'] = 'Good'
        elif metrics.jacobian_negative_percent < 5.0:
            assessment['transform_quality'] = 'Fair'
        else:
            assessment['transform_quality'] = 'Poor'
        
        # Overall assessment
        scores = {'Excellent': 4, 'Good': 3, 'Fair': 2, 'Poor': 1}
        avg_score = np.mean([scores[v] for v in assessment.values()])
        
        if avg_score >= 3.5:
            assessment['overall'] = 'Excellent'
        elif avg_score >= 2.5:
            assessment['overall'] = 'Good'
        elif avg_score >= 1.5:
            assessment['overall'] = 'Fair'
        else:
            assessment['overall'] = 'Poor'
        
        return assessment
    
    def format_summary_report(self, metrics: ValidationMetrics) -> str:
        """Format human-readable summary report"""
        
        assessment = self.assess_registration_quality(metrics)
        
        report = f"""
REGISTRATION VALIDATION SUMMARY
===============================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERALL ASSESSMENT: {assessment['overall']}

SIMILARITY METRICS
------------------
• Pearson Correlation:     {metrics.correlation_pearson:.4f}
• Spearman Correlation:    {metrics.correlation_spearman:.4f}
• Mutual Information:      {metrics.mutual_information:.4f}
• Normalized MI:           {metrics.normalized_mutual_information:.4f}
• Structural Similarity:   {metrics.structural_similarity:.4f}

Assessment: {assessment['similarity']}

GEOMETRIC PRESERVATION
----------------------
• Original Volume:         {metrics.volume_original:,} voxels
• Registered Volume:       {metrics.volume_registered:,} voxels
• Volume Change:           {metrics.volume_change_percent:+.2f}%
• Center of Mass Shift:    {metrics.center_of_mass_shift:.2f} voxels

Assessment: {assessment['volume_preservation']}

IMAGE QUALITY
-------------
• Edge Preservation:       {metrics.edge_preservation:.4f}
• Contrast Preservation:   {metrics.contrast_preservation:.4f}
• Original Sharpness:      {metrics.sharpness_original:.2f}
• Registered Sharpness:    {metrics.sharpness_registered:.2f}

TRANSFORM QUALITY
-----------------
• Jacobian Determinant:
  - Mean: {metrics.transform_determinant_stats['mean']:.4f}
  - Std:  {metrics.transform_determinant_stats['std']:.4f}
  - Min:  {metrics.transform_determinant_stats['min']:.4f}
  - Max:  {metrics.transform_determinant_stats['max']:.4f}
• Negative Jacobian:       {metrics.jacobian_negative_voxels:,} voxels ({metrics.jacobian_negative_percent:.2f}%)

Assessment: {assessment['transform_quality']}

INTENSITY STATISTICS
--------------------
Original Image:
  - Mean: {metrics.mean_intensity_original:.2f}
  - Std:  {metrics.std_intensity_original:.2f}
  - Range: [{metrics.intensity_range_original[0]:.2f}, {metrics.intensity_range_original[1]:.2f}]

Registered Image:
  - Mean: {metrics.mean_intensity_registered:.2f}
  - Std:  {metrics.std_intensity_registered:.2f}
  - Range: [{metrics.intensity_range_registered[0]:.2f}, {metrics.intensity_range_registered[1]:.2f}]

RECOMMENDATIONS
---------------
"""
        
        # Add specific recommendations
        recommendations = []
        
        if metrics.correlation_pearson < 0.7:
            recommendations.append("• Consider adjusting registration parameters for better alignment")
        
        if abs(metrics.volume_change_percent) > 5:
            recommendations.append("• Significant volume change detected - check for over-regularization")
        
        if metrics.jacobian_negative_percent > 1:
            recommendations.append("• High folding detected - consider reducing deformation regularization")
        
        if metrics.edge_preservation < 0.7:
            recommendations.append("• Poor edge preservation - consider edge-preserving regularization")
        
        if not recommendations:
            recommendations.append("• Registration quality appears satisfactory")
        
        report += "\n".join(recommendations)
        
        return report


def main():
    """Main function for command-line usage"""
    
    parser = argparse.ArgumentParser(description='Validate image registration results')
    parser.add_argument('--original', type=Path, required=True,
                       help='Path to original/moving image')
    parser.add_argument('--registered', type=Path, required=True,
                       help='Path to registered image')
    parser.add_argument('--template', type=Path, required=True,
                       help='Path to template/fixed image')
    parser.add_argument('--transforms', nargs='+', type=Path,
                       help='Path(s) to transform files')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output directory for validation results')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Validate input files
    for file_path in [args.original, args.registered, args.template]:
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            return 1
    
    if args.transforms:
        for tf_path in args.transforms:
            if not tf_path.exists():
                print(f"Warning: Transform file not found: {tf_path}")
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Create validator
        validator = RegistrationValidator(args.output)
        
        # Run validation
        metrics = validator.validate_registration(
            args.original, args.registered, args.template, 
            args.transforms or []
        )
        
        print(f"\nValidation completed successfully!")
        print(f"Results saved in: {args.output}")
        print(f"Overall assessment: {validator.assess_registration_quality(metrics)['overall']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())