#!/usr/bin/env python3
"""
Bias Field Correction Quality Assessment Script - FIXED VERSION

This script evaluates the quality of bias field correction by comparing
raw and preprocessed brain MRI images using various metrics and visualizations.
FIXED: Proper aspect ratio handling for brain image display.

Usage:
    python bias_correction_evaluator.py --raw /path/to/raw_image.nii.gz --preprocessed /path/to/preprocessed_image.nii.gz

Requirements:
    - nibabel
    - numpy
    - matplotlib
    - scipy
    - sklearn
    - seaborn
"""

import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

#!/usr/bin/env python3
"""
Improved brain masking methods for bias field correction evaluation.
These methods can replace or enhance the create_brain_mask function.
"""

import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class ImprovedBrainMasking:
    """Enhanced brain masking techniques for better bias field evaluation."""
    
    def __init__(self, raw_data, preprocessed_data):
        self.raw_data = raw_data
        self.preprocessed_data = preprocessed_data
        self.brain_mask = None
    
    def method_1_adaptive_threshold(self, percentile_low=10, percentile_high=85):
        """
        Adaptive thresholding with better morphological operations.
        More robust than single percentile threshold.
        """
        print("Creating brain mask using adaptive thresholding...")
        
        # Use preprocessed image as it typically has better contrast
        img = self.preprocessed_data.copy()
        
        # Remove very low values (likely background/noise)
        img[img < np.percentile(img[img > 0], 5)] = 0
        
        # Calculate adaptive threshold
        non_zero_vals = img[img > 0]
        if len(non_zero_vals) == 0:
            raise ValueError("No non-zero values in image")
        
        low_thresh = np.percentile(non_zero_vals, percentile_low)
        high_thresh = np.percentile(non_zero_vals, percentile_high)
        
        # Create initial mask
        mask = (img > low_thresh) & (img < high_thresh * 3)  # Allow for some high intensity
        
        # Morphological operations for cleanup
        # Remove small objects
        mask = ndimage.binary_opening(mask, structure=np.ones((3,3,3)), iterations=2)
        
        # Fill holes
        mask = ndimage.binary_fill_holes(mask)
        
        # Remove small islands and close gaps
        mask = ndimage.binary_closing(mask, structure=np.ones((5,5,5)), iterations=2)
        
        # Keep only the largest connected component (main brain)
        labeled, num_labels = ndimage.label(mask)
        if num_labels > 1:
            sizes = ndimage.sum(mask, labeled, range(1, num_labels + 1))
            max_label = np.argmax(sizes) + 1
            mask = labeled == max_label
        
        # Final cleanup
        mask = ndimage.binary_fill_holes(mask)
        
        self.brain_mask = mask
        print(f"Adaptive threshold mask created. Brain voxels: {np.sum(mask)}")
        return mask
    
    def method_2_otsu_threshold(self):
        """
        Otsu's method for automatic threshold selection.
        Good for images with bimodal intensity distribution.
        """
        print("Creating brain mask using Otsu thresholding...")
        
        img = self.preprocessed_data.copy()
        
        # Remove zeros and very low values
        non_zero_img = img[img > 0]
        if len(non_zero_img) == 0:
            raise ValueError("No non-zero values in image")
        
        # Otsu's method implementation
        def otsu_threshold(image_values):
            hist, bin_edges = np.histogram(image_values, bins=256)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Normalize histogram
            hist = hist.astype(float) / np.sum(hist)
            
            # Calculate cumulative sums
            cumsum_hist = np.cumsum(hist)
            cumsum_mean = np.cumsum(hist * bin_centers)
            
            # Global mean
            global_mean = cumsum_mean[-1]
            
            # Between-class variance
            between_class_var = np.zeros_like(cumsum_hist)
            
            for i in range(len(cumsum_hist)):
                if cumsum_hist[i] > 0 and cumsum_hist[i] < 1:
                    w1 = cumsum_hist[i]
                    w2 = 1 - w1
                    
                    if w1 > 0 and w2 > 0:
                        mu1 = cumsum_mean[i] / w1
                        mu2 = (global_mean - cumsum_mean[i]) / w2
                        between_class_var[i] = w1 * w2 * (mu1 - mu2) ** 2
            
            # Find optimal threshold
            optimal_idx = np.argmax(between_class_var)
            optimal_threshold = bin_centers[optimal_idx]
            
            return optimal_threshold
        
        threshold = otsu_threshold(non_zero_img)
        print(f"Otsu threshold: {threshold:.2f}")
        
        # Create mask
        mask = img > threshold
        
        # Morphological cleanup
        mask = self._morphological_cleanup(mask)
        
        self.brain_mask = mask
        print(f"Otsu mask created. Brain voxels: {np.sum(mask)}")
        return mask
    
    def method_3_kmeans_clustering(self, n_clusters=3):
        """
        K-means clustering to separate brain tissue from background.
        Useful when simple thresholding fails.
        """
        print(f"Creating brain mask using K-means clustering (k={n_clusters})...")
        
        img = self.preprocessed_data.copy()
        
        # Get non-zero voxels for clustering
        non_zero_mask = img > 0
        non_zero_vals = img[non_zero_mask].reshape(-1, 1)
        
        if len(non_zero_vals) < n_clusters * 10:
            print("Warning: Too few non-zero voxels for reliable clustering")
            return self.method_1_adaptive_threshold()
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(non_zero_vals)
        
        # Find cluster centers
        centers = kmeans.cluster_centers_.flatten()
        sorted_centers = np.sort(centers)
        
        print(f"Cluster centers: {sorted_centers}")
        
        # Assume brain tissue is in the middle/higher intensity clusters
        # Exclude the lowest intensity cluster (likely background/CSF)
        brain_clusters = np.arange(1, n_clusters)  # Exclude cluster 0 (lowest intensity)
        
        # Create mask from selected clusters
        mask = np.zeros_like(img, dtype=bool)
        
        # Map cluster labels back to image space
        cluster_image = np.zeros_like(img)
        cluster_image[non_zero_mask] = labels
        
        for cluster_id in brain_clusters:
            cluster_center = centers[cluster_id]
            # Only include clusters above a reasonable threshold
            if cluster_center > np.percentile(non_zero_vals, 20):
                mask |= (cluster_image == cluster_id)
        
        # Morphological cleanup
        mask = self._morphological_cleanup(mask)
        
        self.brain_mask = mask
        print(f"K-means mask created. Brain voxels: {np.sum(mask)}")
        return mask
    
    def method_4_multi_threshold(self, low_percentile=15, high_percentile=90):
        """
        Multi-step thresholding with edge detection.
        Combines intensity and gradient information.
        """
        print("Creating brain mask using multi-threshold approach...")
        
        img = self.preprocessed_data.copy()
        
        # Step 1: Initial intensity threshold
        non_zero_vals = img[img > 0]
        low_thresh = np.percentile(non_zero_vals, low_percentile)
        high_thresh = np.percentile(non_zero_vals, high_percentile)
        
        intensity_mask = (img > low_thresh) & (img < high_thresh * 2)
        
        # Step 2: Add gradient information to detect brain boundaries
        gradient = np.gradient(img)
        gradient_magnitude = np.sqrt(sum(g**2 for g in gradient))
        
        # Normalize gradient
        gradient_magnitude = gradient_magnitude / np.max(gradient_magnitude)
        
        # Moderate gradient indicates tissue boundaries
        gradient_mask = (gradient_magnitude > 0.05) & (gradient_magnitude < 0.5)
        
        # Step 3: Combine intensity and gradient information
        combined_mask = intensity_mask | gradient_mask
        
        # Step 4: Morphological cleanup
        combined_mask = self._morphological_cleanup(combined_mask, aggressive=True)
        
        self.brain_mask = combined_mask
        print(f"Multi-threshold mask created. Brain voxels: {np.sum(combined_mask)}")
        return combined_mask
    
    def method_5_robust_combination(self):
        """
        Robust method combining multiple approaches.
        Use this when other methods fail or for best results.
        """
        print("Creating brain mask using robust combination method...")
        
        try:
            # Try different methods
            mask1 = self.method_1_adaptive_threshold(percentile_low=8, percentile_high=85)
            mask2 = self.method_2_otsu_threshold()
            mask3 = self.method_3_kmeans_clustering(n_clusters=4)
            
            # Combine using voting
            combined = (mask1.astype(int) + mask2.astype(int) + mask3.astype(int))
            
            # Keep voxels that are positive in at least 2 out of 3 methods
            consensus_mask = combined >= 2
            
            # Final cleanup
            consensus_mask = self._morphological_cleanup(consensus_mask, aggressive=True)
            
            self.brain_mask = consensus_mask
            print(f"Robust combination mask created. Brain voxels: {np.sum(consensus_mask)}")
            return consensus_mask
            
        except Exception as e:
            print(f"Robust combination failed: {e}")
            print("Falling back to adaptive thresholding...")
            return self.method_1_adaptive_threshold()
    
    def _morphological_cleanup(self, mask, aggressive=False):
        """
        Standardized morphological cleanup operations.
        """
        # Define structure elements
        if aggressive:
            small_struct = np.ones((3,3,3))
            medium_struct = np.ones((5,5,5))
            large_struct = np.ones((7,7,7))
        else:
            small_struct = np.ones((2,2,2))
            medium_struct = np.ones((3,3,3))
            large_struct = np.ones((5,5,5))
        
        # Remove small noise
        mask = ndimage.binary_opening(mask, structure=small_struct, iterations=1)
        
        # Fill small holes
        mask = ndimage.binary_closing(mask, structure=medium_struct, iterations=1)
        
        # Fill larger holes
        mask = ndimage.binary_fill_holes(mask)
        
        # Keep only the largest connected component
        labeled, num_labels = ndimage.label(mask)
        if num_labels > 1:
            sizes = ndimage.sum(mask, labeled, range(1, num_labels + 1))
            max_label = np.argmax(sizes) + 1
            mask = labeled == max_label
        
        # Final hole filling
        mask = ndimage.binary_fill_holes(mask)
        
        # Smooth boundaries if aggressive cleanup
        if aggressive:
            mask = ndimage.binary_closing(mask, structure=large_struct, iterations=1)
            mask = ndimage.binary_opening(mask, structure=medium_struct, iterations=1)
        
        return mask
    
    def visualize_mask_comparison(self, masks_dict, slice_idx=None, orientation='axial'):
        """
        Compare different masking methods visually.
        """
        import matplotlib.pyplot as plt
        
        if slice_idx is None:
            slice_idx = self.preprocessed_data.shape[2] // 2
        
        n_methods = len(masks_dict)
        fig, axes = plt.subplots(2, n_methods, figsize=(4*n_methods, 8))
        
        if n_methods == 1:
            axes = axes.reshape(2, 1)
        
        # Show original image on top row
        if orientation == 'axial':
            img_slice = self.preprocessed_data[:, :, slice_idx]
        elif orientation == 'coronal':
            img_slice = self.preprocessed_data[:, slice_idx, :]
        else:  # sagittal
            img_slice = self.preprocessed_data[slice_idx, :, :]
        
        for i, (method_name, mask) in enumerate(masks_dict.items()):
            # Original image
            axes[0, i].imshow(img_slice, cmap='gray')
            axes[0, i].set_title(f'Original Image\n{method_name}')
            axes[0, i].axis('off')
            
            # Mask overlay
            if orientation == 'axial':
                mask_slice = mask[:, :, slice_idx]
            elif orientation == 'coronal':
                mask_slice = mask[:, slice_idx, :]
            else:  # sagittal
                mask_slice = mask[slice_idx, :, :]
            
            axes[1, i].imshow(img_slice, cmap='gray', alpha=0.7)
            axes[1, i].imshow(mask_slice, cmap='Reds', alpha=0.5)
            axes[1, i].set_title(f'Mask Overlay\n{np.sum(mask)} voxels')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()


class BiasFieldEvaluator:
    def __init__(self, raw_path, preprocessed_path):
        """Initialize with paths to raw and preprocessed images."""
        self.raw_path = raw_path
        self.preprocessed_path = preprocessed_path
        self.raw_img = None
        self.preprocessed_img = None
        self.raw_data = None
        self.preprocessed_data = None
        self.brain_mask = None
        self.voxel_spacing = None
        
    def load_images(self):
        """Load and validate MRI images."""
        try:
            print("Loading images...")
            self.raw_img = nib.load(self.raw_path)
            self.preprocessed_img = nib.load(self.preprocessed_path)
            
            self.raw_data = self.raw_img.get_fdata()
            self.preprocessed_data = self.preprocessed_img.get_fdata()
            
            # Get voxel spacing for proper aspect ratio
            self.voxel_spacing = self.raw_img.header.get_zooms()
            print(f"Voxel spacing: {self.voxel_spacing}")
            
            # Ensure same dimensions
            if self.raw_data.shape != self.preprocessed_data.shape:
                raise ValueError(f"Image dimensions don't match: {self.raw_data.shape} vs {self.preprocessed_data.shape}")
            
            print(f"Images loaded successfully. Shape: {self.raw_data.shape}")
            
        except Exception as e:
            print(f"Error loading images: {e}")
            raise
    
    def get_proper_aspect_ratio(self, slice_axis='axial'):
        """Calculate proper aspect ratio based on voxel spacing."""
        if self.voxel_spacing is None:
            return 1.0  # Default to square pixels if spacing unknown
        
        if slice_axis == 'axial':  # XY plane (most common)
            return self.voxel_spacing[1] / self.voxel_spacing[0]  # Y/X spacing ratio
        elif slice_axis == 'sagittal':  # YZ plane
            return self.voxel_spacing[2] / self.voxel_spacing[1]  # Z/Y spacing ratio
        elif slice_axis == 'coronal':  # XZ plane
            return self.voxel_spacing[2] / self.voxel_spacing[0]  # Z/X spacing ratio
        else:
            return 1.0
    
    def choose_best_slice(self):
        """Choose the best slice for visualization based on brain content."""
        # Try different slice orientations and positions to find the most informative one
        
        # For axial slices (Z dimension)
        axial_scores = []
        for z in range(max(1, self.raw_data.shape[2]//4), min(self.raw_data.shape[2], 3*self.raw_data.shape[2]//4)):
            slice_data = self.raw_data[:, :, z]
            # Score based on intensity variation and non-zero content
            score = np.std(slice_data) * np.sum(slice_data > 0)
            axial_scores.append((score, z, 'axial'))
        
        # For coronal slices (Y dimension) 
        coronal_scores = []
        for y in range(max(1, self.raw_data.shape[1]//4), min(self.raw_data.shape[1], 3*self.raw_data.shape[1]//4)):
            slice_data = self.raw_data[:, y, :]
            score = np.std(slice_data) * np.sum(slice_data > 0)
            coronal_scores.append((score, y, 'coronal'))
        
        # For sagittal slices (X dimension)
        sagittal_scores = []
        for x in range(max(1, self.raw_data.shape[0]//4), min(self.raw_data.shape[0], 3*self.raw_data.shape[0]//4)):
            slice_data = self.raw_data[x, :, :]
            score = np.std(slice_data) * np.sum(slice_data > 0)
            sagittal_scores.append((score, x, 'sagittal'))
        
        # Find the best slice across all orientations
        all_scores = axial_scores + coronal_scores + sagittal_scores
        best_score, best_index, best_orientation = max(all_scores, key=lambda x: x[0])
        
        print(f"Best slice: {best_orientation} slice {best_index} (score: {best_score:.0f})")
        return best_index, best_orientation
    
    def get_slice_data(self, slice_index, orientation):
        """Extract slice data based on orientation."""
        if orientation == 'axial':
            raw_slice = self.raw_data[:, :, slice_index]
            prep_slice = self.preprocessed_data[:, :, slice_index]
            if self.brain_mask is not None:
                mask_slice = self.brain_mask[:, :, slice_index]
            else:
                mask_slice = None
        elif orientation == 'coronal':
            raw_slice = self.raw_data[:, slice_index, :]
            prep_slice = self.preprocessed_data[:, slice_index, :]
            if self.brain_mask is not None:
                mask_slice = self.brain_mask[:, slice_index, :]
            else:
                mask_slice = None
        elif orientation == 'sagittal':
            raw_slice = self.raw_data[slice_index, :, :]
            prep_slice = self.preprocessed_data[slice_index, :, :]
            if self.brain_mask is not None:
                mask_slice = self.brain_mask[slice_index, :, :]
            else:
                mask_slice = None
        else:
            raise ValueError(f"Unknown orientation: {orientation}")
        
        return raw_slice, prep_slice, mask_slice
    
    def create_improved_brain_mask(evaluator_instance, method='robust'):
        """
        Replace the original create_brain_mask method with improved versions.
        
        Parameters:
        - evaluator_instance: Instance of BiasFieldEvaluator
        - method: 'adaptive', 'otsu', 'kmeans', 'multi', or 'robust'
        """
        
        masker = ImprovedBrainMasking(evaluator_instance.raw_data, 
                                    evaluator_instance.preprocessed_data)
        
        if method == 'adaptive':
            mask = masker.method_1_adaptive_threshold()
        elif method == 'otsu':
            mask = masker.method_2_otsu_threshold()
        elif method == 'kmeans':
            mask = masker.method_3_kmeans_clustering()
        elif method == 'multi':
            mask = masker.method_4_multi_threshold()
        elif method == 'robust':
            mask = masker.method_5_robust_combination()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        evaluator_instance.brain_mask = mask
        return mask
    
    def compute_intensity_metrics(self):
        """Compute intensity-based metrics for bias field assessment."""
        raw_brain = self.raw_data[self.brain_mask]
        prep_brain = self.preprocessed_data[self.brain_mask]
        
        # Remove zero values
        raw_brain = raw_brain[raw_brain > 0]
        prep_brain = prep_brain[prep_brain > 0]
        
        metrics = {}
        
        # Coefficient of Variation (CV) - lower is better after bias correction
        metrics['cv_raw'] = np.std(raw_brain) / np.mean(raw_brain)
        metrics['cv_preprocessed'] = np.std(prep_brain) / np.mean(prep_brain)
        metrics['cv_ratio'] = metrics['cv_preprocessed'] / metrics['cv_raw']
        
        # Signal-to-Noise Ratio estimation (using background noise approximation)
        def calculate_snr(data):
            if len(data) == 0:
                return 0.0
            
            # Method 1: Use low intensity values as noise proxy
            noise_threshold = np.percentile(data, 10)
            noise_values = data[data < noise_threshold]
            
            if len(noise_values) < 10:  # Not enough noise samples
                # Method 2: Use overall standard deviation
                noise_std = np.std(data)
            else:
                noise_std = np.std(noise_values)
            
            if noise_std == 0 or np.isnan(noise_std) or np.isinf(noise_std):
                return 0.0
            
            signal_mean = np.mean(data)
            if signal_mean == 0 or np.isnan(signal_mean) or np.isinf(signal_mean):
                return 0.0
                
            snr = signal_mean / noise_std
            return snr if not (np.isnan(snr) or np.isinf(snr)) else 0.0
        
        metrics['snr_raw'] = calculate_snr(raw_brain)
        metrics['snr_preprocessed'] = calculate_snr(prep_brain)
        
        # Intensity range
        metrics['range_raw'] = np.percentile(raw_brain, 95) - np.percentile(raw_brain, 5)
        metrics['range_preprocessed'] = np.percentile(prep_brain, 95) - np.percentile(prep_brain, 5)
        
        # Entropy (measure of information content)
        hist_raw, _ = np.histogram(raw_brain, bins=100, density=True)
        hist_prep, _ = np.histogram(prep_brain, bins=100, density=True)
        hist_raw = hist_raw[hist_raw > 0]
        hist_prep = hist_prep[hist_prep > 0]
        
        metrics['entropy_raw'] = -np.sum(hist_raw * np.log2(hist_raw))
        metrics['entropy_preprocessed'] = -np.sum(hist_prep * np.log2(hist_prep))
        
        return metrics
    
    def compute_spatial_metrics(self):
        """Compute spatial uniformity metrics."""
        metrics = {}
        
        # Get best slice for 2D analysis
        slice_idx, slice_orientation = self.choose_best_slice()
        raw_slice, prep_slice, mask_slice = self.get_slice_data(slice_idx, slice_orientation)
        
        # Spatial coefficient of variation across regions
        def regional_cv(image, mask, n_regions=4):
            h, w = image.shape
            region_cvs = []
            
            for i in range(n_regions):
                for j in range(n_regions):
                    r_start, r_end = i * h // n_regions, (i + 1) * h // n_regions
                    c_start, c_end = j * w // n_regions, (j + 1) * w // n_regions
                    
                    region = image[r_start:r_end, c_start:c_end]
                    if mask is not None:
                        region_mask = mask[r_start:r_end, c_start:c_end]
                    else:
                        region_mask = region > 0
                    
                    if np.sum(region_mask) > 10:  # Ensure enough voxels
                        region_values = region[region_mask]
                        if len(region_values) > 0 and np.mean(region_values) > 0:
                            cv = np.std(region_values) / np.mean(region_values)
                            region_cvs.append(cv)
            
            return np.mean(region_cvs) if region_cvs else 0
        
        metrics['spatial_cv_raw'] = regional_cv(raw_slice, mask_slice)
        metrics['spatial_cv_preprocessed'] = regional_cv(prep_slice, mask_slice)
        
        return metrics
    
    def estimate_bias_field(self):
        """Estimate the bias field by computing the ratio of raw to preprocessed."""
        # Avoid division by zero
        bias_field = np.divide(self.raw_data, self.preprocessed_data,
                              out=np.ones_like(self.raw_data),
                              where=self.preprocessed_data != 0)
        
        # Apply brain mask and smooth
        if self.brain_mask is not None:
            bias_field[~self.brain_mask] = 1.0
        bias_field = ndimage.gaussian_filter(bias_field, sigma=1.0)
        
        return bias_field
    
    def generate_plots(self, metrics, save_path=None):
        """Generate comprehensive visualization plots with proper aspect ratios."""
        fig = plt.figure(figsize=(20, 16))
        
        # Choose the best slice for visualization
        slice_idx, slice_orientation = self.choose_best_slice()
        raw_slice, prep_slice, mask_slice = self.get_slice_data(slice_idx, slice_orientation)
        
        # Get proper aspect ratio for this slice orientation
        aspect_ratio = self.get_proper_aspect_ratio(slice_orientation)
        
        # Estimate bias field
        bias_field = self.estimate_bias_field()
        if slice_orientation == 'axial':
            bias_slice = bias_field[:, :, slice_idx]
        elif slice_orientation == 'coronal':
            bias_slice = bias_field[:, slice_idx, :]
        else:  # sagittal
            bias_slice = bias_field[slice_idx, :, :]
        
        # 1. Original images comparison with proper aspect ratio
        plt.subplot(3, 4, 1)
        plt.imshow(raw_slice, cmap='gray', 
                  vmin=np.percentile(raw_slice, 1), 
                  vmax=np.percentile(raw_slice, 99), 
                  aspect=aspect_ratio)
        plt.title(f'Raw Image ({slice_orientation} slice {slice_idx})')
        plt.axis('off')
        
        plt.subplot(3, 4, 2)
        plt.imshow(prep_slice, cmap='gray', 
                  vmin=np.percentile(prep_slice, 1), 
                  vmax=np.percentile(prep_slice, 99), 
                  aspect=aspect_ratio)
        plt.title(f'Preprocessed Image ({slice_orientation} slice {slice_idx})')
        plt.axis('off')
        
        plt.subplot(3, 4, 3)
        plt.imshow(bias_slice, cmap='hot', vmin=0.5, vmax=1.5, aspect=aspect_ratio)
        plt.title('Estimated Bias Field')
        plt.colorbar(shrink=0.6)
        plt.axis('off')
        
        plt.subplot(3, 4, 4)
        if mask_slice is not None:
            plt.imshow(mask_slice, cmap='gray', aspect=aspect_ratio)
        else:
            plt.imshow(prep_slice > 0, cmap='gray', aspect=aspect_ratio)
        plt.title('Brain Mask')
        plt.axis('off')
        
        # 2. Intensity histograms
        raw_brain = self.raw_data[self.brain_mask]
        prep_brain = self.preprocessed_data[self.brain_mask]
        
        plt.subplot(3, 4, 5)
        plt.hist(raw_brain[raw_brain > 0], bins=100, alpha=0.7, label='Raw', density=True)
        plt.hist(prep_brain[prep_brain > 0], bins=100, alpha=0.7, label='Preprocessed', density=True)
        plt.xlabel('Intensity')
        plt.ylabel('Density')
        plt.title('Intensity Histograms')
        plt.legend()
        plt.yscale('log')
        
        # 3. Profile plots (horizontal and vertical through center)
        center_y, center_x = np.array(raw_slice.shape) // 2
        
        plt.subplot(3, 4, 6)
        plt.plot(raw_slice[center_y, :], label='Raw', alpha=0.8)
        plt.plot(prep_slice[center_y, :], label='Preprocessed', alpha=0.8)
        plt.xlabel('Position (pixels)')
        plt.ylabel('Intensity')
        plt.title('Horizontal Profile')
        plt.legend()
        
        plt.subplot(3, 4, 7)
        plt.plot(raw_slice[:, center_x], label='Raw', alpha=0.8)
        plt.plot(prep_slice[:, center_x], label='Preprocessed', alpha=0.8)
        plt.xlabel('Position (pixels)')
        plt.ylabel('Intensity')
        plt.title('Vertical Profile')
        plt.legend()
        
        # 4. Scatter plot
        plt.subplot(3, 4, 8)
        sample_indices = np.random.choice(len(raw_brain), min(5000, len(raw_brain)), replace=False)
        plt.scatter(raw_brain[sample_indices], prep_brain[sample_indices], alpha=0.1, s=1)
        plt.xlabel('Raw Intensity')
        plt.ylabel('Preprocessed Intensity')
        plt.title('Intensity Correlation')
        
        # Add correlation coefficient
        correlation = np.corrcoef(raw_brain, prep_brain)[0, 1]
        plt.text(0.05, 0.95, f'r = {correlation:.3f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 5. Metrics comparison
        plt.subplot(3, 4, 9)
        cv_data = [metrics['cv_raw'], metrics['cv_preprocessed']]
        plt.bar(['Raw', 'Preprocessed'], cv_data, color=['red', 'blue'], alpha=0.7)
        plt.ylabel('Coefficient of Variation')
        plt.title('CV Comparison (Lower is Better)')
        for i, v in enumerate(cv_data):
            plt.text(i, v, f'{v:.3f}', ha='center', va='bottom')
        
        # 6. SNR comparison
        plt.subplot(3, 4, 10)
        snr_data = [metrics['snr_raw'], metrics['snr_preprocessed']]
        
        # Check for valid SNR values
        if any(np.isnan(snr_data)) or any(np.isinf(snr_data)) or max(snr_data) == 0:
            plt.text(0.5, 0.5, 'SNR calculation\nunavailable\n(insufficient data)', 
                    ha='center', va='center', transform=plt.gca().transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            plt.title('SNR Comparison (Higher is Better)')
            plt.xticks([])
            plt.yticks([])
        else:
            plt.bar(['Raw', 'Preprocessed'], snr_data, color=['red', 'blue'], alpha=0.7)
            plt.ylabel('Signal-to-Noise Ratio')
            plt.title('SNR Comparison (Higher is Better)')
            # Ensure y-axis starts from 0 to show relative differences clearly
            plt.ylim(0, max(snr_data) * 1.1)
            for i, v in enumerate(snr_data):
                plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
        
        # 7. Difference image
        plt.subplot(3, 4, 11)
        diff_slice = raw_slice - prep_slice
        plt.imshow(diff_slice, cmap='RdBu_r', 
                  vmin=np.percentile(diff_slice, 5), 
                  vmax=np.percentile(diff_slice, 95), 
                  aspect=aspect_ratio)
        plt.title('Difference (Raw - Preprocessed)')
        plt.colorbar(shrink=0.6)
        plt.axis('off')
        
        # 8. Summary text
        plt.subplot(3, 4, 12)
        plt.axis('off')
        summary_text = f"""
        BIAS CORRECTION ASSESSMENT
        
        Image Info:
        • Slice: {slice_orientation} {slice_idx}
        • Voxel spacing: {self.voxel_spacing[:3]}
        • Aspect ratio: {aspect_ratio:.3f}
        
        Coefficient of Variation:
        • Raw: {metrics['cv_raw']:.4f}
        • Preprocessed: {metrics['cv_preprocessed']:.4f}
        • Ratio: {metrics['cv_ratio']:.4f}
        
        Signal-to-Noise Ratio:
        • Raw: {metrics['snr_raw']:.2f}
        • Preprocessed: {metrics['snr_preprocessed']:.2f}
        
        Entropy:
        • Raw: {metrics['entropy_raw']:.2f}
        • Preprocessed: {metrics['entropy_preprocessed']:.2f}
        
        INTERPRETATION:
        • CV ratio < 1.0: Good correction
        • CV ratio ≈ 1.0: Minimal correction
        • CV ratio > 1.0: Possible overcorrection
        
        Current CV ratio: {metrics['cv_ratio']:.3f}
        """
        
        if metrics['cv_ratio'] < 0.9:
            status = "✓ GOOD CORRECTION"
            color = 'green'
        elif metrics['cv_ratio'] < 1.1:
            status = "⚠ MINIMAL CORRECTION"
            color = 'orange'
        else:
            status = "✗ POSSIBLE OVERCORRECTION"
            color = 'red'
        
        summary_text += f"\nStatus: {status}"
        
        plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
                verticalalignment='top', fontfamily='monospace', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def print_detailed_report(self, intensity_metrics, spatial_metrics):
        """Print a detailed assessment report."""
        print("\n" + "="*60)
        print("BIAS FIELD CORRECTION QUALITY ASSESSMENT REPORT")
        print("="*60)
        
        print(f"\nIMAGE INFORMATION:")
        print(f"   Image shape: {self.raw_data.shape}")
        print(f"   Voxel spacing: {self.voxel_spacing}")
        
        print("\n1. INTENSITY METRICS:")
        print(f"   Coefficient of Variation (CV):")
        print(f"   • Raw image:        {intensity_metrics['cv_raw']:.6f}")
        print(f"   • Preprocessed:     {intensity_metrics['cv_preprocessed']:.6f}")
        print(f"   • Ratio (prep/raw): {intensity_metrics['cv_ratio']:.6f}")
        
        print(f"\n   Signal-to-Noise Ratio (SNR):")
        if np.isnan(intensity_metrics['snr_raw']) or np.isinf(intensity_metrics['snr_raw']):
            print(f"   • Raw image:        N/A (calculation failed)")
        else:
            print(f"   • Raw image:        {intensity_metrics['snr_raw']:.3f}")
            
        if np.isnan(intensity_metrics['snr_preprocessed']) or np.isinf(intensity_metrics['snr_preprocessed']):
            print(f"   • Preprocessed:     N/A (calculation failed)")
        else:
            print(f"   • Preprocessed:     {intensity_metrics['snr_preprocessed']:.3f}")
        
        print(f"\n   Intensity Range (95th - 5th percentile):")
        print(f"   • Raw image:        {intensity_metrics['range_raw']:.2f}")
        print(f"   • Preprocessed:     {intensity_metrics['range_preprocessed']:.2f}")
        
        print(f"\n   Entropy:")
        print(f"   • Raw image:        {intensity_metrics['entropy_raw']:.3f}")
        print(f"   • Preprocessed:     {intensity_metrics['entropy_preprocessed']:.3f}")
        
        print("\n2. SPATIAL METRICS:")
        print(f"   Regional CV (spatial uniformity):")
        print(f"   • Raw image:        {spatial_metrics['spatial_cv_raw']:.6f}")
        print(f"   • Preprocessed:     {spatial_metrics['spatial_cv_preprocessed']:.6f}")
        
        print("\n3. ASSESSMENT:")
        cv_ratio = intensity_metrics['cv_ratio']
        
        if cv_ratio < 0.8:
            assessment = "EXCELLENT - Strong bias field correction achieved"
            recommendation = "The bias field correction appears to be working very well."
        elif cv_ratio < 0.9:
            assessment = "GOOD - Effective bias field correction"
            recommendation = "The bias field correction is working well."
        elif cv_ratio < 1.1:
            assessment = "MINIMAL - Limited bias field correction"
            recommendation = "Consider adjusting correction parameters for better results."
        else:
            assessment = "POOR - Possible overcorrection or failed correction"
            recommendation = "Review correction parameters. Overcorrection may be occurring."
        
        print(f"   Status: {assessment}")
        print(f"   Recommendation: {recommendation}")
        
        print("\n4. KEY INDICATORS:")
        print(f"   • CV reduction: {((1 - cv_ratio) * 100):.1f}%")
        
        # Handle SNR change calculation safely
        snr_raw = intensity_metrics['snr_raw']
        snr_prep = intensity_metrics['snr_preprocessed']
        if not (np.isnan(snr_raw) or np.isinf(snr_raw) or np.isnan(snr_prep) or np.isinf(snr_prep)):
            print(f"   • SNR change: {snr_prep - snr_raw:.2f}")
        else:
            print(f"   • SNR change: N/A (calculation unavailable)")
        
        if spatial_metrics['spatial_cv_preprocessed'] < spatial_metrics['spatial_cv_raw']:
            print(f"   • Spatial uniformity: IMPROVED")
        else:
            print(f"   • Spatial uniformity: UNCHANGED or DEGRADED")
        
        print("\n" + "="*60)
    
    def run_evaluation(self, save_plot=None):
        """Run the complete evaluation pipeline."""
        print("Starting bias field correction evaluation...")
        
        # Load images
        self.load_images()
        
        # Create brain mask
        self.create_improved_brain_mask(method='robust')
        
        # Compute metrics
        print("Computing intensity metrics...")
        intensity_metrics = self.compute_intensity_metrics()
        
        print("Computing spatial metrics...")
        spatial_metrics = self.compute_spatial_metrics()
        
        # Generate report
        self.print_detailed_report(intensity_metrics, spatial_metrics)
        
        # Generate plots
        print("Generating visualization plots...")
        self.generate_plots(intensity_metrics, save_plot)
        
        return intensity_metrics, spatial_metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate bias field correction quality')
    parser.add_argument('--raw', required=True, help='Path to raw MRI image')
    parser.add_argument('--preprocessed', required=True, help='Path to preprocessed MRI image')
    parser.add_argument('--save-plot', help='Path to save the evaluation plot')
    
    args = parser.parse_args()
    
    try:
        evaluator = BiasFieldEvaluator(args.raw, args.preprocessed)
        intensity_metrics, spatial_metrics = evaluator.run_evaluation(args.save_plot)
        
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise

if __name__ == "__main__":
    main()