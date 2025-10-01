"""
Medical Image Processing Pipeline
==================================

A comprehensive Python tool for medical image enhancement and analysis.

Features:
- DICOM/JPG/PNG support
- Intensity normalization and histogram equalization
- Gaussian denoising
- Multi-method edge detection (Laplacian, Canny)
- Comprehensive visualization

Author: [Your Name]
Date: October 2025
GitHub: github.com/yourusername/medical-image-processing

Usage:
    python medical_image_pipeline.py <image_path>
    
Example:
    python medical_image_pipeline.py data/ct_scan.dcm
    python medical_image_pipeline.py data/xray.jpg

Requirements:
    - OpenCV (cv2)
    - NumPy
    - Matplotlib
    - pydicom (optional, for DICOM support)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# DICOM support (optional)
try:
    import pydicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False
    print("pydicom not installed. DICOM support disabled.")
    print("   Install with: pip install pydicom\n")

def load_image(image_path):
    """
    Load image from file (supports JPG, PNG, DICOM)
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        numpy.ndarray: Grayscale image array
    """
    if image_path.lower().endswith('.dcm'):
        if not DICOM_AVAILABLE:
            raise ImportError("pydicom required for DICOM files. Install with: pip install pydicom")
        
        print("Loading DICOM file...")
        ds = pydicom.dcmread(image_path)
        img = ds.pixel_array
        
        # Print DICOM metadata
        if 'Modality' in ds:
            print(f"   Modality: {ds.Modality}")
        if 'PatientName' in ds:
            print(f"   Patient: {ds.PatientName}")
        print(f"   Image shape: {img.shape}")
        
        return img
    
    else:
        print("Loading image file...")
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot load image at {image_path}")
        print(f"   Image shape: {img.shape}")
        return img

def main_pipeline(image_path):
    """
    Complete medical image processing pipeline:
    1. Load image (supports DICOM/JPG/PNG)
    2. Normalize intensity
    3. Histogram equalization
    4. Gaussian blur for denoising
    5. Laplacian edge detection
    6. Canny edge detection
    7. Visualization & save results
    
    Args:
        image_path (str): Path to input image
    """
    
    # Create results folder
    os.makedirs("results", exist_ok=True)
    
    # -------------------------------
    # 1. Load image
    # -------------------------------
    img = load_image(image_path)
    
    # -------------------------------
    # 2. Intensity normalization
    # -------------------------------
    print("Normalizing intensity...")
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    
    # -------------------------------
    # 3. Histogram equalization
    # -------------------------------
    print("Applying histogram equalization...")
    img_eq = cv2.equalizeHist(img_norm.astype(np.uint8))
    
    # -------------------------------
    # 4. Gaussian blur (noise reduction)
    # -------------------------------
    print("Gaussian denoising...")
    img_blur = cv2.GaussianBlur(img_eq, (5, 5), 0)
    
    # -------------------------------
    # 5. Laplacian edge detection
    # -------------------------------
    print("Laplacian edge detection...")
    laplacian = cv2.Laplacian(img_blur, cv2.CV_64F)
    laplacian_norm = cv2.convertScaleAbs(laplacian)
    
    # -------------------------------
    # 6. Canny edge detection
    # -------------------------------
    print("Canny edge detection...")
    edges = cv2.Canny(img_blur, 100, 200)
    
    # -------------------------------
    # 7. Enhanced Visualization
    # -------------------------------
    print("Generating visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title("1. Original Image", fontsize=12, fontweight='bold')
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(img_norm, cmap='gray')
    axes[0, 1].set_title("2. Normalized", fontsize=12, fontweight='bold')
    axes[0, 1].axis("off")
    
    axes[0, 2].imshow(img_eq, cmap='gray')
    axes[0, 2].set_title("3. Histogram Equalized", fontsize=12, fontweight='bold')
    axes[0, 2].axis("off")
    
    axes[1, 0].imshow(img_blur, cmap='gray')
    axes[1, 0].set_title("4. Gaussian Denoising", fontsize=12, fontweight='bold')
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(laplacian_norm, cmap='hot')
    axes[1, 1].set_title("5. Laplacian Edges", fontsize=12, fontweight='bold')
    axes[1, 1].axis("off")
    
    axes[1, 2].imshow(edges, cmap='gray')
    axes[1, 2].set_title("6. Canny Edges", fontsize=12, fontweight='bold')
    axes[1, 2].axis("off")
    
    plt.suptitle("Medical Image Processing Pipeline", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save figure
    plt.savefig("results/complete_pipeline.png", dpi=150, bbox_inches='tight')
    print("Saved: results/complete_pipeline.png")
    
    plt.show()
    
    # -------------------------------
    # 8. Save individual results
    # -------------------------------
    print("Saving individual results...")
    cv2.imwrite("results/01_original.png", img)
    cv2.imwrite("results/02_normalized.png", img_norm)
    cv2.imwrite("results/03_equalized.png", img_eq)
    cv2.imwrite("results/04_denoised.png", img_blur)
    cv2.imwrite("results/05_laplacian.png", laplacian_norm)
    cv2.imwrite("results/06_canny.png", edges)
    
    # -------------------------------
    # 9. Print statistics
    # -------------------------------
    print("\n Processing Statistics:")
    print(f"   Original intensity range: [{img.min()}, {img.max()}]")
    print(f"   Normalized range: [{img_norm.min()}, {img_norm.max()}]")
    print(f"   Laplacian edges (>50): {np.sum(laplacian_norm > 50)} pixels")
    print(f"   Canny edges: {np.sum(edges > 0)} pixels")
    print(f"   Edge density: {np.sum(edges > 0) / edges.size * 100:.2f}%")
    

# -------------------------------
# Main execution
# -------------------------------
if __name__ == "__main__":
    import sys
    
    # Support command-line arguments
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default path (change this to your image)
        image_path = r"C:\Users\56813\Desktop\synpic57896.jpg"
    
    print("=" * 60)
    print(" Medical Image Processing Pipeline")
    print("=" * 60)
    print(f" Input: {image_path}\n")
    
    try:
        main_pipeline(image_path)
        print("\n" + "=" * 60)
        print(" Pipeline completed successfully!")
        print(" Check the 'results/' folder for all outputs.")
        print("=" * 60)
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()