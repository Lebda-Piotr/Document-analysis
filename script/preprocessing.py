import os
import cv2
import numpy as np
from pathlib import Path
import logging
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import imutils
import unicodedata
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def detect_and_correct_skew(image, delta=1, limit=5):
    """
    Detect and correct skew in an image.
    
    Args:
        image: Input image
        delta: Angle step in degrees
        limit: Maximum angle to search
        
    Returns:
        Deskewed image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Threshold the image to get edges
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Find all angles within the limit with delta steps
    angles = np.arange(-limit, limit + delta, delta)
    scores = []
    
    # For each angle, rotate the image and calculate the projection profile
    for angle in angles:
        rotated = imutils.rotate_bound(thresh, angle)
        hist = cv2.reduce(rotated, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
        scores.append(np.sum((hist[1:] - hist[:-1]) ** 2))
    
    # Find the best angle
    best_angle_idx = np.argmax(scores)
    best_angle = angles[best_angle_idx]
    
    # Only rotate if the angle is significant (avoid small rotations)
    if abs(best_angle) >= 0.5:
        rotated = imutils.rotate_bound(image, best_angle)
        logger.debug(f"Corrected skew by {best_angle} degrees")
        return rotated
    
    return image

def adaptive_threshold(image):
    """
    Apply adaptive thresholding to enhance text visibility.
    
    Args:
        image: Grayscale input image
        
    Returns:
        Processed image
    """
    # Apply adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    return adaptive

def enhance_contrast(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        image: Grayscale input image
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
        
    Returns:
        Contrast enhanced image
    """
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # Apply CLAHE
    enhanced = clahe.apply(image)
    
    return enhanced

def remove_noise(image, strength=7):
    """
    Remove noise while preserving edges.
    
    Args:
        image: Input grayscale image
        strength: Denoising strength
        
    Returns:
        Denoised image
    """
    # Non-local means denoising
    denoised = cv2.fastNlMeansDenoising(image, None, strength, 7, 21)
    
    return denoised

def detect_document_borders(image):
    """
    Detect document borders and crop to content area.
    
    Args:
        image: Input image
        
    Returns:
        Cropped image focusing on document content
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Blur the image to remove noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply edge detection
    edges = cv2.Canny(blurred, 75, 200)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours found, return original
    if not contours:
        return image
    
    # Find the largest contour (assuming it's the document)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # If contour is too small, it's likely not a document
    if cv2.contourArea(largest_contour) < 0.1 * image.shape[0] * image.shape[1]:
        return image
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Add a small margin
    margin = 10
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(image.shape[1] - x, w + 2 * margin)
    h = min(image.shape[0] - y, h + 2 * margin)
    
    # Crop the image
    cropped = image[y:y+h, x:x+w]
    
    return cropped

def binarize_image(image, method='adaptive'):
    """
    Convert image to binary (black and white) using different methods.
    
    Args:
        image: Grayscale input image
        method: Binarization method ('otsu', 'adaptive', or 'sauvola')
        
    Returns:
        Binary image
    """
    if method == 'otsu':
        # Apply Otsu's method
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'adaptive':
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
    else:  # Default to adaptive
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
    
    return binary

def apply_unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """
    Apply unsharp masking to sharpen the image.
    
    Args:
        image: Input image
        kernel_size: Size of Gaussian kernel
        sigma: Standard deviation of Gaussian
        amount: Strength of sharpening
        threshold: Minimum brightness change
        
    Returns:
        Sharpened image
    """
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    
    # Apply threshold - only apply sharpening where the difference is above threshold
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        sharpened[low_contrast_mask] = image[low_contrast_mask]
    
    return sharpened

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    """
    Adjust brightness and contrast of an image.
    
    Args:
        image: Input image
        brightness: Brightness adjustment (-100 to 100)
        contrast: Contrast adjustment (-100 to 100)
        
    Returns:
        Adjusted image
    """
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        
        image = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)
    
    if contrast != 0:
        alpha_c = 131 * (contrast + 127) / (127 * (131 - contrast))
        gamma_c = 127 * (1 - alpha_c)
        
        image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)
    
    return image

def process_image(input_path, output_path):
    """
    Process a single image with multiple enhancement techniques.
    
    Args:
        input_path: Path to input image
        output_path: Path to save processed image
        
    Returns:
        Success status (True/False)
    """
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Read the image with Unicode path support
        with open(input_path, 'rb') as f:
            img_array = np.frombuffer(f.read(), dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.error(f"Failed to read image: {input_path}")
            return False
        
        # Store original dimensions
        original_height, original_width = image.shape[:2]
        
        # Auto-detect and crop to document borders
        image = detect_document_borders(image)
        
        # Detect and correct skew
        image = detect_and_correct_skew(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Denoise (but conservatively to preserve text)
        denoised = remove_noise(gray, strength=5)
        
        # Enhance contrast using CLAHE
        enhanced = enhance_contrast(denoised, clip_limit=2.0)
        
        # Apply gentle sharpening with unsharp mask
        sharpened = apply_unsharp_mask(enhanced, amount=0.8, threshold=5)
        
        # Ensure the image has a minimum resolution for OCR
        min_resolution = 300  # DPI
        current_resolution = max(image.shape[0], image.shape[1])
        
        if current_resolution < 1000:  # If image is small, resize to improve OCR
            scale_factor = 1.5
            new_width = int(image.shape[1] * scale_factor)
            new_height = int(image.shape[0] * scale_factor)
            resized = cv2.resize(sharpened, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        else:
            resized = sharpened
        
        # Save the processed image with Unicode path support
        success = cv2.imwrite(str(output_path), resized)
        if not success:
            # Fallback to imencode if imwrite fails
            _, img_encoded = cv2.imencode('.png', resized)
            with open(output_path, 'wb') as f:
                f.write(img_encoded)
        
        logger.info(f"Successfully processed: {input_path} -> {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error processing {input_path}: {str(e)}")
        return False

def is_image_file(file_path):
    """
    Check if a file is an image based on its extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file is an image, False otherwise
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}
    return file_path.suffix.lower() in image_extensions

def preprocess_images(input_folder, output_folder, max_workers=None, max_files=None):
    """
    Preprocess all images in a folder, preserving the folder structure.
    
    Args:
        input_folder: Path to folder containing input images
        output_folder: Path to folder to save processed images
        max_workers: Maximum number of worker processes
        max_files: Maximum number of files to process (None for all)
        
    Returns:
        Tuple of (success_count, failed_count)
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files while preserving directory structure
    input_path = Path(input_folder)
    image_files = []
    output_files = []
    
    for root, _, files in os.walk(input_folder):
        rel_path = os.path.relpath(root, input_folder)
        out_dir = os.path.join(output_folder, rel_path) if rel_path != '.' else output_folder
        
        for file in files:
            file_path = Path(os.path.join(root, file))
            if is_image_file(file_path):
                # Handle Unicode paths properly
                try:
                    file_path_str = str(file_path)
                    out_file = os.path.join(out_dir, file)
                    image_files.append(file_path_str)
                    output_files.append(Path(out_file))
                except UnicodeEncodeError:
                    logger.warning(f"Skipping file with problematic characters: {file_path}")
                    continue
    
    # Limit files if requested
    if max_files is not None:
        image_files = image_files[:max_files]
        output_files = output_files[:max_files]
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Process images in parallel
    success_count = 0
    failed_count = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Map input and output paths
        results = list(tqdm(
            executor.map(process_image, image_files, output_files),
            total=len(image_files),
            desc="Processing images"
        ))
    
    # Count successes and failures
    success_count = sum(results)
    failed_count = len(results) - success_count
    
    logger.info(f"Preprocessing complete: {success_count} succeeded, {failed_count} failed")
    
    return success_count, failed_count

def get_user_input():
    """
    Get input and output folders from user.
    
    Returns:
        Tuple of (input_folder, output_folder)
    """
    parser = argparse.ArgumentParser(description='Preprocess images for OCR with support for Polish characters')
    parser.add_argument('-i', '--input', help='Input folder containing images (default: ask interactively)')
    parser.add_argument('-o', '--output', help='Output folder for processed images (default: ask interactively)')
    parser.add_argument('-w', '--workers', type=int, help='Number of worker processes (default: 80% of CPU cores)')
    parser.add_argument('-m', '--max-files', type=int, help='Maximum number of files to process (default: all)')
    
    args = parser.parse_args()
    
    input_folder = args.input
    output_folder = args.output
    max_workers = args.workers
    max_files = args.max_files
    
    # If input folder not provided as argument, ask interactively
    if not input_folder:
        input_folder = input("Enter the input folder path: ")
    
    # If output folder not provided as argument, ask interactively
    if not output_folder:
        output_folder = input("Enter the output folder path: ")
    
    # Normalize paths to handle Unicode characters properly
    input_folder = os.path.normpath(input_folder)
    output_folder = os.path.normpath(output_folder)
    
    # Calculate default max_workers if not specified
    if max_workers is None:
        import multiprocessing
        max_workers = max(1, int(multiprocessing.cpu_count() * 0.8))
    
    return input_folder, output_folder, max_workers, max_files

if __name__ == "__main__":
    # Get user input
    input_folder, output_folder, max_workers, max_files = get_user_input()
    
    logger.info(f"Processing images from: {input_folder}")
    logger.info(f"Saving processed images to: {output_folder}")
    logger.info(f"Using {max_workers} worker processes")
    
    if max_files:
        logger.info(f"Processing up to {max_files} files")
    
    # Process the images
    preprocess_images(input_folder, output_folder, max_workers=max_workers, max_files=max_files)