# file_utils.py
import os
from pdf2image import convert_from_path
from PIL import Image
import cv2
import numpy as np

def convert_pdf_to_images(pdf_path, output_folder=None):
    """Convert PDF to list of images using pdf2image"""
    images = []
    
    try:
        pil_images = convert_from_path(
            pdf_path,
            dpi=300,
            fmt='jpeg',
            thread_count=4,
            poppler_path=get_poppler_path()
        )
        
        for i, pil_image in enumerate(pil_images):
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            if output_folder:
                os.makedirs(output_folder, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(pdf_path))[0]
                output_path = os.path.join(output_folder, f"{base_name}_page_{i+1}.jpg")
                cv2.imwrite(output_path, cv_image)
            
            images.append(cv_image)
            
    except Exception as e:
        print(f"Error converting PDF: {str(e)}")
        raise
    
    return images

def get_poppler_path():
    """Helper to find poppler path across different OSes"""
    # Try common paths - adjust as needed for your deployment
    possible_paths = [
        '/usr/bin',                  # Linux default
        '/usr/local/bin',            # Mac Homebrew
        r'C:\Program Files\poppler-<version>\bin',  # Windows
        r'C:\poppler\bin'            # Windows alternative
    ]
    
    for path in possible_paths:
        if os.path.exists(os.path.join(path, 'pdftoppm')):
            return path
    
    return None  # Will use system PATH if not found

def get_supported_files(folder_path):
    """Get all supported files (images and PDFs) in folder and subfolders"""
    supported_extensions = ('.jpg', '.jpeg', '.png', '.pdf')
    file_paths = []
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(supported_extensions):
                file_paths.append(os.path.join(root, file))
    
    return file_paths