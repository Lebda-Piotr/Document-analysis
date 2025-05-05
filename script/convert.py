import os
from PIL import Image
from pdf2image import convert_from_path
import shutil

def convert_pdf_to_jpg(pdf_path, output_folder):
    """Convert a PDF file to JPG images using pdf2image"""
    try:
        # Convert PDF to list of images
        # Added dpi parameter to improve quality and poppler_path for Windows users
        images = convert_from_path(
            pdf_path, 
            fmt='jpeg',
            dpi=300,  # Higher DPI for better quality
            # Uncomment and set this path if you're on Windows
            poppler_path=r"E:\Nowy folder (2)\praktyki\documentation\script\poppler-24.08.0\Library\bin"
        )
        
        # Check if images were actually created
        if not images or len(images) == 0:
            print(f"Warning: No images were extracted from {pdf_path}")
            return
            
        # Save each page as separate JPG
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        for i, image in enumerate(images):
            if len(images) > 1:
                output_path = os.path.join(output_folder, f"{base_name}_page{i+1}.jpg")
            else:
                output_path = os.path.join(output_folder, f"{base_name}.jpg")
            
            # Ensure we're working with RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            image.save(output_path, 'JPEG', quality=95)
            
            # Verify the image was saved correctly
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"Converted: {output_path}")
            else:
                print(f"Warning: Failed to save {output_path} properly")
    except Exception as e:
        print(f"Failed to convert PDF {pdf_path}: {e}")

def convert_image_to_jpg(image_path, output_path):
    """Convert an image file to JPG format"""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if needed (JPG doesn't support alpha channel)
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            img.save(output_path, 'JPEG', quality=95)
    except Exception as e:
        print(f"Error converting image {image_path}: {e}")

def process_folder(input_folder, output_base):
    """Process all files in a folder and its subfolders"""
    for root, dirs, files in os.walk(input_folder):
        # Create corresponding output directory structure
        relative_path = os.path.relpath(root, input_folder)
        output_folder = os.path.join(output_base, relative_path)
        os.makedirs(output_folder, exist_ok=True)
        
        for file in files:
            file_lower = file.lower()
            input_path = os.path.join(root, file)
            
            # Get filename without extension
            base_name = os.path.splitext(file)[0]
            output_path = os.path.join(output_folder, f"{base_name}.jpg")
            
            if file_lower.endswith('.pdf'):
                print(f"Processing PDF: {input_path}")
                convert_pdf_to_jpg(input_path, output_folder)
            elif file_lower.endswith(('.jpeg', '.jpg', '.png', '.bmp', '.tiff', '.tif', '.gif')):
                # Skip if output already exists and is newer than input
                if os.path.exists(output_path) and os.path.getmtime(input_path) <= os.path.getmtime(output_path):
                    continue
                
                print(f"Processing image: {input_path}")
                convert_image_to_jpg(input_path, output_path)

def main():
    input_folder = input("Enter the input folder path: ").strip()
    output_folder = input("Enter the output folder path: ").strip()
    
    # Validate paths
    if not os.path.isdir(input_folder):
        print("Error: Input folder does not exist.")
        return
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    print("Starting conversion...")
    process_folder(input_folder, output_folder)
    print("Conversion completed!")

if __name__ == "__main__":
    # Print information about requirements
    print("PDF to JPG Converter")
    print("====================")
    print("Note: This program requires:")
    print("- poppler-utils on Linux/macOS (install via package manager)")
    print("- poppler for Windows (download from: https://github.com/oschwartz10612/poppler-windows/releases)")
    print("  If on Windows, uncomment and set the poppler_path in the code\n")
    
    main()