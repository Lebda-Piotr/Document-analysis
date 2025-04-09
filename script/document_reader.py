import cv2
import pytesseract
import re
import json
import os
from typing import List, Dict
from datetime import datetime
from preprocessing import process_document
from langdetect import detect, LangDetectException
from utils import convert_pdf_to_images, get_supported_files
from document_fields import DocumentFieldDetector

def detect_language(text: str) -> str:
    """Detects text language"""
    try:
        lang = detect(text)
        if lang == 'pl':
            return 'pl'
        elif lang == 'nl':
            return 'nl'
        elif lang == 'de':
            return 'du'
        else:
            return 'en'
    except LangDetectException:
        return 'en'  # Default to English

def basic_preprocessing(gray):
    """Basic image preprocessing"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    gray = cv2.medianBlur(gray, 3)
    return gray

def perform_ocr(image, preprocess='none', lang='pol+eng'):
    """
    Performs OCR on an image
    
    Args:
        image: Image to analyze
        preprocess: 'none', 'light' or 'basic'
        lang: Tesseract language code
    
    Returns:
        Tuple: (raw_text, detected_language)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    if preprocess == 'light':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
    elif preprocess == 'basic':
        gray = basic_preprocessing(gray)
    
    custom_config = r'--oem 3 --psm 6'
    raw_text = pytesseract.image_to_string(gray, config=custom_config, lang=lang)
    detected_lang = detect_language(raw_text)
    
    return raw_text, detected_lang

def extract_dates(text: str) -> List[str]:
    """Extracts dates from text"""
    date_patterns = [
        (r'\b\d{2}-\d{2}-\d{4}\b', '%d-%m-%Y'),
        (r'\b\d{2}/\d{2}/\d{4}\b', '%d/%m/%Y'),
        (r'\b\d{4}-\d{2}-\d{2}\b', '%Y-%m-%d'),
        (r'\b\d{1,2}\s+[a-z]+\s+\d{4}\b', '%d %B %Y')
    ]
    
    found_dates = []
    for pattern, date_format in date_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            try:
                date_str = match.group()
                datetime.strptime(date_str, date_format)
                found_dates.append(date_str)
            except ValueError:
                continue
    return list(set(found_dates))

def extract_document_numbers(text: str) -> List[str]:
    """Extracts document numbers from text"""
    patterns = [
        r'\b[A-Z]{2,}/\d{3,}/\d{4}\b',
        r'\bNr\s*\.?\s*\d{3,}[/-]\d{3,}\b',
        r'\b\d{3,}[/-]\d{3,}\b'
    ]
    numbers = []
    for pattern in patterns:
        numbers.extend(re.findall(pattern, text, flags=re.IGNORECASE))
    return numbers

def extract_emails(text: str) -> List[str]:
    """Extracts email addresses from text"""
    return list(set(re.findall(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        text,
        flags=re.IGNORECASE
    )))

def extract_phones(text: str) -> List[str]:
    """Extracts phone numbers from text"""
    patterns = [
        r'\b\d{3}[-\s]?\d{3}[-\s]?\d{3}\b',
        r'\b\d{2}[-\s]?\d{3}[-\s]?\d{2}[-\s]?\d{2}\b',
        r'\+\d{2}\s?\d{3}\s?\d{3}\s?\d{3}\b'
    ]
    phones = []
    for pattern in patterns:
        phones.extend(re.findall(pattern, text))
    return list(set(phones))

def extract_metadata(text: str) -> Dict[str, List[str]]:
    """Extracts metadata from text"""
    return {
        'dates': extract_dates(text),
        'document_numbers': extract_document_numbers(text),
        'emails': extract_emails(text),
        'phones': extract_phones(text)
    }

def save_metadata(data: Dict, filename: str = "metadata.json"):
    """Saves metadata to JSON file"""
    output_path = os.path.join('data', 'output', filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved metadata to {output_path}")

def save_text_comparison(raw_text: str, filename: str = "ocr_output.txt"):
    """Saves OCR output to text file"""
    output_path = os.path.join('data', 'output', filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("OCR RESULT:\n")
        f.write("=" * 80 + "\n\n")
        f.write(raw_text)
    print(f"Saved OCR result to {output_path}")

def print_metadata(metadata: Dict):
    """Prints metadata in readable format"""
    print("\nExtracted metadata:")
    print(f"Dates: {', '.join(metadata.get('dates', []))}")
    print(f"Document numbers: {', '.join(metadata.get('document_numbers', []))}")
    print(f"Emails: {', '.join(metadata.get('emails', []))}")
    print(f"Phones: {', '.join(metadata.get('phones', []))}")

def process_document_with_metadata(input_path: str, is_pdf=False):
    """
    Processes document and extracts metadata
    
    Args:
        input_path: Path to input document (image or PDF)
        is_pdf: Whether the input is a PDF file
    
    Returns:
        List of dictionaries with processing results (one per page if PDF)
    """
    results = []
    
    if is_pdf:
        # Convert PDF to images
        images = convert_pdf_to_images(input_path)
        
        # Process each page
        for i, image in enumerate(images):
            print(f"\nProcessing PDF page {i+1}...")
            result = _process_single_image(image, f"{os.path.basename(input_path)}_page_{i+1}")
            results.append(result)
    else:
        # Process single image
        processed_image = process_document(
            input_path, 
            os.path.join('data', 'output', 'temp_processed.jpg'), 
            show_plots=False
        )
        result = _process_single_image(processed_image, os.path.basename(input_path))
        results.append(result)
    
    return results

def _process_single_image(image, source_name):
    """Helper function to process a single image"""
    # Perform OCR
    raw_text, language = perform_ocr(image)
    
    # Save OCR result with unique name
    base_name = os.path.splitext(source_name)[0]
    save_text_comparison(raw_text, f"ocr_output_{base_name}.txt")
    
    # Print language info
    print(f"\nDetected language: {'Polish' if language == 'pl' else 'English'}")
    print(f"OCR result saved to ocr_output_{base_name}.txt")
    
    # Extract metadata
    metadata = extract_metadata(raw_text)
    result = {
        'source': source_name,
        'language': language,
        'raw_text': raw_text,
        'metadata': metadata
    }
    
    # Save and display results
    save_metadata(result, f"metadata_{base_name}.json")
    print_metadata(metadata)
    
    return result

if __name__ == "__main__":
    input_path = os.path.join('data', 'input', 'doc11.jpg')
    process_document_with_metadata(input_path)