import os
import json
import cv2
import pytesseract
from PIL import Image
from typing import List, Dict, Optional
from datetime import datetime
import re

class DocumentFieldDetector:
    def __init__(self, output_dir: str = "../data/output/dataset"):
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.annotations_path = os.path.join(output_dir, "annotations.json")
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Initialize with empty annotations if file doesn't exist
        if not os.path.exists(self.annotations_path):
            with open(self.annotations_path, "w") as f:
                json.dump({}, f)

    def detect_fields(self, image_path: str) -> Dict:
        """Detect text fields in a document and return structured annotations"""
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
            
        # Convert to RGB and save to dataset directory
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Generate unique ID and save image
        image_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"{image_id}.jpg"
        image_save_path = os.path.join(self.images_dir, image_filename)
        pil_image.save(image_save_path, quality=95)
        
        # Perform OCR with layout analysis
        d = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        # Group words into paragraphs
        paragraphs = self._group_words_into_paragraphs(d)
        
        # Extract fields and their bounding boxes
        annotations = []
        for paragraph in paragraphs:
            text = " ".join([word['text'] for word in paragraph])
            bbox = self._get_paragraph_bbox(paragraph)
            field_type = self._classify_field(text)
            annotations.append({
                "field": field_type if field_type else "Unknown",
                "value": text,
                "bbox": bbox
            })
        
        # Prepare annotation structure
        annotation_data = {
            "image_id": image_id,
            "image_path": image_save_path,
            "annotations": annotations
        }
        
        # Save to annotations file
        self._save_annotation(annotation_data)
        
        return annotation_data

    def _group_words_into_paragraphs(self, ocr_data: Dict) -> List[List[Dict]]:
        """Group words into paragraphs based on their proximity"""
        paragraphs = []
        current_paragraph = []
        last_y = None
        
        for i in range(len(ocr_data['text'])):
            if int(ocr_data['conf'][i]) > 60:  # Only consider confident detections
                word_data = {
                    'text': ocr_data['text'][i].strip(),
                    'x': ocr_data['left'][i],
                    'y': ocr_data['top'][i],
                    'w': ocr_data['width'][i],
                    'h': ocr_data['height'][i]
                }
                if last_y is None or abs(word_data['y'] - last_y) < 20:  # Adjust threshold as needed
                    current_paragraph.append(word_data)
                else:
                    paragraphs.append(current_paragraph)
                    current_paragraph = [word_data]
                last_y = word_data['y']
        
        if current_paragraph:
            paragraphs.append(current_paragraph)
        
        return paragraphs

    def _get_paragraph_bbox(self, paragraph: List[Dict]) -> List[int]:
        """Calculate the bounding box for a paragraph"""
        x_min = min(word['x'] for word in paragraph)
        y_min = min(word['y'] for word in paragraph)
        x_max = max(word['x'] + word['w'] for word in paragraph)
        y_max = max(word['y'] + word['h'] for word in paragraph)
        return [x_min, y_min, x_max, y_max]

    def _classify_field(self, text: str) -> Optional[str]:
        """Classify text into field types based on patterns"""
        # Date patterns
        date_patterns = [
            r'\b\d{2}-\d{2}-\d{4}\b',
            r'\b\d{2}/\d{2}/\d{4}\b',
            r'\b\d{4}-\d{2}-\d{2}\b',
            r'\b\d{1,2}\s+[a-z]+\s+\d{4}\b'
        ]
        
        for pattern in date_patterns:
            if re.fullmatch(pattern, text, flags=re.IGNORECASE):
                return "Date"
        
        # Document number patterns
        doc_patterns = [
            r'\b[A-Z]{2,}/\d{3,}/\d{4}\b',
            r'\bNr\s*\.?\s*\d{3,}[/-]\d{3,}\b',
            r'\b\d{3,}[/-]\d{3,}\b'
        ]
        
        for pattern in doc_patterns:
            if re.fullmatch(pattern, text, flags=re.IGNORECASE):
                return "Document_Number"
        
        # Email pattern
        if re.fullmatch(r'\b[A-Za-z0-9._%+-]+@[A-Za-z09.-]+\.[A-Z|a-z]{2,}\b', text):
            return "Email"
            
        # Phone pattern
        phone_patterns = [
            r'\b\d{3}[-\s]?\d{3}[-\s]?\d{3}\b',
            r'\b\d{2}[-\s]?\d{3}[-\s]?\d{2}[-\s]?\d{2}\b',
            r'\+\d{2}\s?\d{3}\s?\d{3}\s?\d{3}\b'
        ]
        
        for pattern in phone_patterns:
            if re.fullmatch(pattern, text):
                return "Phone"
        
        # If no specific pattern matches, return None (we'll skip this field)
        return None

    def _save_annotation(self, new_data: Dict):
        """Save new annotation to the annotations file"""
        # Load existing annotations
        with open(self.annotations_path, "r") as f:
            all_annotations = json.load(f)
        
        # Add new annotation
        all_annotations[new_data["image_id"]] = new_data
        
        # Save back to file
        with open(self.annotations_path, "w") as f:
            json.dump(all_annotations, f, indent=2)