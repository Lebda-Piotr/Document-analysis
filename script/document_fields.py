import os
import json
import cv2
import pytesseract
from PIL import Image
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import re
from preprocessing import enhance_image_quality

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

    def detect_fields(self, image_path: str, preprocess: bool = True) -> Dict:
        """Detect text fields in a document and return structured annotations"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
            
        # Convert to RGB and save to dataset directory
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply preprocessing if requested
        if preprocess:
            rgb_image = enhance_image_quality(rgb_image)
        
        pil_image = Image.fromarray(rgb_image)
        
        # Generate unique ID and save image
        image_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"{image_id}.jpg"
        image_save_path = os.path.join(self.images_dir, image_filename)
        pil_image.save(image_save_path, quality=95)
        
        # Perform OCR with layout analysis
        d = pytesseract.image_to_data(rgb_image, output_type=pytesseract.Output.DICT)
        
        # Group words into paragraphs with improved algorithm
        paragraphs = self._group_words_into_paragraphs(d)
        
        # Extract fields and their bounding boxes
        annotations = []
        for paragraph in paragraphs:
            # Skip paragraphs with just noise
            if not paragraph or len(paragraph) == 0:
                continue
                
            text = " ".join([word['text'] for word in paragraph])
            
            # Skip very short or noise text
            if len(text.strip()) < 2:
                continue
            
            bbox = self._get_paragraph_bbox(paragraph)
            field_type = self._classify_field(text)
            
            # Only include fields with a recognized type or with sufficient text
            if field_type or len(text.strip().split()) >= 2:
                annotations.append({
                    "field": field_type if field_type else "Unknown",
                    "value": text,
                    "bbox": bbox,
                    "confidence": self._calculate_confidence(paragraph)
                })
        
        # Merge overlapping annotations
        merged_annotations = self._merge_overlapping_annotations(annotations)
        
        # Prepare annotation structure
        annotation_data = {
            "image_id": image_id,
            "image_path": image_save_path,
            "annotations": merged_annotations
        }
        
        # Save to annotations file
        self._save_annotation(annotation_data)
        
        return annotation_data

    def _group_words_into_paragraphs(self, ocr_data: Dict) -> List[List[Dict]]:
        """Group words into paragraphs based on proximity and alignment"""
        if not ocr_data['text']:
            return []
            
        # Filter out low confidence and empty text
        filtered_words = []
        for i in range(len(ocr_data['text'])):
            if int(ocr_data['conf'][i]) > 50 and ocr_data['text'][i].strip():
                filtered_words.append({
                    'text': ocr_data['text'][i].strip(),
                    'x': ocr_data['left'][i],
                    'y': ocr_data['top'][i],
                    'w': ocr_data['width'][i],
                    'h': ocr_data['height'][i],
                    'conf': ocr_data['conf'][i]
                })
        
        if not filtered_words:
            return []
            
        # Sort words by y-coordinate (top to bottom)
        filtered_words.sort(key=lambda w: w['y'])
        
        # Group by lines first (close y values)
        lines = []
        current_line = [filtered_words[0]]
        avg_height = sum(w['h'] for w in filtered_words) / len(filtered_words)
        y_threshold = min(avg_height * 0.7, 15)  # Dynamic threshold based on text height
        
        for i in range(1, len(filtered_words)):
            if abs(filtered_words[i]['y'] - current_line[0]['y']) <= y_threshold:
                current_line.append(filtered_words[i])
            else:
                # Sort words in current line by x-coordinate
                current_line.sort(key=lambda w: w['x'])
                lines.append(current_line)
                current_line = [filtered_words[i]]
        
        # Add the last line
        if current_line:
            current_line.sort(key=lambda w: w['x'])
            lines.append(current_line)
        
        # Process lines to identify horizontally separated text blocks
        paragraphs = []
        
        for line in lines:
            if len(line) <= 1:
                paragraphs.append(line)
                continue
            
            # Calculate average word width and spacing in this line
            avg_word_width = sum(w['w'] for w in line) / len(line)
            
            # Calculate spaces between words
            spaces = []
            for i in range(1, len(line)):
                space = line[i]['x'] - (line[i-1]['x'] + line[i-1]['w'])
                spaces.append(space)
            
            # If we have enough spaces, calculate standard deviation to detect unusual gaps
            if spaces:
                avg_space = sum(spaces) / len(spaces)
                # If average space is too large or line has few words, treat as separate blocks
                if avg_space > avg_word_width * 2 or len(line) <= 3:
                    # Use a threshold of 2x the average space or 3x the average word width
                    space_threshold = max(avg_space * 2, avg_word_width * 3)
                    
                    current_block = [line[0]]
                    for i in range(1, len(line)):
                        space = line[i]['x'] - (line[i-1]['x'] + line[i-1]['w'])
                        if space <= space_threshold:
                            current_block.append(line[i])
                        else:
                            if current_block:
                                paragraphs.append(current_block)
                            current_block = [line[i]]
                    
                    if current_block:
                        paragraphs.append(current_block)
                else:
                    # Treat entire line as one paragraph if spaces are consistent
                    paragraphs.append(line)
            else:
                # Single word or no spaces
                paragraphs.append(line)
        
        return paragraphs

    def _get_paragraph_bbox(self, paragraph: List[Dict]) -> List[int]:
        """Calculate the bounding box for a paragraph with padding"""
        if not paragraph:
            return [0, 0, 0, 0]
            
        x_min = min(word['x'] for word in paragraph)
        y_min = min(word['y'] for word in paragraph)
        x_max = max(word['x'] + word['w'] for word in paragraph)
        y_max = max(word['y'] + word['h'] for word in paragraph)
        
        # Add small padding (5% of width/height)
        padding_x = int((x_max - x_min) * 0.05)
        padding_y = int((y_max - y_min) * 0.05)
        
        return [
            max(0, x_min - padding_x),
            max(0, y_min - padding_y),
            x_max + padding_x,
            y_max + padding_y
        ]

    def _calculate_confidence(self, paragraph: List[Dict]) -> float:
        """Calculate average confidence for words in paragraph"""
        if not paragraph:
            return 0
        return sum(float(word.get('conf', 0)) for word in paragraph) / len(paragraph)

    def _classify_field(self, text: str) -> Optional[str]:
        """Classify text into field types based on patterns"""
        # Date patterns
        date_patterns = [
            r'\b\d{1,2}[-./]\d{1,2}[-./]\d{2,4}\b',  # DD-MM-YYYY, DD/MM/YYYY, etc.
            r'\b\d{4}[-./]\d{1,2}[-./]\d{1,2}\b',    # YYYY-MM-DD, etc.
            r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b'  # 15 January 2023
        ]
        
        for pattern in date_patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return "Date"
        
        # Document number patterns
        doc_patterns = [
            r'\b[A-Z]{1,3}[-./]\d{3,}[-./]\d{2,4}\b',  # Format like AB/123/2023
            r'\bNr\.?\s*\d{3,}[/-]\d{2,}\b',           # Format like Nr. 12345/22
            r'\b\d{3,}[/-]\d{3,}\b',                    # Format like 12345/678
            r'\bID\s*\d{4,}\b',                         # ID followed by numbers
            r'\bREF\s*[:.]?\s*\d{4,}\b'                 # REF followed by numbers
        ]
        
        for pattern in doc_patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return "Document_Number"
        
        # Email pattern - more comprehensive
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        if re.search(email_pattern, text):
            return "Email"
            
        # Phone pattern - more comprehensive
        phone_patterns = [
            r'\b(?:\+\d{1,3}\s?)?(?:\(\d{1,4}\)|\d{1,4})[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,4}\b',  # Various formats
            r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{3,4}\b',                                                      # XXX-XXX-XXXX
            r'\b\d{2}[-.\s]?\d{3}[-.\s]?\d{2}[-.\s]?\d{2}\b',                                            # XX-XXX-XX-XX
            r'\+\d{1,3}\s?\d{2,3}\s?\d{3}\s?\d{2,4}\b'                                                   # +XX XXX XXX XXX
        ]
        
        for pattern in phone_patterns:
            if re.search(pattern, text):
                return "Phone"
        
        return None

    def _merge_overlapping_annotations(self, annotations: List[Dict]) -> List[Dict]:
        """Merge annotations with significant overlap"""
        if not annotations or len(annotations) <= 1:
            return annotations
            
        # Sort by y-coordinate (top to bottom)
        annotations.sort(key=lambda a: a['bbox'][1])
        
        merged = []
        skip_indices = set()
        
        for i in range(len(annotations)):
            if i in skip_indices:
                continue
                
            current = annotations[i]
            x1, y1, x2, y2 = current['bbox']
            current_area = (x2 - x1) * (y2 - y1)
            
            merged_with_something = False
            
            for j in range(i+1, len(annotations)):
                if j in skip_indices:
                    continue
                    
                other = annotations[j]
                ox1, oy1, ox2, oy2 = other['bbox']
                
                # Calculate overlap area
                overlap_x1 = max(x1, ox1)
                overlap_y1 = max(y1, oy1)
                overlap_x2 = min(x2, ox2)
                overlap_y2 = min(y2, oy2)
                
                if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                    other_area = (ox2 - ox1) * (oy2 - oy1)
                    smaller_area = min(current_area, other_area)
                    
                    # If overlap is significant (>50% of the smaller box)
                    if overlap_area > smaller_area * 0.5:
                        # Merge the boxes
                        merged_x1 = min(x1, ox1)
                        merged_y1 = min(y1, oy1)
                        merged_x2 = max(x2, ox2)
                        merged_y2 = max(y2, oy2)
                        
                        # Decide which field type to keep (prefer known types)
                        field_type = current['field']
                        if field_type == 'Unknown' and other['field'] != 'Unknown':
                            field_type = other['field']
                        
                        # Merge text values
                        value = f"{current['value']} {other['value']}"
                        
                        # Create merged annotation
                        current = {
                            "field": field_type,
                            "value": value,
                            "bbox": [merged_x1, merged_y1, merged_x2, merged_y2],
                            "confidence": max(current.get('confidence', 0), other.get('confidence', 0))
                        }
                        
                        x1, y1, x2, y2 = current['bbox']
                        current_area = (x2 - x1) * (y2 - y1)
                        skip_indices.add(j)
                        merged_with_something = True
            
            merged.append(current)
        
        return merged

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