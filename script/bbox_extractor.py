import pytesseract
from PIL import Image
import json
from typing import List, Dict

class BoundingBoxExtractor:
    @staticmethod
    def get_bboxes(image_path: str) -> List[Dict]:
        """Zwraca listę bounding boxów i tekstu z obrazu."""
        image = Image.open(image_path)
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        bboxes = []
        for i in range(len(data['text'])):
            if data['text'][i].strip():  # Pomijanie pustych
                bboxes.append({
                    "text": data['text'][i],
                    "x": data['left'][i],
                    "y": data['top'][i],
                    "width": data['width'][i],
                    "height": data['height'][i],
                    "page_num": int(image_path.stem.rsplit('_page', 1)[-1]) if '_page' in image_path.stem else 1
                })
        return bboxes

    @staticmethod
    def match_entities_with_bboxes(entities: Dict, bboxes: List[Dict]) -> Dict:
        """Łączy encje z ich bounding boxami."""
        annotated_entities = {}
        for entity_type, values in entities.items():
            annotated_entities[entity_type] = []
            for value in values:
                for bbox in bboxes:
                    if value in bbox['text']:
                        annotated_entities[entity_type].append({
                            "value": value,
                            "bbox": [bbox['x'], bbox['y'], bbox['width'], bbox['height']],
                            "page": bbox['page_num']
                        })
                        break
        return annotated_entities