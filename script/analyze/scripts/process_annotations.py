import sys
from pathlib import Path

current_dir = Path(__file__).parent  # Scripts directory
parent_dir = current_dir.parent      # Analyze directory
sys.path.append(str(parent_dir))

import logging
from processing.enhancer import AnnotationProcessor

processor = AnnotationProcessor()
result = processor.process_annotations(
    input_path='data/annotations.json',
    output_path='processed_annotations.json'
)

print(f"Processed {result['statistics']['total']} entries")
print(f"Valid entries: {result['statistics']['valid']}")
print(f"Empty entries: {result['statistics']['empty']}")
