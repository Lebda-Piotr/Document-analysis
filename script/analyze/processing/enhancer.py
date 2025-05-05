import re
import json
import logging
import os
from tqdm import tqdm
from llama_cpp import Llama
from datetime import datetime
from typing import Dict, List, Optional, Set

class AnnotationProcessor:
    def __init__(self):
        self.logger = self._setup_logger()
        self.llm = None
        self.patterns = {}
        self._initialize_components()

    def _setup_logger(self):
        """Configure logging system"""
        logger = logging.getLogger('AnnotationProcessor')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _initialize_components(self):
        """Initialize components with error handling"""
        try:
            # Verify model file exists
            model_path=r"E:\Nowy folder (2)\praktyki\documentation\script\analyze\processing\models\phi-3-mini-4k-instruct.Q4_K_M.gguf"
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Initialize LLM
            self.llm = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_threads=4,
                verbose=False
            )

            # Field detection patterns
            self.patterns = {
                'Date': [
                    r'\b\d{4}-\d{2}-\d{2}\b',
                    r'\b\d{2}/\d{2}/\d{4}\b',
                    r'\bDatum\b.*\d{4}'
                ],
                'DocumentNumber': [
                    r'\b[A-Z]{2,}/\d{3,}/\d{4}\b',
                    r'\b\d{3,}-\d{3,}\b'
                ],
                'Phone': [
                    r'\b\+\d{2}\s?\d{3}\s?\d{3}\s?\d{3}\b',
                    r'\b\d{3}[-\s]?\d{3}[-\s]?\d{3}\b'
                ],
                'Email': [
                    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                ]
            }

            self.logger.info("Components initialized successfully")

        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            raise

    def process_annotations(self, input_path: str, output_path: str, save_interval: int = 20) -> Dict:
        """Process annotations with auto-saving"""
        processed_data = self._load_existing_results(output_path) or {
            "valid_entries": [],
            "invalid_entries": [],
            "statistics": {"total": 0, "valid": 0, "empty": 0},
            "processed_ids": set()
        }

        try:
            with open(input_path) as f:
                raw_data = json.load(f)

            for doc_id, document in tqdm(raw_data.items(), desc="Processing documents"):
                if doc_id in processed_data["processed_ids"]:
                    continue

                for annotation in document.get('annotations', []):
                    result = self._process_single_annotation(annotation)
                    self._update_statistics(processed_data, result)

                    if result['valid']:
                        processed_data["valid_entries"].append(result)
                    else:
                        processed_data["invalid_entries"].append(result)

                processed_data["processed_ids"].add(doc_id)
                processed_data["statistics"]["total"] += 1

                # Save progress periodically
                if len(processed_data["processed_ids"]) % save_interval == 0:
                    self._save_results(processed_data, output_path)

            # Final save
            self._save_results(processed_data, output_path)
            return processed_data

        except KeyboardInterrupt:
            self.logger.info("Interrupt received. Saving progress...")
            self._save_results(processed_data, output_path)
            return processed_data
        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}")
            return {"status": "error", "message": str(e)}
        finally:
            self._cleanup()

    def _process_single_annotation(self, annotation: Dict) -> Dict:
        """Process a single annotation"""
        try:
            value = annotation.get('value', '').strip()
            
            # Filter empty fields
            if not value:
                return {
                    "original": annotation,
                    "valid": False,
                    "reason": "empty_value"
                }

            # Detect field name
            return {
                "original": annotation,
                "suggested_name": self._detect_field_type(value),
                "valid": True
            }

        except Exception as e:
            self.logger.error(f"Error processing entry: {str(e)}")
            return {
                "original": annotation,
                "valid": False,
                "reason": "processing_error"
            }

    def _detect_field_type(self, text: str) -> str:
        """Detect field name using patterns or AI"""
        # Check regex patterns first
        for field_type, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return field_type

        # Fallback to AI analysis
        return self._analyze_with_ai(text)

    def _analyze_with_ai(self, text: str) -> str:
        """Analyze text using local AI model"""
        prompt = f"""Classify this document text into one of these categories:
        - Date
        - DocumentNumber
        - Phone
        - Email
        - Other
        
        Return ONLY the category name.
        
        Text: "{text[:200]}"
        Category: """

        try:
            response = self.llm(
                prompt,
                max_tokens=20,
                temperature=0.1,
                stop=["\n", "</s>"]
            )
            return response['choices'][0]['text'].strip()
        except Exception as e:
            self.logger.error(f"AI analysis failed: {str(e)}")
            return "Unknown"

    def _update_statistics(self, data: Dict, result: Dict):
        """Update processing statistics"""
        if result['valid']:
            data["statistics"]["valid"] += 1
        else:
            if result.get('reason') == "empty_value":
                data["statistics"]["empty"] += 1

    def _load_existing_results(self, path: str) -> Optional[Dict]:
        """Load existing results if available"""
        try:
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                    data["processed_ids"] = set(data.get("processed_ids", []))
                    return data
        except Exception as e:
            self.logger.warning(f"Couldn't load existing results: {str(e)}")
        return None

    def _save_results(self, data: Dict, path: str):
        """Save results safely"""
        temp_path = f"{path}.tmp"
        try:
            save_data = data.copy()
            save_data["processed_ids"] = list(save_data["processed_ids"])
            
            with open(temp_path, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            os.replace(temp_path, path)
            self.logger.info(f"Saved progress to {path}")

        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _cleanup(self):
        """Clean up resources"""
        if self.llm:
            try:
                self.llm.close()
            except Exception as e:
                self.logger.error(f"Error closing AI model: {str(e)}")
            finally:
                self.llm = None

    def __del__(self):
        self._cleanup()