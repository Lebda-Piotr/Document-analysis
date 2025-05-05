import json
from tqdm import tqdm
from transformers import pipeline
from llama_cpp import Llama

class AnnotationEnhancer:
    def __init__(self):
        # CPU-optimized models
        self.translator = pipeline(
            "translation",
            model="Helsinki-NLP/opus-mt-mul-pl",
            device=-1  # Force CPU
        )
        
        # Phi-3 Mini 4K Instruct (3.8B parameters, 2GB RAM usage)
        self.llm = Llama(
            model_path="phi-3-mini-4k-instruct.Q4_K_M.gguf",
            n_ctx=2048,
            n_threads=4,
            verbose=False
        )

    def enhance_entry(self, entry):
        # Extract text from annotation
        text = entry.get('value', '')
        
        # Step 1: Auto-generate field name
        field_name = self._generate_field_name(text)
        
        # Step 2: Translate text to Polish
        translated_text = self._translate_text(text)
        
        return {
            "original_field": entry,
            "suggested_name": field_name,
            "translated_text": translated_text,
            "confidence": 0.8  # Placeholder for confidence score
        }

    def _generate_field_name(self, text):
        prompt = f"""Analyze this document text and suggest an appropriate field name in English.
        Return ONLY the field name without explanations.
        Examples:
        Text: "2023-01-01" → Date
        Text: "user@example.com" → Email
        Text: "{text}" → """
        
        response = self.llm(
            prompt,
            max_tokens=20,
            temperature=0.1,
            stop=["\n"]
        )
        return response['choices'][0]['text'].strip()

    def _translate_text(self, text):
        if len(text) < 3:  # Skip short texts
            return text
        return self.translator(text, max_length=400)[0]['translation_text']

def process_annotations(input_path, output_path):
    enhancer = AnnotationEnhancer()
    
    with open(input_path) as f:
        data = json.load(f)
    
    enhanced = []
    for key in tqdm(data.keys()):
        enhanced.append(enhancer.enhance_entry(data[key]))
        # Save progress every 100 entries
        if len(enhanced) % 100 == 0:
            with open(output_path, 'w') as f:
                json.dump(enhanced, f, indent=2)
    
    return enhanced