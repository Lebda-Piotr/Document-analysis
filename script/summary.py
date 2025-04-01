import os
import re
from datetime import datetime
from typing import Dict, Optional
import google.generativeai as genai  # Poprawiony import

class GeminiDocumentSummarizer:
    """Proper implementation for Gemini API 0.3.0+"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Missing Gemini API key. Set GEMINI_API_KEY environment variable.")
        
        # Initialize the client
        genai.configure(api_key=self.api_key)  # Konfiguracja API
        self.client = genai
        
        # Specify the model name
        self.model_name = 'gemini-2.0-flash-thinking-exp-01-21'
    
    def generate_summary(self, raw_text: str, metadata: Dict) -> Dict:
        """Generates document summary using Gemini"""
        result = {'summary': None, 'filepath': None}
        
        if not raw_text.strip():
            return result
            
        try:
            prompt = self._create_prompt(raw_text, metadata)
            response = self.client.generate_content(
                model=self.model_name,
                contents=prompt
            )
            
            if response.text:
                result['summary'] = self._clean_response(response.text)
                result['filepath'] = self._save_to_file(result['summary'])
                
        except Exception as e:
            print(f"Gemini API Error: {str(e)}")
        
        return result
    
    def _create_prompt(self, text: str, meta: Dict) -> str:
        """Creates optimized prompt for document analysis"""
        return f"""Analyze this document and provide summary in Polish:
        
Document metadata:
- Dates: {', '.join(meta.get('dates', [])) or 'None'}
- Numbers: {', '.join(meta.get('document_numbers', [])) or 'None'}

Document content (first 28K chars):
{text[:28000]}

Provide summary in this format:
Type: [document type]
Parties: [issuer] → [receiver]
Key dates: [list]
Amounts: [detected]
Summary: [2-3 sentences in Polish]"""
    
    def _clean_response(self, text: str) -> str:
        """Cleans model output"""
        return re.sub(r'\*+', '', text).strip()
    
    def _save_to_file(self, content: str) -> str:
        """Saves analysis to timestamped file"""
        os.makedirs('data/output/analysis', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"data/output/analysis/gemini_{timestamp}.txt"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"DOCUMENT ANALYSIS\n{'='*40}\n\n{content}")
        
        return filepath

# Safe initialization
try:
    summarizer = GeminiDocumentSummarizer()
except Exception as e:
    print(f"Gemini initialization failed: {e}")
    summarizer = None

def get_document_summary(raw_text: str, metadata: Dict) -> Dict:
    return summarizer.generate_summary(raw_text, metadata) if summarizer else {
        'summary': "Analysis unavailable - Gemini not configured",
        'filepath': None
    }