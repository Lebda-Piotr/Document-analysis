import os
import re
from datetime import datetime
from typing import Dict, Optional
import google.generativeai as genai

class GeminiDocumentSummarizer:
    """Implementacja dla Gemini API w wersji 0.3.0+"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Brak klucza API Gemini. Ustaw zmienną środowiskową GEMINI_API_KEY.")
        
        # Inicjalizacja klienta
        genai.configure(api_key=self.api_key)
        
        # Pobranie modelu
        self.model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp-01-21')
    
    def generate_summary(self, raw_text: str, metadata: Dict) -> Dict:
        """Generuje podsumowanie dokumentu przy użyciu Gemini"""
        result = {'summary': None, 'filepath': None}
        
        if not raw_text.strip():
            return result
            
        try:
            prompt = self._create_prompt(raw_text, metadata)
            response = self.model.generate_content(prompt)
            
            if response.text:
                result['summary'] = self._clean_response(response.text)
                result['filepath'] = self._save_to_file(result['summary'])
                
        except Exception as e:
            print(f"Błąd API Gemini: {str(e)}")
        
        return result
    
    def _create_prompt(self, text: str, meta: Dict) -> str:
        """Tworzy zoptymalizowany prompt do analizy dokumentu"""
        return f"""Przeanalizuj ten dokument i przygotuj podsumowanie w języku polskim:
        
Metadane dokumentu:
- Daty: {', '.join(meta.get('dates', [])) or 'Brak'}
- Numery: {', '.join(meta.get('document_numbers', [])) or 'Brak'}

Treść dokumentu (pierwsze 28K znaków):
{text[:28000]}

Przygotuj podsumowanie w następującym formacie:
Typ: [typ dokumentu]
Strony: [nadawca] → [odbiorca]
Kluczowe daty: [lista]
Kwoty: [wykryte]
Podsumowanie: [2-3 zdania po polsku]"""
    
    def _clean_response(self, text: str) -> str:
        """Czyści odpowiedź modelu"""
        return re.sub(r'\*+', '', text).strip()
    
    def _save_to_file(self, content: str) -> str:
        """Zapisuje analizę do pliku z sygnaturą czasową"""
        os.makedirs('data/output/analysis', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"data/output/analysis/gemini_{timestamp}.txt"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"ANALIZA DOKUMENTU\n{'='*40}\n\n{content}")
        
        return filepath

# Bezpieczna inicjalizacja
try:
    summarizer = GeminiDocumentSummarizer()
except Exception as e:
    print(f"Błąd inicjalizacji Gemini: {e}")
    summarizer = None

def get_document_summary(raw_text: str, metadata: Dict) -> Dict:
    return summarizer.generate_summary(raw_text, metadata) if summarizer else {
        'summary': "Analiza niedostępna - Gemini nie skonfigurowany",
        'filepath': None
    }