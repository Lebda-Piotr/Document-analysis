import cv2
import pytesseract
import re
import json
import os
from typing import List, Dict, Tuple
from datetime import datetime
from preprocessing import process_document
# from symspellpy import SymSpell, Verbosity  # Zakomentowane - nieużywane w tej wersji
from langdetect import detect, LangDetectException
from summary import get_document_summary

"""
# Zakomentowana sekcja słowników - zachowana na przyszłość

# Inicjalizacja słowników
sym_spell_pl = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
sym_spell_en = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

# Ścieżki do słowników
DICT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'dictionaries')

# Ładowanie słowników
def load_dictionaries():
    ""Ładuje słowniki dla obu języków""
    try:
        # Słownik polski
        pl_dict_path = os.path.join(DICT_DIR, 'pl_dict.txt')
        sym_spell_pl.load_dictionary(pl_dict_path, term_index=0, count_index=1, encoding="utf-8")
    except (FileNotFoundError, UnicodeDecodeError) as e:
        print(f"Błąd ładowania słownika polskiego: {e}")
        # Rozszerzony fallback dla polskiego ze znakami diakrytycznymi
        pl_fallback = {
            "wrocław": 1000, "lipca": 1000, "adres": 1000, "dokumentu": 1000,
            "zażółć": 800, "gęślą": 800, "jaźń": 800, "żółw": 900, 
            "miesiąc": 1000, "piątek": 1000, "środa": 1000, "firma": 1000,
            "faktura": 1000, "płatność": 900, "należność": 900, "ulica": 950
        }
        fallback_path = os.path.join(DICT_DIR, 'pl_fallback.txt')
        with open(fallback_path, "w", encoding="utf-8") as f:
            for word, freq in pl_fallback.items():
                f.write(f"{word} {freq}\n")
        sym_spell_pl.load_dictionary(fallback_path, term_index=0, count_index=1, encoding="utf-8")

    try:
        # Słownik angielski
        en_dict_path = os.path.join(DICT_DIR, 'en-80k.txt')
        sym_spell_en.load_dictionary(en_dict_path, term_index=0, count_index=1, encoding="utf-8")
    except (FileNotFoundError, UnicodeDecodeError) as e:
        print(f"Błąd ładowania słownika angielskiego: {e}")
        # Fallback dla angielskiego
        en_fallback = {"address":1000, "document":1000, "invoice":1000, "payment":950, "date":950}
        fallback_path = os.path.join(DICT_DIR, 'en_fallback.txt')
        with open(fallback_path, "w", encoding="utf-8") as f:
            for word, freq in en_fallback.items():
                f.write(f"{word} {freq}\n")
        sym_spell_en.load_dictionary(fallback_path, term_index=0, count_index=1, encoding="utf-8")

load_dictionaries()
"""

def detect_language(text: str) -> str:
    """Wykrywa język tekstu"""
    try:
        lang = detect(text)
        return 'pl' if lang == 'pl' else 'en'
    except LangDetectException:
        return 'pl'  # Domyślnie polski

"""
# Zakomentowana funkcja autokorekty - zachowana na przyszłość
def correct_spelling(text: str, language: str) -> str:
    ""
    Poprawia literówki w tekście
    Args:
        text: Tekst do poprawienia
        language: 'pl' lub 'en'
    Returns:
        Tekst po autokorekcie
    ""
    sym_spell = sym_spell_pl if language == 'pl' else sym_spell_en
    lines = text.split('\n')
    corrected_lines = []
    
    for line in lines:
        words = line.split()
        corrected_words = []
        
        for word in words:
            # Pomijaj liczby, daty, numery dokumentów i specjalne znaki
            if (re.match(r'^[\W\d_]+$', word) or 
                re.match(r'\b\d{2}-\d{2}-\d{4}\b', word) or
                re.match(r'\b[A-Z]{2,}/\d{3,}/\d{4}\b', word)):
                corrected_words.append(word)
                continue
            
            # Popraw słowo
            suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
            if suggestions:
                corrected_word = suggestions[0].term
                # Zachowaj wielkość liter oryginału
                if word.istitle():
                    corrected_word = corrected_word.title()
                elif word.isupper():
                    corrected_word = corrected_word.upper()
                corrected_words.append(corrected_word)
            else:
                corrected_words.append(word)
        
        corrected_lines.append(' '.join(corrected_words))
    
    return '\n'.join(corrected_lines)
"""

def basic_preprocessing(gray):
    """Podstawowy preprocessing obrazu"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    gray = cv2.medianBlur(gray, 3)
    return gray

def perform_ocr(image, preprocess='none', lang='pol+eng'):
    """
    Wykonaj OCR na obrazie
    Args:
        image: Obraz do analizy
        preprocess: 'none', 'light' lub 'basic'
        lang: język dla Tesseract
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
    
    # Wykryj język (bez autokorekty)
    detected_lang = detect_language(raw_text)
    
    # Zwróć tylko surowy tekst i język (bez poprawionego tekstu)
    return raw_text, detected_lang

def extract_dates(text: str) -> List[str]:
    """Znajdź daty w różnych formatach"""
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
    """Znajdź numery dokumentów"""
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
    """Znajdź adresy email"""
    return list(set(re.findall(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        text,
        flags=re.IGNORECASE
    )))

def extract_phones(text: str) -> List[str]:
    """Znajdź numery telefonów"""
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
    """Główna funkcja ekstrakcji metadanych"""
    return {
        'dates': extract_dates(text),
        'document_numbers': extract_document_numbers(text),
        'emails': extract_emails(text),
        'phones': extract_phones(text)
    }

def save_metadata(data: Dict, filename: str = "metadata.json"):
    """Zapisz wyniki do JSON"""
    output_path = os.path.join('data', 'output', filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Zapisano metadane do {output_path}")

def save_text_comparison(raw_text: str, filename: str = "ocr_output.txt"):
    """Zapisz wynik OCR do pliku TXT"""
    output_path = os.path.join('data', 'output', filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("WYNIK OCR:\n")
        f.write("=" * 80 + "\n\n")
        f.write(raw_text)
    
    print(f"Zapisano wynik OCR do {output_path}")

def print_metadata(metadata: Dict):
    """Wypisz metadane w czytelnej formie"""
    print("\nWyodrębnione metadane:")
    print(f"Daty: {', '.join(metadata.get('dates', []))}")
    print(f"Numery dokumentów: {', '.join(metadata.get('document_numbers', []))}")
    print(f"Emails: {', '.join(metadata.get('emails', []))}")
    print(f"Telefony: {', '.join(metadata.get('phones', []))}")

def process_document_with_metadata(input_path: str):
    """
    Pełne przetwarzanie dokumentu z ekstrakcją metadanych
    """
    # 1. Przetwórz obraz
    processed_image = process_document(input_path, os.path.join('data', 'output', 'temp_processed.jpg'))
    
    # 2. Wykonaj OCR (bez autokorekty)
    raw_text, language = perform_ocr(processed_image)
    
    # 3. Zapisz wynik OCR do pliku
    save_text_comparison(raw_text)
    
    # Wyświetl informację na konsoli
    print(f"\nWykryty język: {'polski' if language == 'pl' else 'angielski'}")
    print("Wynik OCR zapisano do pliku ocr_output.txt")
    
    # 4. Wyodrębnij metadane
    metadata = extract_metadata(raw_text)
    result = {
        'language': language,
        'raw_text': raw_text,
        'metadata': metadata
    }
    
    # 5. Zapisz i wyświetl wyniki
    save_metadata(result)
    print_metadata(metadata)
    
    # 6. Analiza z modelem językowym
    if len(raw_text.strip()) > 50:
        analysis_result = get_document_summary(raw_text, metadata)
        
        if analysis_result.get('summary'):
            result['llm_analysis'] = analysis_result['summary']
            
            if analysis_result.get('filepath'):
                print(f"\nZapisano analizę do pliku: {analysis_result['filepath']}")
    return result

if __name__ == "__main__":
    input_path = os.path.join('data', 'input', 'doc2.jpg')
    process_document_with_metadata(input_path)