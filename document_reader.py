import cv2
import numpy as np
import pytesseract
from langdetect import detect_langs
import Levenshtein
import difflib

def basic_preprocessing(gray):
    """Podstawowy preprocessing obrazu"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    gray = cv2.medianBlur(gray, 3)
    return gray

def advanced_preprocessing(gray):
    """Zaawansowany preprocessing obrazu"""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    gray = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    kernel = np.ones((2,2), np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    return gray

def perform_ocr(image, preprocess='none', lang='pol'):
    """
    Uproszczona funkcja OCR skupiająca się na najlepszych metodach
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Tylko podstawowe opcje preprocessingu
    if preprocess == 'light':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        gray = cv2.medianBlur(gray, 1)
    elif preprocess == 'basic':
        gray = basic_preprocessing(gray)
    elif preprocess == 'advanced':
        gray = advanced_preprocessing(gray)
    
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(gray, config=custom_config, lang=lang)
    return text

def detect_document_language(text_sample):
    """
    Wykryj język tekstu z dokumentu
    
    Args:
        text_sample (str): Próbka tekstu do analizy
        
    Returns:
        list: Lista możliwych języków z prawdopodobieństwami
    """
    try:
        languages = detect_langs(text_sample)
        return languages
    except:
        return [{'lang': 'pl', 'prob': 1.0}]

def get_ocr_language_code(languages):
    """
    Konwertuj wykryte języki na kody językowe Tesseract
    
    Args:
        languages (list): Lista języków z detect_langs
        
    Returns:
        str: String z kodami języków dla Tesseract
    """
    lang_map = {
        'pl': 'pol',
        'en': 'eng',
        'de': 'deu',
        'fr': 'fra',
        'es': 'spa'
    }
    
    tesseract_langs = []
    for lang in languages:
        code = lang.lang
        if code in lang_map:
            tesseract_langs.append(lang_map[code])
    
    if not tesseract_langs:
        return 'pol+eng'
    
    return '+'.join(tesseract_langs[:2])

def calculate_ocr_metrics(original, ocr):
    """
    Oblicz szczegółowe metryki jakości OCR
    
    Args:
        original (str): Tekst oryginalny/referencyjny
        ocr (str): Tekst rozpoznany przez OCR
        
    Returns:
        dict: Słownik z metrykami jakości
    """
    word_diff = difflib.SequenceMatcher(None, original, ocr)
    
    return {
        'levenshtein_distance': Levenshtein.distance(original, ocr),
        'similarity_ratio': word_diff.ratio(),
        'word_accuracy': calculate_word_accuracy(original, ocr),
        'character_accuracy': calculate_character_accuracy(original, ocr),
        'error_rate': calculate_error_rate(original, ocr)
    }

def calculate_word_accuracy(original, ocr):
    """Oblicz dokładność na poziomie słów"""
    original_words = original.split()
    ocr_words = ocr.split()
    correct = sum(1 for ow, cw in zip(original_words, ocr_words) if ow == cw)
    return correct / max(len(original_words), 1)

def calculate_character_accuracy(original, ocr):
    """Oblicz dokładność na poziomie znaków"""
    matches = sum(1 for o, c in zip(original, ocr) if o == c)
    return matches / max(len(original), 1)

def calculate_error_rate(original, ocr):
    """Oblicz współczynnik błędów"""
    return Levenshtein.distance(original, ocr) / max(len(original), 1)

def generate_diff_report(original, ocr, output_file="diff_report.html"):
    """
    Generuj raport różnic w formacie HTML
    
    Args:
        original (str): Tekst oryginalny
        ocr (str): Tekst z OCR
        output_file (str): Ścieżka do pliku wyjściowego
    """
    differ = difflib.HtmlDiff()
    html = differ.make_file(
        original.splitlines(), 
        ocr.splitlines(),
        fromdesc="Oryginał",
        todesc="OCR"
    )
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wygenerowano raport różnic: {output_file}")

def compare_ocr_results(input_path, output_file="ocr_results_simple.txt"):
    """
    Uproszczona funkcja porównująca tylko najlepsze metody
    """
    from preprocessing import process_document
    
    processed_image = process_document(input_path, 'temp_processed.jpg', simple_preprocess=True)
    raw_image = cv2.imread(input_path)
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    
    # Wykryj język
    sample_text = perform_ocr(raw_image, preprocess='none')
    languages = detect_document_language(sample_text[:500])
    lang_code = get_ocr_language_code(languages)
    
    # Testuj tylko najlepsze metody
    results = {
        "Przetworzony obraz (bez preprocessingu)": perform_ocr(processed_image, preprocess='none', lang=lang_code),
        "Przetworzony obraz (lekki preprocessing)": perform_ocr(processed_image, preprocess='light', lang=lang_code),
    }
    
    # Zapisz wyniki
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Użyte języki Tesseract: {lang_code}\n\n")
        for method, text in results.items():
            f.write(f"=== {method.upper()} ===\n")
            f.write(text + "\n\n")
    
    print(f"Zapisano uproszczone wyniki OCR do: {output_file}")

if __name__ == "__main__":
    input_path = "doc2.jpg"
    compare_ocr_results(input_path)