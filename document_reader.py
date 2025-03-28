import cv2
import pytesseract
from preprocessing import process_document

def perform_ocr(image, preprocess=False):
    """
    Wykonaj OCR na obrazie
    
    Args:
        image (numpy.ndarray): Obraz do odczytu
        preprocess (bool): Czy użyć preprocessingu
    
    Returns:
        str: Rozpoznany tekst
    """
    # Konwersja do skali szarości
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Opcjonalny preprocessing
    if preprocess:
        # Progowanie Otsu
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Redukcja szumów
        gray = cv2.medianBlur(gray, 3)
    
    # Konfiguracja Tesseract
    custom_config = r'--oem 3 --psm 6'
    
    # Wykonaj OCR
    text = pytesseract.image_to_string(gray, config=custom_config, lang='pol')
    
    return text

def compare_ocr_results(input_path, output_file="ocr_results.txt"):
    """
    Porównaj wyniki OCR dla surowego i przetworzonego obrazu i zapisz do pliku
    
    Args:
        input_path (str): Ścieżka do pliku wejściowego
        output_file (str): Ścieżka do pliku wyjściowego
    """
    # Wczytaj i przetwórz obraz
    processed_image = process_document(input_path, 'temp_processed.jpg')
    
    # Wczytaj surowy obraz
    raw_image = cv2.imread(input_path)
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    
    # OCR na surowym obrazie
    raw_text_default = perform_ocr(raw_image, preprocess=False)
    raw_text_preprocessed = perform_ocr(raw_image, preprocess=True)
    
    # OCR na przetworzonym obrazie
    processed_text_default = perform_ocr(processed_image, preprocess=False)
    processed_text_preprocessed = perform_ocr(processed_image, preprocess=True)
    
    # Zapisz wyniki do pliku
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("--- OCR NA SUROWYM OBRAZIE ---\n")
        f.write("\n=== SUROWY OBRAZ (domyślny) ===\n")
        f.write(raw_text_default + "\n")
        
        f.write("\n=== SUROWY OBRAZ (preprocessowany) ===\n")
        f.write(raw_text_preprocessed + "\n")
        
        f.write("\n--- OCR NA PRZETWORZONYM OBRAZIE ---\n")
        f.write("\n=== PRZETWORZONY OBRAZ (domyślny) ===\n")
        f.write(processed_text_default + "\n")
        
        f.write("\n=== PRZETWORZONY OBRAZ (preprocessowany) ===\n")
        f.write(processed_text_preprocessed + "\n")

if __name__ == "__main__":
    input_path = "doc11.jpg"
    compare_ocr_results(input_path)