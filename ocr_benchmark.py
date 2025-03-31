import cv2
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import ipywidgets as widgets
import pytesseract
from preprocessing import process_document
from document_reader import (
    perform_ocr, 
    detect_document_language, 
    get_ocr_language_code, 
    calculate_ocr_metrics, 
    basic_preprocessing, 
    advanced_preprocessing,
    generate_diff_report
)

def benchmark_ocr_methods(input_path, reference_text=None):
    """
    Kompleksowe porównanie różnych metod OCR z metrykami jakości
    
    Args:
        input_path (str): Ścieżka do pliku obrazu
        reference_text (str): Tekst referencyjny (jeśli dostępny)
        
    Returns:
        pandas.DataFrame: Wyniki benchmarku
    """
    # Wczytaj i przetwórz obraz
    processed_image = process_document(input_path, 'temp_processed.jpg')
    raw_image = cv2.imread(input_path)
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    
    # Wykryj język
    sample_text = perform_ocr(raw_image, preprocess='basic')
    languages = detect_document_language(sample_text[:500])
    lang_code = get_ocr_language_code(languages)
    
    # Metody do przetestowania
    methods = [
        ('Surowy (bez preprocessingu)', raw_image, 'none'),
        ('Surowy (basic preprocessing)', raw_image, 'basic'),
        ('Surowy (advanced preprocessing)', raw_image, 'advanced'),
        ('Przetworzony (bez preprocessingu)', processed_image, 'none'),
        ('Przetworzony (basic preprocessing)', processed_image, 'basic'),
        ('Przetworzony (advanced preprocessing)', processed_image, 'advanced')
    ]
    
    # Przeprowadź testy
    results = []
    for name, image, preprocess in methods:
        text = perform_ocr(image, preprocess=preprocess, lang=lang_code)
        
        if reference_text:
            metrics = calculate_ocr_metrics(reference_text, text)
        else:
            metrics = {'text_length': len(text)}
        
        results.append({
            'method': name,
            'text': text,
            **metrics
        })
    
    # Generuj raport
    df = pd.DataFrame(results)
    report_file = "ocr_benchmark_report.xlsx"
    
    with pd.ExcelWriter(report_file) as writer:
        df.to_excel(writer, sheet_name='Podsumowanie', index=False)
        
        # Dodatkowy arkusz ze szczegółami
        details = []
        for _, row in df.iterrows():
            details.append({
                'Metoda': row['method'],
                'Długość tekstu': len(row['text']),
                'Liczba linii': row['text'].count('\n') + 1,
                'Liczba słów': len(row['text'].split())
            })
        
        pd.DataFrame(details).to_excel(
            writer, 
            sheet_name='Statystyki', 
            index=False
        )
    
    print(f"\nZakończono testy. Wyniki zapisano w: {report_file}")
    
    # Generuj raport różnic jeśli dostępny tekst referencyjny
    if reference_text:
        best_method = df.loc[df['similarity_ratio'].idxmax()]
        generate_diff_report(
            reference_text, 
            best_method['text'],
            "best_method_diff.html"
        )
    
    return df

def interactive_ocr_test(image_path):
    """
    Interaktywne testowanie różnych parametrów OCR
    
    Args:
        image_path (str): Ścieżka do pliku obrazu
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Widgety do kontroli parametrów
    preprocess_select = widgets.Dropdown(
        options=['none', 'basic', 'advanced'],
        value='basic',
        description='Preprocessing:'
    )
    
    language_select = widgets.Dropdown(
        options=['pol', 'eng', 'pol+eng', 'deu', 'fra'],
        value='pol',
        description='Język:'
    )
    
    threshold_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=255,
        description='Próg binaryzacji:'
    )
    
    output = widgets.Output()
    
    def update_ocr(change):
        with output:
            output.clear_output()
            
            # Pobierz aktualne parametry
            preprocess = preprocess_select.value
            lang = language_select.value
            
            # Wykonaj OCR
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            if threshold_slider.value > 0:
                _, gray = cv2.threshold(gray, threshold_slider.value, 255, cv2.THRESH_BINARY)
            
            if preprocess == 'basic':
                gray = basic_preprocessing(gray)
            elif preprocess == 'advanced':
                gray = advanced_preprocessing(gray)
            
            # Wyświetl przetworzony obraz
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(gray, cmap='gray')
            plt.title("Przetworzony obraz")
            plt.axis('off')
            
            # Wykonaj OCR
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(gray, config=custom_config, lang=lang)
            
            # Wyświetl wynik
            plt.subplot(1, 2, 2)
            plt.text(0, 0.5, text, fontsize=10, wrap=True)
            plt.axis('off')
            plt.show()
            
            print(f"Rozpoznany tekst ({len(text)} znaków):\n{text[:500]}...")
    
    # Połącz widgety z funkcją aktualizacji
    for widget in [preprocess_select, language_select, threshold_slider]:
        widget.observe(update_ocr, names='value')
    
    # Wywołaj początkową aktualizację
    update_ocr(None)
    
    # Wyświetl interfejs
    display(widgets.VBox([
        preprocess_select,
        language_select,
        threshold_slider,
        output
    ]))

if __name__ == "__main__":
    input_path = "doc2.jpg"
    benchmark_ocr_methods(input_path)
    # interactive_ocr_test(input_path)  # Odkomentuj, aby uruchomić wersję interaktywną