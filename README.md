# System Analizy Dokumentów z Wykorzystaniem AI

## Opis projektu

System do przetwarzania skanów dokumentów z funkcjami:
- Korekcja perspektywy i poprawa jakości obrazu
- Ekstrakcja tekstu (OCR) z wykorzystaniem Tesseract
- Analiza zawartości przez model Gemini AI
- Generowanie czytelnych podsumowań

## Wymagania

- Python 3.8+
- Tesseract OCR
- Konto Google Cloud (Gemini API)

## Instalacja

1. Zainstaluj zależności:
```bash
pip install opencv-python pytesseract pillow google-generativeai langdetect numpy matplotlib pandas ipywidgets
```

2. Zainstaluj Tesseract OCR:

Windows: Pobierz z UB Mannheim

macOS: 
```bash 
brew install tesseract tesseract-lang
```

Linux: 
```bash
sudo apt install tesseract-ocr libtesseract-dev tesseract-ocr-pol
```

3. Skonfiguruj klucz Gemini API:
```bash
export GEMINI_API_KEY="twój_klucz_api"
```

## Uruchomienie

```bash
# Wersja z interfejsem graficznym (zalecana)
python gui_app.py

# Wersja konsolowa (używa domyślnej ścieżki do pliku)
python document_reader.py
```

## Instrukcja użycia GUI

1. Kliknij "Przeglądaj..." i wybierz dokument (JPG/PNG)

2. Kliknij "Analizuj dokument"

3. Wynik zostanie zapisany w data/output/analysis/ i automatycznie otwarty

## Licensja

MIT license