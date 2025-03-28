import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_image(image_path):
    """Wczytaj obraz i konwertuj z BGR (OpenCV) na RGB (PIL)"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Nie można wczytać obrazu. Sprawdź ścieżkę.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def remove_shadow(img):
    """Usuń cień poprzez normalizację oświetlenia"""
    rgb_planes = cv2.split(img)
    result_planes = []
    for plane in rgb_planes:
        # Rozmycie dla ekstrakcji tła (oświetlenia)
        dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        blurred = cv2.medianBlur(dilated, 21)
        # Różnica między oryginałem a tłem
        diff = 255 - cv2.absdiff(plane, blurred)
        # Normalizacja histogramu
        norm_diff = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        result_planes.append(norm_diff)
    return cv2.merge(result_planes)

def plot_comparison(original, processed, title="Porównanie"):
    """Wizualizacja oryginału vs. przetworzony obraz"""
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Oryginał")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(processed)
    plt.title("Po usunięciu cienia")
    plt.axis('off')
    
    plt.suptitle(title)
    plt.show()

def save_image(image, output_path):
    """Zapisz obraz jako plik JPEG"""
    Image.fromarray(image).save(output_path, quality=95)

if __name__ == "__main__":
    # Ścieżki do plików
    input_path = "doc2.jpg"  
    output_path = "dokument_no_shadow.jpg"
    
    try:
        # 1. Wczytaj obraz
        original_image = load_image(input_path)
        
        # 2. Usuń cień
        processed_image = remove_shadow(original_image)
        
        # 3. Zapisz wynik
        save_image(processed_image, output_path)
        print(f"Zapisano przetworzony obraz jako: {output_path}")
        
        # 4. Wizualizacja
        plot_comparison(original_image, processed_image)
        
    except Exception as e:
        print(f"Błąd: {e}")