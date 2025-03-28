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
        dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        blurred = cv2.medianBlur(dilated, 21)
        diff = 255 - cv2.absdiff(plane, blurred)
        norm_diff = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        result_planes.append(norm_diff)
    return cv2.merge(result_planes)

def improved_edge_detection(img, use_alternative=False):
    """Zaawansowane wykrywanie krawędzi dokumentu"""
    # Konwersja do skali szarości
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Redukcja szumów
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    
    if not use_alternative:
        # Standardowe podejście
        thresh = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        edges = cv2.Canny(
            thresh, 
            threshold1=cv2.mean(gray)[0] * 0.5, 
            threshold2=cv2.mean(gray)[0] * 1.5
        )
    else:
        # Alternatywne podejście dla trudnych przypadków
        # Wyrównanie histogramu
        gray = cv2.equalizeHist(gray)
        
        # Progowanie Otsu
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        edges = cv2.Canny(gray, 10, 250)
    
    # Dylatacja krawędzi dla połączenia przerw
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    
    return edges

def advanced_document_contour(edges, img_area):
    """Zaawansowane znajdowanie konturu dokumentu"""
    # Znajdź kontury
    contours, _ = cv2.findContours(
        edges, 
        cv2.RETR_LIST, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Filtrowanie konturów
    document_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        # Zaawansowane kryteria dla dokumentu:
        # - około 4 wierzchołki
        # - znaczący obszar (nie za mały, nie za duży)
        # - prawie prostokątny
        if (len(approx) == 4 and 
            area > img_area * 0.1 and  # min 10% obszaru obrazu
            area < img_area * 0.95 and  # max 95% obszaru obrazu
            cv2.isContourConvex(approx)):
            document_contours.append(approx)
    
    # Wybierz kontur najbardziej przypominający prostokąt
    if document_contours:
        return max(document_contours, key=cv2.contourArea)
    
    return None

def fallback_full_image_processing(img):
    """Przetwarzanie gdy nie znaleziono konturu - pełny obraz"""
    # Konwersja do skali szarości
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Wyrównanie histogramu
    gray = cv2.equalizeHist(gray)
    
    # Binaryzacja Otsu
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Znajdź marginesy
    rows = np.where(~np.all(binary == 255, axis=1))[0]
    cols = np.where(~np.all(binary == 255, axis=0))[0]
    
    if len(rows) > 0 and len(cols) > 0:
        # Przytnij do granic dokumentu
        cropped = img[rows.min():rows.max()+1, cols.min():cols.max()+1]
        return cropped
    
    return img

def correct_perspective(img, contour):
    """Skoryguj perspektywę dokumentu (prostowanie)"""
    # Punkty narożników dokumentu
    pts = contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    
    # Porządkowanie punktów: [lewy-górny, prawy-górny, prawy-dolny, lewy-dolny]
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    (tl, tr, br, bl) = rect
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (int(width), int(height)))
    return warped

def plot_comparison(original, processed, title="Porównanie"):
    """Wizualizacja oryginału vs. przetworzony obraz"""
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Oryginał")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(processed)
    plt.title("Przetworzony")
    plt.axis('off')
    
    plt.suptitle(title)
    plt.show()

def process_document(input_path, output_path):
    """Główna funkcja przetwarzania dokumentu"""
    try:
        # 1. Wczytaj obraz
        original_image = load_image(input_path)
        
        # Oblicz całkowity obszar obrazu
        img_area = original_image.shape[0] * original_image.shape[1]
        
        # 2. Usuń cień
        no_shadow_image = remove_shadow(original_image)
        
        # 3. Wykryj krawędzie i kontur dokumentu
        edged_image = improved_edge_detection(no_shadow_image)
        contour = advanced_document_contour(edged_image, img_area)
        
        if contour is not None:
            # 4. Korekta perspektywy (jeśli znaleziono kontur)
            warped_image = correct_perspective(no_shadow_image, contour)
            final_image = warped_image
        else:
            # Próba alternatywnego wykrywania krawędzi
            edged_image_alt = improved_edge_detection(no_shadow_image, use_alternative=True)
            contour_alt = advanced_document_contour(edged_image_alt, img_area)
            
            if contour_alt is not None:
                warped_image = correct_perspective(no_shadow_image, contour_alt)
                final_image = warped_image
            else:
                # Fallback - przetwarzanie całego obrazu
                print("Nie znaleziono konturu dokumentu. Stosuję przetwarzanie pełnego obrazu.")
                final_image = fallback_full_image_processing(no_shadow_image)
        
        # Zapisz i zwizualizuj wynik
        Image.fromarray(final_image).save(output_path, quality=95)
        print(f"Zapisano przetworzony obraz jako: {output_path}")
        plot_comparison(original_image, final_image)
        
        return final_image
    
    except Exception as e:
        print(f"Błąd: {e}")
        return None

if __name__ == "__main__":
    input_path = "doc2.jpg"
    output_path = "dokument_processed.jpg"
    process_document(input_path, output_path)