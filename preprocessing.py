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

def improved_edge_detection(img):
    """Zaawansowane wykrywanie krawędzi dokumentu"""
    # Konwersja do skali szarości
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Redukcja szumów
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Adaptacyjne progowanie
    thresh = cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Algorytm Canny z adaptacyjnymi progami
    edges = cv2.Canny(
        thresh, 
        threshold1=cv2.mean(gray)[0] * 0.5, 
        threshold2=cv2.mean(gray)[0] * 1.5
    )
    
    # Dylatacja krawędzi dla połączenia przerw
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    return edges

def advanced_document_contour(edges):
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
        
        # Kryteria dla dokumentu:
        # - około 4 wierzchołki
        # - znaczący obszar
        # - prawie prostokątny
        if (len(approx) == 4 and 
            area > 1000 and  # minimalna wielkość
            cv2.isContourConvex(approx)):
            document_contours.append(approx)
    
    # Wybierz kontur najbardziej przypominający prostokąt
    if document_contours:
        return max(document_contours, key=cv2.contourArea)
    
    return None

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

def rotate_image(img, angle):
    """Obróć obraz o podany kąt (z wypełnieniem tła)"""
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return rotated

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

def save_image(image, output_path):
    """Zapisz obraz jako plik JPEG"""
    Image.fromarray(image).save(output_path, quality=95)

if __name__ == "__main__":
    # Ścieżki do plików
    input_path = "doc1.png"
    output_path = "dokument_processed.jpg"
    
    try:
        # 1. Wczytaj obraz
        original_image = load_image(input_path)
        
        # 2. Usuń cień
        no_shadow_image = remove_shadow(original_image)
        
        # 3. Wykryj krawędzie i kontur dokumentu (ULEPSZONE)
        edged_image = improved_edge_detection(no_shadow_image)
        contour = advanced_document_contour(edged_image)
        
        if contour is not None:
            # 4. Korekta perspektywy (jeśli znaleziono kontur)
            warped_image = correct_perspective(no_shadow_image, contour)
            
            # 5. Obrót o mały kąt (opcjonalnie, jeśli dokument jest przechylony)
            final_image = rotate_image(warped_image, angle=0)  # Zmień angle jeśli potrzeba
        else:
            print("Nie znaleziono konturu dokumentu. Pomijam korektę perspektywy.")
            final_image = no_shadow_image
        
        # 6. Zapisz wynik
        save_image(final_image, output_path)
        print(f"Zapisano przetworzony obraz jako: {output_path}")
        
        # 7. Wizualizacja
        plot_comparison(original_image, final_image)
        
    except Exception as e:
        print(f"Błąd: {e}")