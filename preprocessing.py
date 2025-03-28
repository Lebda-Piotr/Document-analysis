import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math

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
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    
    if not use_alternative:
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
        gray = cv2.equalizeHist(gray)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.Canny(gray, 10, 250)
    
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    return edges

def advanced_document_contour(edges, img_area):
    """Zaawansowane znajdowanie konturu dokumentu"""
    contours, _ = cv2.findContours(
        edges, 
        cv2.RETR_LIST, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    document_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        if (len(approx) == 4 and 
            area > img_area * 0.20 and
            area < img_area * 0.97 and
            cv2.isContourConvex(approx)):
            document_contours.append(approx)
    
    if document_contours:
        return max(document_contours, key=cv2.contourArea)
    return None

def order_points(pts):
    """Porządkuj punkty: top-left, top-right, bottom-right, bottom-left"""
    # inicjalizacja listy współrzędnych w kolejności: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # top-left będzie mieć najmniejszą sumę, bottom-right największą sumę
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # top-right będzie mieć najmniejszą różnicę, bottom-left największą różnicę
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):
    """Transformacja perspektywy czteropunktowa"""
    # Porządkowanie punktów
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Obliczenie szerokości dokumentu
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    max_width = max(int(widthA), int(widthB))

    # Obliczenie wysokości dokumentu
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    max_height = max(int(heightA), int(heightB))

    # Zdefiniowanie docelowych punktów
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    # Obliczenie macierzy transformacji i zastosowanie
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    return warped

def estimate_capital_height(image):
    """Oszacuj wysokość wielkich liter w obrazie"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    heights = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 5 and w > 5:
            heights.append(h)
    
    return np.median(heights) if len(heights) > 0 else 30

def correct_perspective(img, target_capital_height=30):
    """Skoryguj perspektywę dokumentu"""
    # Usuń cień
    no_shadow_image = remove_shadow(img)
    
    # Wyszukaj krawędzie dokumentu
    edges = improved_edge_detection(no_shadow_image)
    
    try:
        # Próba znalezienia konturu dokumentu
        img_area = img.shape[0] * img.shape[1]
        contour = advanced_document_contour(edges, img_area)
        
        if contour is not None:
            # Spłaszcz kontur do punktów
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            # Jeśli znaleziono cztery punkty
            if len(approx) == 4:
                # Transformacja perspektywy
                warped = four_point_transform(no_shadow_image, approx.reshape(4, 2))
                
                # Przeskaluj do optymalnej wysokości wielkich liter
                current_height = estimate_capital_height(warped)
                scale_factor = target_capital_height / current_height
                
                new_width = int(warped.shape[1] * scale_factor)
                new_height = int(warped.shape[0] * scale_factor)
                
                return cv2.resize(warped, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    except Exception as e:
        print(f"Błąd podczas korekcji perspektywy: {e}")
    
    # Jeśli nie udało się przetworzyć, zwróć oryginalny obraz
    return img

def process_document(input_path, output_path):
    """Główna funkcja przetwarzania dokumentu"""
    try:
        # 1. Wczytaj obraz
        original_image = load_image(input_path)
        
        # 2. Korekta perspektywy
        final_image = correct_perspective(original_image)
        
        # 3. Zapisz wynik
        Image.fromarray(final_image).save(output_path, quality=95)
        print(f"Zapisano przetworzony obraz jako: {output_path}")
        
        # Opcjonalnie: porównaj obrazy
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title("Oryginał")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(final_image)
        plt.title("Przetworzony")
        plt.axis('off')
        
        plt.show()
        
        return final_image
    
    except Exception as e:
        print(f"Błąd: {e}")
        return None

if __name__ == "__main__":
    input_path = "doc11.jpg"
    output_path = "dokument_processed.jpg"
    process_document(input_path, output_path)