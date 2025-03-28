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
            area > img_area * 0.1 and 
            area < img_area * 0.95 and 
            cv2.isContourConvex(approx)):
            document_contours.append(approx)
    
    return max(document_contours, key=cv2.contourArea) if document_contours else None

def fallback_full_image_processing(img):
    """Przetwarzanie gdy nie znaleziono konturu - pełny obraz"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    rows = np.where(~np.all(binary == 255, axis=1))[0]
    cols = np.where(~np.all(binary == 255, axis=0))[0]
    
    if len(rows) > 0 and len(cols) > 0:
        cropped = img[rows.min():rows.max()+1, cols.min():cols.max()+1]
        return cropped
    
    return img

def correct_perspective(img, contour):
    """Skoryguj perspektywę dokumentu (prostowanie)"""
    pts = contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    
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

def scale_to_dpi(image, target_dpi=300):
    """Skaluj obraz do zadanej rozdzielczości DPI"""
    # Domyślnie zakładamy 96 DPI dla obrazów
    original_dpi = 96
    scale_factor = target_dpi / original_dpi
    
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    
    scaled_image = cv2.resize(
        image, 
        (new_width, new_height), 
        interpolation=cv2.INTER_LANCZOS4
    )
    
    return scaled_image

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
        
        # 5. Skalowanie do 300 DPI
        final_image_scaled = scale_to_dpi(final_image)
        
        # Zapisz
        Image.fromarray(final_image_scaled).save(output_path, quality=95)
        print(f"Zapisano przetworzony obraz jako: {output_path}")
        
        return final_image_scaled
    
    except Exception as e:
        print(f"Błąd: {e}")
        return None

if __name__ == "__main__":
    input_path = "doc11.jpg"
    output_path = "dokument_processed.jpg"
    process_document(input_path, output_path)