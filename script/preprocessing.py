import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
import os

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
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    """Transformacja perspektywy czteropunktowa"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    max_width = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    max_height = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

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

def enhance_image_quality(image):
    """Poprawa jakości obrazu przed OCR"""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return image

def adjust_for_ocr(image, dpi=300, simple_mode=True):
    """Dostosowanie obrazu dla lepszego OCR - uproszczona wersja"""
    # Skalowanie do docelowego DPI
    scale_factor = dpi / 72.0
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    if simple_mode:
        # Tylko konwersja do skali szarości
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return gray
    else:
        # Lekkie poprawienie kontrastu (CLAHE)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(gray)

def correct_perspective(img, target_capital_height=30):
    """Skoryguj perspektywę dokumentu"""
    no_shadow_image = remove_shadow(img)
    edges = improved_edge_detection(no_shadow_image)
    
    try:
        img_area = img.shape[0] * img.shape[1]
        contour = advanced_document_contour(edges, img_area)
        
        if contour is not None:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            if len(approx) == 4:
                warped = four_point_transform(no_shadow_image, approx.reshape(4, 2))
                current_height = estimate_capital_height(warped)
                scale_factor = target_capital_height / current_height
                new_width = int(warped.shape[1] * scale_factor)
                new_height = int(warped.shape[0] * scale_factor)
                return cv2.resize(warped, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    except Exception as e:
        print(f"Błąd podczas korekcji perspektywy: {e}")
    
    return img

def process_document(input_path, output_path, simple_preprocess=True, show_plots=False):
    """Główna funkcja przetwarzania dokumentu - uproszczona"""
    try:
        # 1. Wczytaj obraz
        original_image = load_image(input_path)
        
        # 2. Korekta perspektywy
        final_image = correct_perspective(original_image)
        
        # 3. Poprawa jakości (uproszczona)
        final_image = enhance_image_quality(final_image)
        
        # 4. Dostosowanie dla OCR (uproszczone)
        final_image = adjust_for_ocr(final_image, simple_mode=simple_preprocess)
        
        # 5. Zapisz wynik
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        Image.fromarray(final_image).save(output_path, quality=95)
        print(f"Zapisano przetworzony obraz jako: {output_path}")
        
        if show_plots:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(original_image)
            plt.title("Oryginał")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(final_image, cmap='gray')
            plt.title("Przetworzony")
            plt.axis('off')
            
            plt.show()
        return final_image
    
    except Exception as e:
        print(f"Błąd: {e}")
        return None

if __name__ == "__main__":
    input_path = os.path.join('data', 'input', 'doc11.jpg')
    output_path = os.path.join('data', 'output', 'dokument_processed.jpg')
    process_document(input_path, output_path)