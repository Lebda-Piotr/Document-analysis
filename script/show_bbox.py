import cv2
import numpy as np
from document_fields import DocumentFieldDetector
from utils import get_supported_files
from preprocessing import process_document
import os
import argparse
import tkinter as tk
from PIL import Image, ImageTk

def display_documents_with_boxes(input_folder: str, use_preprocessing: bool = True):
    """Display all document images in a folder with text field bounding boxes"""
    # Get supported image files
    image_paths = get_supported_files(input_folder)
    if not image_paths:
        print(f"No supported images found in: {input_folder}")
        return

    # Initialize detector with temporary output directory
    detector = DocumentFieldDetector(output_dir=os.path.join(input_folder, "temp_annotations"))

    for idx, img_path in enumerate(image_paths):
        print(f"Processing {img_path}...")

        # Read image
        orig_image = cv2.imread(img_path)
        if orig_image is None:
            print(f"Could not read: {img_path}")
            continue

        # Use preprocessing if enabled
        if use_preprocessing:
            image = process_document(img_path, "", return_image=True, show_plots=False)
            # If preprocessing failed, use original image
            if image is None:
                image = orig_image.copy()
        else:
            image = orig_image.copy()

        # Create a copy for displaying
        display_image = orig_image.copy()

        # Calculate scale factor for the original image
        height, width = display_image.shape[:2]
        max_size = 1200
        scale = min(max_size / width, max_size / height)

        # Get annotations
        try:
            annotations = detector.detect_fields(img_path, preprocess=use_preprocessing)['annotations']
            print(f"Found {len(annotations)} text fields")
        except Exception as e:
            print(f"Processing failed for {img_path}: {str(e)}")
            continue

        # Draw bounding boxes with field types
        colors = {
            "Date": (0, 255, 0), # Green
            "Document_Number": (0, 0, 255), # Blue
            "Email": (255, 0, 0), # Red
            "Phone": (255, 165, 0), # Orange
            "Unknown": (128, 0, 128) # Purple
        }

        # Add overlay for better text visibility
        overlay = display_image.copy()

        for field in annotations:
            x1, y1, x2, y2 = field['bbox']
            field_type = field['field']
            color = colors.get(field_type, (128, 0, 128)) # Default to purple for unknown

            # Draw semi-transparent background
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

            # Draw border
            cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)

            # Add label with field type and text
            label = f"{field_type}: {field['value'][:20]}"
            if len(field['value']) > 20:
                label += "..."

            # Calculate text position and background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]  # Increased font size
            text_bg_x2 = x1 + text_size[0] + 5
            text_bg_y2 = y1 - 5
            text_bg_y1 = text_bg_y2 - text_size[1] - 5

            # Draw text background
            cv2.rectangle(display_image, (x1, text_bg_y1), (text_bg_x2, text_bg_y2), color, -1)

            # Draw text
            cv2.putText(display_image, label, (x1 + 2, y1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # Increased font size

        # Apply the overlay with transparency
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, display_image, 1 - alpha, 0, display_image)

        # Resize for display while maintaining quality
        display_height = int(height * scale)
        display_width = int(width * scale)
        display_image = cv2.resize(display_image, (display_width, display_height),
                                   interpolation=cv2.INTER_AREA)

        # Convert image to Tkinter format
        display_image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(display_image_rgb)

        # Create Tkinter window
        root = tk.Tk()
        root.title(f"Document {idx+1}/{len(image_paths)} - {os.path.basename(img_path)}")
        canvas = tk.Canvas(root, width=display_width, height=display_height)
        canvas.pack()
        tk_image = ImageTk.PhotoImage(pil_image)
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)

        # Function to show bounding boxes on hover
        def on_hover(event):
            canvas.delete("hover")
            for field in annotations:
                x1, y1, x2, y2 = field['bbox']
                if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                    canvas.create_rectangle(x1, y1, x2, y2, outline="yellow", width=2, tags="hover")
                    label = f"{field['field']}: {field['value'][:20]}"
                    if len(field['value']) > 20:
                        label += "..."
                    canvas.create_text(x1 + 2, y1 - 7, anchor=tk.NW, text=label, fill="white", font=("Helvetica", 12), tags="hover")

        canvas.bind("<Motion>", on_hover)

        root.mainloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document field visualization tool")
    parser.add_argument("--input", "-i", type=str, default="data/output/dataset/images",
                        help="Path to folder containing document images")
    parser.add_argument("--no-preprocess", action="store_true",
                        help="Disable preprocessing step")

    args = parser.parse_args()

    display_documents_with_boxes(args.input, not args.no_preprocess)