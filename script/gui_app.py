import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
from document_reader import process_document_with_metadata
from utils import get_supported_files, convert_pdf_to_images
from PIL import Image, ImageTk
import threading
from document_fields import DocumentFieldDetector
import cv2
import tempfile
import sys
import traceback

class DocumentAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Document Analyzer")
        self.root.geometry("800x650")
        self.is_analyzing = False
        self.file_paths = []
        self.current_file_index = 0
        self.show_preview_flag = False
        self.stop_processing = False  # Flag to stop processing
        
        # Styling
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TButton', padding=6, font=('Arial', 10))
        self.style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        self.style.configure('Disabled.TButton', foreground='gray')
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # File selection panel
        file_frame = ttk.LabelFrame(main_frame, text="Select Documents", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))

        self.file_path = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path, width=60).pack(side=tk.LEFT, padx=(0, 10))
        
        btn_frame = ttk.Frame(file_frame)
        btn_frame.pack(side=tk.LEFT)
        
        ttk.Button(btn_frame, text="Files...", command=self.browse_files).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Folder...", command=self.browse_folder).pack(side=tk.LEFT, padx=5)

        # Preview frame (collapsible)
        self.preview_frame = ttk.LabelFrame(main_frame, text="Document Preview", padding=5)
        self.preview_toggle = ttk.Button(
            main_frame, 
            text="Show Preview", 
            command=self.toggle_preview,
            width=15
        )
        self.preview_toggle.pack(pady=(5, 0))
        
        self.image_label = ttk.Label(self.preview_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # Navigation and buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(10, 0))

        # Navigation buttons
        nav_frame = ttk.Frame(control_frame)
        nav_frame.pack(side=tk.LEFT)
        
        self.prev_btn = ttk.Button(
            nav_frame,
            text="◄ Previous",
            command=self.show_previous_file,
            state=tk.DISABLED,
            width=10
        )
        self.prev_btn.pack(side=tk.LEFT)
        
        self.next_btn = ttk.Button(
            nav_frame,
            text="Next ►",
            command=self.show_next_file,
            state=tk.DISABLED,
            width=10
        )
        self.next_btn.pack(side=tk.LEFT, padx=5)
        
        self.file_counter = ttk.Label(nav_frame, text="File 0 of 0")
        self.file_counter.pack(side=tk.LEFT, padx=10)

        # Action buttons
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(side=tk.RIGHT)
        
        self.analyze_button = ttk.Button(
            btn_frame, 
            text="Analyze", 
            command=self.start_analysis,
            width=10
        )
        self.analyze_button.pack(side=tk.LEFT)
        
        self.stop_button = ttk.Button(
            btn_frame,
            text="Stop",
            command=self.stop_analysis,
            state=tk.DISABLED,
            width=10
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            btn_frame, 
            text="Clear", 
            command=self.clear_all,
            width=10
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            btn_frame, 
            text="Exit", 
            command=self.root.quit,
            width=10
        ).pack(side=tk.LEFT)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready for analysis")
        ttk.Label(
            main_frame, 
            textvariable=self.status_var, 
            relief=tk.SUNKEN, 
            anchor=tk.W
        ).pack(fill=tk.X, pady=(10, 0))

        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')

    def toggle_preview(self):
        """Toggle preview visibility"""
        self.show_preview_flag = not self.show_preview_flag
        
        if self.show_preview_flag:
            self.preview_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
            self.preview_toggle.config(text="Hide Preview")
            if self.file_paths:
                self.load_current_preview()
        else:
            self.preview_frame.pack_forget()
            self.preview_toggle.config(text="Show Preview")

    def load_current_preview(self):
        """Load preview for current file"""
        if not self.file_paths or not self.show_preview_flag:
            return
            
        current_path = self.file_paths[self.current_file_index]
        try:
            if current_path.lower().endswith('.pdf'):
                # Show first page of PDF
                images = convert_pdf_to_images(current_path)
                if images:
                    img = Image.fromarray(cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB))
            else:
                img = Image.open(current_path)
            
            img.thumbnail((700, 500))
            photo = ImageTk.PhotoImage(img)
            
            self.image_label.configure(image=photo)
            self.image_label.image = photo
        except Exception as e:
            self.safe_error_report(f"Cannot load preview: {str(e)}")

    def browse_files(self):
        filetypes = (
            ('Supported files', '*.jpg *.jpeg *.png *.pdf'),
            ('Images', '*.jpg *.jpeg *.png'),
            ('PDFs', '*.pdf'),
            ('All files', '*.*')
        )
        
        filenames = filedialog.askopenfilenames(
            title='Select documents to analyze',
            initialdir=os.path.join(os.getcwd(), 'data', 'input'),
            filetypes=filetypes
        )
        
        if filenames:
            self.file_paths = list(filenames)
            self.current_file_index = 0
            self.update_file_display()

    def browse_folder(self):
        folder_path = filedialog.askdirectory(
            title='Select folder with documents',
            initialdir=os.path.join(os.getcwd(), 'data', 'input')
        )
        
        if folder_path:
            self.file_paths = get_supported_files(folder_path)
            if not self.file_paths:
                messagebox.showwarning("Warning", "No supported files found in selected folder")
                return
                
            self.current_file_index = 0
            self.update_file_display()

    def update_file_display(self):
        if not self.file_paths:
            return
            
        current_path = self.file_paths[self.current_file_index]
        self.file_path.set(f"{current_path} ({self.current_file_index+1}/{len(self.file_paths)})")
        
        # Update navigation buttons
        self.prev_btn.config(state=tk.NORMAL if self.current_file_index > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if self.current_file_index < len(self.file_paths)-1 else tk.DISABLED)
        self.file_counter.config(text=f"File {self.current_file_index+1} of {len(self.file_paths)}")
        
        # Load preview if enabled
        if self.show_preview_flag:
            self.load_current_preview()

    def show_previous_file(self):
        if self.current_file_index > 0:
            self.current_file_index -= 1
            self.update_file_display()

    def show_next_file(self):
        if self.current_file_index < len(self.file_paths) - 1:
            self.current_file_index += 1
            self.update_file_display()

    def start_analysis(self):
        if not self.file_paths:
            messagebox.showwarning("Warning", "No files selected for analysis")
            return
        
        if self.is_analyzing:
            messagebox.showwarning("Warning", "Analysis already in progress")
            return
        
        self.is_analyzing = True
        self.stop_processing = False
        self.status_var.set(f"Analyzing {len(self.file_paths)} documents...")
        self.progress.pack(fill=tk.X, pady=(5, 0))
        self.progress.start()
        self.analyze_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        thread = threading.Thread(target=self.run_batch_analysis, daemon=True)
        thread.start()
        self.check_thread(thread)

    def stop_analysis(self):
        self.stop_processing = True
        self.status_var.set("Stopping analysis...")

    def check_thread(self, thread):
        if thread.is_alive():
            self.root.after(100, lambda: self.check_thread(thread))
        else:
            self.progress.stop()
            self.progress.pack_forget()
            self.is_analyzing = False
            self.analyze_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            if self.stop_processing:
                self.status_var.set("Analysis stopped by user")
            else:
                self.status_var.set("Analysis completed")

    def safe_error_report(self, error_msg):
        """Thread-safe error reporting without recursion"""
        try:
            if self.root.winfo_exists():
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", 
                    error_msg[:500]  # Limit message length
                ))
                self.root.after(0, lambda: self.status_var.set(
                    f"Error: {error_msg[:100]}..."  # Truncate long messages
                ))
        except:
            pass

    def run_batch_analysis(self):
        try:
            # Create temp directory for PDF conversions
            temp_dir = tempfile.mkdtemp()
            
            # Track all processed annotations
            all_annotations = {}
            
            # Load existing annotations if file exists
            annotations_path = os.path.join('data', 'output', 'dataset', 'annotations.json')
            if os.path.exists(annotations_path):
                try:
                    with open(annotations_path, 'r') as f:
                        all_annotations = json.load(f)
                except (json.JSONDecodeError, IOError) as e:
                    self.safe_error_report(f"Error loading annotations: {str(e)}")
                    all_annotations = {}
            
            # Process files in chunks
            chunk_size = 10  # Process 10 files at a time
            for i in range(0, len(self.file_paths), chunk_size):
                if self.stop_processing:
                    break
                    
                chunk = self.file_paths[i:i+chunk_size]
                for j, file_path in enumerate(chunk):
                    if self.stop_processing:
                        break
                        
                    self.update_status(f"Processing file {i+j+1} of {len(self.file_paths)}: {os.path.basename(file_path)}")
                    
                    try:
                        field_detector = DocumentFieldDetector(
                            output_dir=os.path.join('data', 'output', 'dataset')
                        )
                        
                        if file_path.lower().endswith('.pdf'):
                            images = convert_pdf_to_images(file_path, temp_dir)
                            if not images:
                                raise ValueError("Failed to convert PDF to images")
                                
                            for page_num, image in enumerate(images):
                                if self.stop_processing:
                                    break
                                    
                                temp_image_path = os.path.join(temp_dir, f"temp_page_{page_num}.jpg")
                                cv2.imwrite(temp_image_path, image)
                                
                                result = field_detector.detect_fields(temp_image_path)
                                
                                if 'image_id' in result:
                                    all_annotations[result['image_id']] = result
                        
                        else:  # Regular image file
                            result = field_detector.detect_fields(file_path)
                            if 'image_id' in result:
                                all_annotations[result['image_id']] = result
                    
                    except Exception as e:
                        error_msg = f"Error processing {os.path.basename(file_path)}: {str(e)}"
                        print(error_msg)  # Log to console
                        self.safe_error_report(error_msg)
                        continue
                
                # Save intermediate results periodically
                if not self.stop_processing:
                    self.save_annotations(all_annotations, annotations_path)
            
            # Final save if not stopped
            if not self.stop_processing:
                self.save_annotations(all_annotations, annotations_path)
                self.update_status(f"Completed analysis of {len(self.file_paths)} documents")
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)  # Log full traceback to console
            self.safe_error_report(f"Unexpected error: {str(e)}")
        finally:
            # Clean up temp files
            try:
                for f in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, f))
                os.rmdir(temp_dir)
            except Exception as e:
                print(f"Error cleaning temp files: {str(e)}")

    def save_annotations(self, annotations, path):
        """Thread-safe annotation saving"""
        try:
            with open(path, 'w') as f:
                json.dump(annotations, f, indent=2)
        except Exception as e:
            self.safe_error_report(f"Error saving annotations: {str(e)}")

    def update_status(self, message):
        """Thread-safe status update"""
        if self.root.winfo_exists():
            self.root.after(0, lambda: self.status_var.set(message[:200]))  # Limit message length

    def clear_all(self):
        if not self.is_analyzing:
            self.file_paths = []
            self.current_file_index = 0
            self.file_path.set("")
            self.image_label.configure(image='')
            self.image_label.image = None
            self.status_var.set("Ready for analysis")
            self.prev_btn.config(state=tk.DISABLED)
            self.next_btn.config(state=tk.DISABLED)
            self.file_counter.config(text="File 0 of 0")

    def on_close(self):
        if self.is_analyzing:
            if messagebox.askokcancel("Close", "Analysis in progress. Are you sure you want to quit?"):
                self.stop_processing = True
                self.root.destroy()
        else:
            self.root.destroy()

def main():
    root = tk.Tk()
    # Increase recursion limit to prevent stack overflow
    sys.setrecursionlimit(10000)
    app = DocumentAnalyzerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()