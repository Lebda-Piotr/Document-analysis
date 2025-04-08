import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
from document_reader import process_document_with_metadata
from PIL import Image, ImageTk
import threading
from document_fields import DocumentFieldDetector

class DocumentAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Analizator Dokumentów")
        self.root.geometry("800x600")
        self.is_analyzing = False  # Flaga śledząca stan analizy
        
        # Stylizacja
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TButton', padding=6, font=('Arial', 10))
        self.style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        self.style.configure('Disabled.TButton', foreground='gray')  # Styl dla wyłączonego przycisku
        
        # Obsługa zamknięcia okna
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.create_widgets()
        
    def create_widgets(self):
        # Główny kontener
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Panel wyboru pliku
        file_frame = ttk.LabelFrame(main_frame, text="Wybierz dokument", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.file_path = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path, width=50).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(file_frame, text="Przeglądaj...", command=self.browse_file).pack(side=tk.LEFT)
        
        # Panel podglądu obrazu
        preview_frame = ttk.LabelFrame(main_frame, text="Podgląd dokumentu", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True)
        
        self.image_label = ttk.Label(preview_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Panel przycisków
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.analyze_button = ttk.Button(
            button_frame, 
            text="Analizuj dokument", 
            command=self.start_analysis
        )
        self.analyze_button.pack(side=tk.LEFT)
        
        ttk.Button(button_frame, text="Wyczyść", command=self.clear_all).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Wyjdź", command=self.root.quit).pack(side=tk.RIGHT)
        
        # Panel statusu
        self.status_var = tk.StringVar()
        self.status_var.set("Gotowy do analizy")
        ttk.Label(
            main_frame, 
            textvariable=self.status_var, 
            relief=tk.SUNKEN, 
            anchor=tk.W
        ).pack(fill=tk.X, pady=(10, 0))
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
    
    def browse_file(self):
        filetypes = (
            ('Obrazy', '*.jpg *.jpeg *.png'),
            ('Wszystkie pliki', '*.*')
        )
        
        filename = filedialog.askopenfilename(
            title='Wybierz dokument do analizy',
            initialdir=os.path.join(os.getcwd(), 'data', 'input'),
            filetypes=filetypes
        )
        
        if filename:
            self.file_path.set(filename)
            self.show_preview(filename)
    
    def show_preview(self, image_path):
        try:
            img = Image.open(image_path)
            img.thumbnail((600, 400))
            photo = ImageTk.PhotoImage(img)
            
            self.image_label.configure(image=photo)
            self.image_label.image = photo  # Keep a reference
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie można wyświetlić podglądu: {str(e)}")
    
    def start_analysis(self):
        if not self.file_path.get():
            messagebox.showwarning("Uwaga", "Wybierz plik do analizy")
            return
        
        if self.is_analyzing:
            messagebox.showwarning("Uwaga", "Analiza już w trakcie. Proszę czekać...")
            return
        
        self.is_analyzing = True
        self.status_var.set("Analizowanie dokumentu...")
        self.progress.pack(fill=tk.X, pady=(5, 0))
        self.progress.start()
        
        # Wyłącz przyciski podczas analizy
        self.analyze_button.config(state=tk.DISABLED, style='Disabled.TButton')
        
        thread = threading.Thread(target=self.run_analysis, daemon=True)
        thread.start()
        
        self.check_thread(thread)
    
    def check_thread(self, thread):
        if thread.is_alive():
            self.root.after(100, lambda: self.check_thread(thread))
        else:
            self.progress.stop()
            self.progress.pack_forget()
            self.is_analyzing = False
            self.analyze_button.config(state=tk.NORMAL, style='TButton')
    
    def run_analysis(self):
        try:
            input_path = self.file_path.get()
            
            # Initialize field detector
            field_detector = DocumentFieldDetector(
                output_dir=os.path.join('..', 'data', 'output', 'dataset')
            )
            
            # Detect fields
            result = field_detector.detect_fields(input_path)
            
            # Show success message
            self.root.after(0, lambda: self.status_var.set(
                f"Detected {len(result['annotations'])} fields in document"
            ))
            
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set("Error during analysis"))
            self.root.after(0, lambda: messagebox.showerror("Error", f"Analysis error: {str(e)}"))
    
    def clear_all(self):
        if not self.is_analyzing:
            self.file_path.set("")
            self.image_label.configure(image='')
            self.image_label.image = None
            self.status_var.set("Gotowy do analizy")
    
    def on_close(self):
        if self.is_analyzing:
            if messagebox.askokcancel("Zamknij", "Analiza w trakcie. Czy na pewno chcesz zamknąć aplikację?"):
                self.root.destroy()
        else:
            self.root.destroy()

def main():
    root = tk.Tk()
    app = DocumentAnalyzerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()