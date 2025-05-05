import os
import time
import gc
import json
import re
import layoutparser as lp
from pathlib import Path
import logging
from datetime import datetime
from PIL import Image
import numpy as np
import cv2
import pytesseract
from tqdm import tqdm
import langdetect
import pandas as pd
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configure Tesseract - Change to your path if different
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Paths to model files
CONFIG_PATH = "E:\\Nowy folder (2)\\praktyki\\documentation\\models\\mask_rcnn_R_50_FPN_3x.yaml"
MODEL_PATH = "E:\\Nowy folder (2)\\praktyki\\documentation\\models\\R-50.pkl"

# Load layoutparser model (Detectron2)
model = None
try:
    if os.path.exists(CONFIG_PATH) and os.path.exists(MODEL_PATH):
        model = lp.Detectron2LayoutModel(
            config_path=CONFIG_PATH,
            model_path=MODEL_PATH,
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
            device="cpu"
        )
    else:
        logger.warning(
            f"Model files not found. Check paths: {CONFIG_PATH}, {MODEL_PATH}. Layoutparser functionality will be unavailable.")
except Exception as e:
    logger.warning(
        f"Error loading layoutparser model: {e}. Layoutparser functionality will be unavailable.")

# ================== ORYGINALNE FUNKCJE POMOCNICZE ==================
def clean_unicode_text(text):
    replacements = {
        '\u201c': '"', '\u201d': '"', '\u2018': "'", '\u2019': "'",
        '\u00df': 'ss', '\ufb01': 'fi', '\ufb02': 'fl', '\u00bb': '»',
        '\u00ab': '«', '\u2013': '-', '\u2014': '--', '\u00e4': 'ä',
        '\u00f6': 'ö', '\u00fc': 'ü', '\u00c4': 'Ä', '\u00d6': 'Ö',
        '\u00dc': 'Ü'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = ''.join(ch for ch in text if ord(ch) >= 32 or ch in '\n\r\t')
    return text.strip()

def clean_phone_number(phone):
    """Standardize phone number formats"""
    # Remove all non-digit characters
    cleaned = re.sub(r'\D', '', phone)
    
    # Format international numbers
    if cleaned.startswith('31'):
        return f"+{cleaned}"
    elif len(cleaned) == 10:
        return f"{cleaned[:3]} {cleaned[3:6]} {cleaned[6:8]} {cleaned[8:]}"
    return phone  # Return original if doesn't match patterns

doc_num_patterns = [
    r'(Rechnung(?:s-?)?(?:nummer|nr\.?)[:.\s]*[A-Za-z0-9_/-]+)',
    r'(Beleg(?:nummer|nr\.?)[:.\s]*[A-Za-z0-9_/-]+)',
    r'(Factuurnr\.?[:.\s]*[A-Za-z0-9_/-]+)',
    r'([A-Za-z]+/[A-Za-z]+/\d+/\d+[A-Za-z]*)',
    r'(\d{4}\.\d{2}\.\d{3})',
    r'Rechnung\s+Nr\.\s+([\w\-/]+)'
]

date_patterns = [
    r'(\d{2}[.-/]\d{2}[.-/]\d{4})',
    r'(\d{2}[.-/]\d{2}[.-/]\d{2})',
    r'(\d{4}[.-/]\d{2}[.-/]\d{2})',
    r'(\d{1,2}\.\s*[A-Za-zäöüÄÖÜß]+\s*\d{4})',
    r'(\d{1,2}\s+[A-Za-zäöüÄÖÜß]+\s+\d{4})',
    r'(\d{1,2}-[a-z]{3}-\d{2})',
    r'(\d{1,2}\s+[a-z]{3,}\s+\d{4})'
]

amount_patterns = [
    r'(\d+(?:\.\d{3})*,\d{2}\s*€)',
    r'(€\s*\d+(?:\.\d{3})*,\d{2})',
    r'(Rechnungsbetrag:?\s*\d+(?:\.\d{3})*,\d{2})',
    r'(Summe:?\s*\d+[,.]\d{2})'
]

email_pattern = r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"
phone_pattern = r"(\+?\(?\d{2,3}\)?[-.\s]?\d{2,3}[-.\s]?\d{2}[-.\s]?\d{2}[-.\s]?\d{2,3})"

def clean_dates_extract_ids(dates, doc_numbers):
    cleaned_dates = []
    additional_ids = []
    id_patterns = [
        r'^(\d{4}\.\d{2}\.\d{3})$',  # Tylko ten format
    ]
    for date in dates:
        is_id = False
        for p in id_patterns:
            if re.match(p, date):
                additional_ids.append(date)
                is_id = True
                break
        if not is_id:
            cleaned_dates.append(date)
    final_doc_numbers = list(set(doc_numbers + additional_ids))
    return cleaned_dates, final_doc_numbers

def is_valid_date(date_str):
    """Validate common date formats found in Dutch documents"""
    if not isinstance(date_str, str):
        return False
        
    date_patterns = [
        r'^\d{1,2}[-\s][A-Za-z]{3,}[-\s]\d{2,4}$',  # 15-nov-23 or 15 nov 2023
        r'^\d{1,2}[-\s]\d{1,2}[-\s]\d{2,4}$',       # 15-11-2023
        r'^\d{1,2}\s[A-Za-z]+\s\d{4}$',              # 15 november 2023
        r'^\d{4}[-\s]\d{1,2}[-\s]\d{1,2}$',          # 2023-11-15
    ]
    
    for pattern in date_patterns:
        if re.fullmatch(pattern, date_str, re.IGNORECASE):
            return True
    return False

# ================== NOWE FUNKCJE ==================
def group_document_pages(files):
    documents = defaultdict(list)
    for file in files:
        stem = file.stem
        if '_page' in stem:
            base, page = stem.rsplit('_page', 1)
            try:
                documents[base].append((int(page), file))
            except ValueError:
                documents[stem].append((1, file))
        else:
            documents[stem].append((1, file))
    for doc in documents:
        documents[doc].sort(key=lambda x: x[0])
    return documents

def extract_name_from_filename(filename):
    patterns = [
        r'^([A-Z][a-z]+_[A-Z][a-z]+)',
        r'^([A-Z][a-z]+)',
        r'([A-Za-z]+-\d+)'
    ]
    for p in patterns:
        match = re.search(p, filename)
        if match:
            return match.group(1).replace('_', ' ')
    return None

def enhanced_sender_receiver(text, filename):
    sender = ""
    receiver = ""
    
    # Standard Dutch tax office receiver pattern
    tax_office_pattern = r"Belastingdienst.*Postbus\s2536.*6401\sDA\sHEERLEN"
    if re.search(tax_office_pattern, text, re.DOTALL | re.IGNORECASE):
        receiver = "Belastingdienst\nPostbus 2536\n6401 DA HEERLEN"
    
    # Look for customer address patterns
    address_pattern = r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)*\n(?:[A-Za-z]+\.?\s)?[A-Za-z0-9\s]+[-\s]\d+\s*[A-Z]*\n\d{4}\s[A-Z]{2}\s[A-Za-z]+"
    matches = re.finditer(address_pattern, text)
    
    for match in matches:
        address_block = match.group(0)
        if not sender and "Belastingdienst" not in address_block:
            sender = address_block.strip()
    
    return sender, receiver

def extract_entities_from_text(text):
    if not isinstance(text, str):
        text = str(text)
    
    text = clean_unicode_text(text)

    # Initialize all possible return values
    dates = []
    amounts = []
    doc_identifiers = []
    emails = []
    phones = []
    sender = ""
    receiver = ""

    try:
        # Extract dates with validation
        potential_dates = re.findall(r'|'.join(date_patterns), text, re.IGNORECASE)
        dates = [d for d in potential_dates if isinstance(d, str) and is_valid_date(d)]
        
        # Process phone numbers
        raw_phones = re.findall(phone_pattern, text)
        phones = [clean_phone_number(p) for p in raw_phones if isinstance(p, str)]
        
        # Other extractions
        for p in amount_patterns:
            found = re.findall(p, text, re.IGNORECASE)
            amounts.extend([f for f in found if isinstance(f, str)])

        for p in doc_num_patterns:
            found = re.findall(p, text, re.IGNORECASE)
            doc_identifiers.extend([f for f in found if isinstance(f, str)])

        emails = [e for e in re.findall(email_pattern, text, re.IGNORECASE) if isinstance(e, str)]
        
        # Get sender/receiver
        sender, receiver = enhanced_sender_receiver(text, "filename")
        
    except Exception as e:
        logger.error(f"Error in entity extraction: {str(e)}")
    
    return dates, amounts, doc_identifiers, emails, phones, sender, receiver

# ================== ZMODYFIKOWANA LOGIKA PRZETWARZANIA ==================
def process_multi_page_document(pages, output_path):
    try:
        start_time = time.time()
        all_text = []
        doc_metadata = {
            'dates': set(),
            'amounts': set(),
            'doc_ids': set(),
            'sender': "",
            'receiver': "",
            'emails': set(),
            'phones': set()
        }

        for page_num, img_path in pages:
            try:
                # Open and process image
                image = Image.open(img_path).convert('RGB')
                image_np = np.array(image)
                
                # Perform OCR
                text = pytesseract.image_to_string(image, lang='deu+nld+eng+fra')
                clean_text = clean_unicode_text(text)
                all_text.append(clean_text)

                # Extract entities with error handling
                extraction_result = extract_entities_from_text(clean_text)
                dates, amounts, ids, emails, phones, sender, receiver = (
                    extraction_result if len(extraction_result) == 7 
                    else ([], [], [], [], [], "", "")
                )

                # Update metadata with fallback values
                doc_metadata['dates'].update(d for d in dates if isinstance(d, str))
                doc_metadata['amounts'].update(a for a in amounts if isinstance(a, str))
                doc_metadata['doc_ids'].update(i for i in ids if isinstance(i, str))
                doc_metadata['emails'].update(e for e in emails if isinstance(e, str))
                doc_metadata['phones'].update(p for p in phones if isinstance(p, str))
                doc_metadata['sender'] = sender if isinstance(sender, str) else ""
                doc_metadata['receiver'] = receiver if isinstance(receiver, str) else ""

                if model:
                    try:
                        # Layout analysis
                        layout = model.detect(image_np)
                        
                        # Improved sender/receiver detection from layout
                        blocks = sorted(layout, key=lambda b: b.block.y_1)
                        for block in blocks:
                            block_text = getattr(block, 'text', '')
                            
                            if not doc_metadata['sender'] and any(
                                kw in block_text.lower() 
                                for kw in ['absender', 'nadawca', 'from:', 'afzender']
                            ):
                                doc_metadata['sender'] = block_text.strip()
                            
                            if not doc_metadata['receiver'] and any(
                                kw in block_text.lower() 
                                for kw in ['empfänger', 'adresat', 'to:', 'ontvanger', 'an:']
                            ):
                                doc_metadata['receiver'] = block_text.strip()

                    except Exception as layout_error:
                        logger.warning(f"Layout analysis error: {str(layout_error)}")

            except Exception as page_error:
                logger.error(f"Page processing error: {str(page_error)}")
                continue

        full_text = '\n'.join(all_text)

        # Language detection with fallback
        try:
            lang = langdetect.detect(full_text) if full_text else "unknown"
        except:
            lang = "unknown"

        # Prepare final result with all safeguards
        result = {
            "file_name": pages[0][1].name.split('_page')[0] if pages else "unknown",
            "pages": len(pages),
            "ground_truth": {
                "gt_parse": {
                    "entities": {
                        "dates": list(doc_metadata['dates']),
                        "amounts": list(doc_metadata['amounts']),
                        "document_identifiers": list(doc_metadata['doc_ids']),
                        "parties": {
                            "sender": doc_metadata['sender'],
                            "receiver": doc_metadata['receiver']
                        },
                        "contact_data": {
                            "emails": list(doc_metadata['emails']),
                            "phone_numbers": list(doc_metadata['phones'])
                        }
                    },
                    "language": lang
                },
                "raw_text": full_text,
                "metadata": {
                    "document_type": pages[0][1].parent.name if pages and pages[0] else "unknown",
                    "processing_time": round(time.time() - start_time, 2),
                    "timestamp": datetime.now().isoformat()
                }
            }
        }

        # Safe JSON writing
        try:
            with open(output_path, 'a', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
        except Exception as write_error:
            logger.error(f"Failed to write results: {str(write_error)}")
            return False

        return True

    except Exception as e:
        logger.error(f"Document processing failed: {str(e)}", exc_info=True)
        return False

def generate_annotations(data_dir="data", max_files=None):
    base_dir = Path(data_dir) / "dokumenty3"
    output_path = Path(data_dir) / "donut_annotations.jsonl"

    if output_path.exists():
        output_path.unlink()

    image_files = list(base_dir.rglob('*.jpg')) + list(base_dir.rglob('*.png'))
    documents = group_document_pages(image_files)

    logger.info(f"Przetwarzanie {len(documents)} dokumentów...")

    processed = 0
    for doc_name, pages in tqdm(documents.items(), desc="Dokumenty"):
        if max_files and processed >= max_files:
            break

        if process_multi_page_document(pages, output_path):
            processed += 1

        if processed % 5 == 0:
            gc.collect()

    generate_summary_report(output_path)

    logger.info(f"Zakończono. Przetworzono {processed} dokumentów.")

def generate_summary_report(output_path):
    try:
        data = []
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))

        df = pd.DataFrame([{
            "file": x["file_name"],
            "type": x["ground_truth"]["metadata"]["document_type"],
            "pages": x["pages"],
            "has_sender": bool(x["ground_truth"]["gt_parse"]["entities"]["parties"]["sender"]),
            "has_receiver": bool(x["ground_truth"]["gt_parse"]["entities"]["parties"]["receiver"]),
            "num_dates": len(x["ground_truth"]["gt_parse"]["entities"]["dates"]),
            "num_amounts": len(x["ground_truth"]["gt_parse"]["entities"]["amounts"]),
            "num_emails": len(x["ground_truth"]["gt_parse"]["entities"]["contact_data"]["emails"]),
            "num_phones": len(x["ground_truth"]["gt_parse"]["entities"]["contact_data"]["phone_numbers"]),
            "language": x["ground_truth"]["gt_parse"]["language"]
        } for x in data])

        print("\n=== Podsumowanie ===")
        print(f"Łącznie dokumentów: {len(df)}")
        print(f"Typy dokumentów:\n{df['type'].value_counts().to_string()}")
        print(f"\nJęzyki:\n{df['language'].value_counts().to_string()}")
        print("\nStatystyki:")
        print(f"- Średnia liczba stron: {df['pages'].mean():.1f}")
        print(f"- Dokumenty z nadawcą: {df['has_sender'].sum()} ({df['has_sender'].mean() * 100:.1f}%)")
        print(f"- Dokumenty z odbiorcą: {df['has_receiver'].sum()} ({df['has_receiver'].mean() * 100:.1f}%)")
        print(f"- Średnio dat na dokument: {df['num_dates'].mean():.1f}")
        print(f"- Średnio kwot na dokument: {df['num_amounts'].mean():.1f}")
        print(f"- Średnio emaili na dokument: {df['num_emails'].mean():.1f}")
        print(f"- Średnio numerów telefonów na dokument: {df['num_phones'].mean():.1f}")

    except Exception as e:
        logger.error(f"Błąd generowania raportu: {str(e)}")

if __name__ == "__main__":
    generate_annotations(data_dir="data", max_files=10)