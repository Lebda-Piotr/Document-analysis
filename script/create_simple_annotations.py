import re
import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from PIL import Image
import pytesseract
import spacy
from bbox_extractor import BoundingBoxExtractor
from spacy.pipeline import EntityRuler
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span  
from typing import Union  
from dateparser import parse
from langdetect import detect

class DocumentProcessor:
    def __init__(self):
        self.nlp_nl = spacy.load("nl_core_news_md")
        self.nlp_de = spacy.load("de_core_news_md")
        self.nlp_en = spacy.load("en_core_web_md")
        self.nlp_da = spacy.load("da_core_news_md")
        self.nlp_fr = spacy.load("fr_core_news_md")
        
        self._add_custom_patterns()
        self._setup_matchers()
    
    def _init_matchers(self):
        """Inicjalizacja wszystkich matcherów raz"""
        self.recipient_matcher = Matcher(self.nlp_de.vocab)
        self.address_matcher = Matcher(self.nlp_de.vocab)
        
        # Wzorce dla nazwisk
        recipient_patterns = [
            [{"LOWER": {"IN": ["für", "fur"]}}, 
            {"LOWER": {"IN": ["herrn", "frau"]}}, 
            {"POS": "PROPN", "IS_TITLE": True}]
        ]
        
        # Wzorce dla adresów
        address_patterns = [
            [{"IS_ALPHA": True}, {"IS_DIGIT": True}, {"IS_PUNCT": True},
            {"TEXT": {"REGEX": r"\b\d{5}\b"}}, {"IS_ALPHA": True}]
        ]
        
        for pattern in recipient_patterns:
            self.recipient_matcher.add("RECIPIENT", [pattern])
            
        for pattern in address_patterns:
            self.address_matcher.add("ADDRESS", [pattern])

    def _add_custom_patterns(self):
        # Wspólne wzorce dla wszystkich języków
        for nlp in [self.nlp_nl, self.nlp_de, self.nlp_en, self.nlp_da, self.nlp_fr]:
            ruler = nlp.add_pipe("entity_ruler", before="ner")
            
            patterns = [
                # ID dokumentu: 1234.56.789
                {
                    "label": "DOC_ID",
                    "pattern": [
                        {"TEXT": {"REGEX": r"^\d{4}$"}},
                        {"TEXT": {"REGEX": r"^[.,]$"}},
                        {"TEXT": {"REGEX": r"^\d{2}$"}},
                        {"TEXT": {"REGEX": r"^[.,]$"}},
                        {"TEXT": {"REGEX": r"^\d{3}$"}}
                    ]
                },
                # ID dokumentu: CAP/UCF/24/017R
                {
                    "label": "DOC_ID",
                    "pattern": [
                        {"TEXT": {"REGEX": r"^[A-Z]{2,4}$"}},
                        {"TEXT": "/"},
                        {"TEXT": {"REGEX": r"^[A-Z]{2,4}$"}},
                        {"TEXT": "/"},
                        {"TEXT": {"REGEX": r"^\d{2}$"}},
                        {"TEXT": "/"},
                        {"TEXT": {"REGEX": r"^\d{3,4}[A-Z]?$"}}
                    ]
                },
                # IdNr: 92 047 135 005 (wersja ze spacjami)
                {
                    "label": "DOC_ID",
                    "pattern": [
                        {"LOWER": {"IN": ["idnr", "id-nr", "steuernummer", "steuernr"]}},
                        {"IS_PUNCT": True, "OP": "?"},
                        {"TEXT": {"REGEX": r"^\d{2}$"}},
                        {"TEXT": {"REGEX": r"^\d{3}$"}},
                        {"TEXT": {"REGEX": r"^\d{3}$"}},
                        {"TEXT": {"REGEX": r"^\d{3}$"}}
                    ]
                },
                # Steuernummer 339/2962/4644 (wersja z ukośnikami)
                {
                    "label": "DOC_ID",
                    "pattern": [
                        {"LOWER": {"IN": ["steuernummer", "steuernr"]}},
                        {"TEXT": {"REGEX": r"^\d{3}$"}},
                        {"TEXT": "/"},
                        {"TEXT": {"REGEX": r"^\d{4}$"}},
                        {"TEXT": "/"},
                        {"TEXT": {"REGEX": r"^\d{4}$"}}
                    ]
                },
                # Format daty: 03. Juli 2024 (niemiecki)
                {
                    "label": "DATE",
                    "pattern": [
                        {"TEXT": {"REGEX": r"^\d{1,2}$"}},
                        {"TEXT": "."},
                        {"LOWER": {"IN": [
                            "januar", "februar", "märz", "april", "mai", "juni",
                            "juli", "august", "september", "oktober", "november", "dezember"
                        ]}},
                        {"TEXT": {"REGEX": r"^\d{4}$"}}
                    ]
                },
                # Format daty: 28.06.24
                {
                    "label": "DATE",
                    "pattern": [
                        {"TEXT": {"REGEX": r"^\d{1,2}$"}},
                        {"TEXT": {"REGEX": r"^[./-]$"}},
                        {"TEXT": {"REGEX": r"^\d{1,2}$"}},
                        {"TEXT": {"REGEX": r"^[./-]$"}},
                        {"TEXT": {"REGEX": r"^\d{2,4}$"}}
                    ]
                },
                # Finanzamt jako organizacja
                {
                    "label": "ORG",
                    "pattern": [{"LOWER": "finanzamt"}, {"IS_ALPHA": True, "OP": "*"}]
                }
            ]
            ruler.add_patterns(patterns)

    def _setup_matchers(self):
        # Matcher dla numerów referencyjnych w kontekście
        self.matcher = Matcher(self.nlp_nl.vocab)
        self.matcher.add("REF_PATTERN", [
            [{"LOWER": {"IN": ["kenmerk", "nr", "referentie"]}},
             {"IS_PUNCT": True, "OP": "?"},
             {"TEXT": {"REGEX": r"^[\w./-]{6,}$"}}]
        ])

    def _clean_text(self, text):
        # Poprawa typowych błędów OCR
        replacements = {
            '‘': "'", '’': "'", '“': '"', '”': '"',
            '\u00a0': ' ', '\\n': ' ', ' voop ': ' voor ',
            'ﬁ': 'fi', 'ﬂ': 'fl'
        }
        for wrong, correct in replacements.items():
            text = text.replace(wrong, correct)
        return re.sub(r'\s+', ' ', text).strip()

    def _extract_dates(self, doc):
        """
        Ekstrakcja dat bez standaryzacji formatu, tylko weryfikacja poprawności
        """
        dates = []
        
        # Encje z modelu NLP
        for ent in doc.ents:
            if ent.label_ == "DATE":
                # Weryfikacja czy to rzeczywiście data (ignorujemy np. "letni", "roczny")
                if any(c.isdigit() for c in ent.text) and not ent.text.isalpha():
                    dates.append(ent.text)
        
        # Wzorce dat z zachowaniem oryginalnych separatorów
        date_patterns = [
            r'\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b',  # DD.MM.YYYY lub DD-MM-YY itp.
            r'\b\d{1,2}\s+(?:januar|februar|märz|april|mai|juni|juli|august|september|oktober|november|dezember)\s+\d{4}\b',
            r'\b(?:jan|feb|mär|apr|mai|jun|jul|aug|sep|okt|nov|dez)[a-z]*\s+\d{1,2},?\s+\d{4}\b'
        ]
        
        for pattern in date_patterns:
            for match in re.finditer(pattern, doc.text, re.IGNORECASE):
                date_str = match.group()
                # Prosta walidacja - czy zawiera rok w rozsądnym zakresie
                year_match = re.search(r'(19|20)\d{2}', date_str)
                if year_match and 1900 <= int(year_match.group()) <= 2100:
                    dates.append(date_str)
        
        # Usuwanie duplikatów zachowując kolejność wystąpień
        seen = set()
        unique_dates = []
        for date in dates:
            if date not in seen:
                seen.add(date)
                unique_dates.append(date)
        
        return unique_dates[:10]  # Limit 10 najwcześniejszych dat

    def _extract_document_ids(self, doc):
        ids = []
        
        # Encje z NER
        for ent in doc.ents:
            if ent.label_ == "DOC_ID":
                ids.append(ent.text)
        
        # Dopasowania z matchera
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            if self.nlp_nl.vocab.strings[match_id] == "REF_PATTERN":
                ids.append(doc[start:end].text)
        
        # Normalizacja
        normalized_ids = []
        for doc_id in ids:
            clean_id = re.sub(r'[^\w./-]', '', doc_id.replace(',', '.'))
            if len(clean_id) >= 8:  # Minimalna długość ID
                normalized_ids.append(clean_id)
        
        return list(set(normalized_ids))[:3]  # Limit 3 najważniejszych

    def _extract_contact_info(self, text):
        """
        Rozszerzone wykrywanie numerów telefonów i emaili
        """
        # Rozszerzone wzorce dla numerów międzynarodowych
        phone_patterns = [
            r'(?:\+?\d{2,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{3,5}[-.\s]?\d{3,4}',  # Międzynarodowe
            r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{3}\b',  # Niemieckie 9-cyfrowe
            r'\b\d{2,4}/\d{3,8}\b'  # Numery w formacie 123/456789
        ]
        
        phones = []
        for pattern in phone_patterns:
            phones.extend(re.findall(pattern, text))
        
        # Filtrowanie fałszywych pozycji
        valid_phones = []
        for phone in phones:
            clean_phone = re.sub(r'[^\d+]', '', phone)
            if 6 <= len(clean_phone) <= 15:  # Rozsądny zakres długości
                valid_phones.append(phone)
        
        # Emails z walidacją domeny
        email_pattern = r'\b[\w.-]+@(?!example\.com)[\w.-]+\.(?:[a-z]{2,8}|xn--[a-z0-9]+)\b'
        emails = list(set(re.findall(email_pattern, text, re.IGNORECASE)))
        
        return valid_phones[:3], emails[:2]

    def process_page(self, image_path):
        try:
            # OCR z bounding boxami
            bboxes = BoundingBoxExtractor.get_bboxes(image_path)
            text = " ".join([bbox['text'] for bbox in bboxes])
            text = self._clean_text(text)
            
            # Detekcja języka
            try:
                lang = detect(text[:850])  # Próbka tekstu dla lepszej dokładności
            except:
                lang = 'de' if 'finanzamt' in text.lower() else 'nl'
                
            nlp = {
                'nl': self.nlp_nl,
                'de': self.nlp_de,
                'en': self.nlp_en,
                'da': self.nlp_da,
                'fr': self.nlp_fr
            }.get(lang, self.nlp_de)
            
            # Przetwarzanie NLP
            doc = nlp(text[:100000])
            
            # Ekstrakcja encji
            dates = self._extract_dates(doc)
            doc_ids = self._extract_document_ids(doc)
            phones, emails = self._extract_contact_info(text)
            
            # Organizacje z filtrami
            org_blacklist = {'eingegangen', 'uber', 'sehr', 'geehrte', 'steuerzahler'}
            organizations = list(set([
                ent.text for ent in doc.ents 
                if ent.label_ == "ORG"
                and not any(bad_word in ent.text.lower() for bad_word in org_blacklist)
                and len(ent.text) > 3
            ]))[:5]  # Limit 5 organizacji
            
            # Ekstrakcja nazwiska i adresu
            dates = self._extract_dates(doc)
            doc_ids = self._extract_document_ids(doc)
            phones, emails = self._extract_contact_info(text)  # Tu przekazujemy text, nie doc
            organizations = self._filter_organizations(doc)
            recipient_name = self._extract_recipient_name(doc)
            recipient_address = self._extract_recipient_address(doc)
            
            # Przygotowanie struktury outputu - ZACHOWUJEMY STARĄ I NOWĄ STRUKTURĘ
            return {
                # Stara struktura (dla kompatybilności)
                "text": text,
                "entities": {
                    "dates": dates,
                    "document_ids": doc_ids,
                    "phones": phones,
                    "emails": emails,
                    "organizations": organizations,
                    "language": lang
                },
                # Nowa struktura
                "content": {
                    "header": self._match_with_bboxes({
                        "tax_office": next((ent.text for ent in doc.ents if ent.label_ == "ORG" and "finanzamt" in ent.text.lower()), ""),
                        "date": dates[0] if dates else ""
                    }, bboxes),
                    "recipient": self._match_with_bboxes({
                        "name": recipient_name,
                        "address": recipient_address
                    }, bboxes),
                    "tax_info": self._match_with_bboxes({
                        "tax_number": next((id for id in doc_ids if '/' in id), ""),
                        "id_number": next((id for id in doc_ids if 'idnr' in id.lower()), "")
                    }, bboxes),
                    "contact": {
                        "phones": phones[:2],
                        "emails": emails[:1]
                    }
                },
                "metadata": {
                    "processing_time": datetime.now().isoformat(),
                    "bboxes": bboxes  # Pełne dane bboxów do debugowania
                }
            }
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None
    
    def _match_with_bboxes(self, entity, bboxes):
        """Bezpieczne dopasowanie encji do bboxów"""
        if not entity or not bboxes:
            return None
            
        if isinstance(entity, dict):  # Jeśli już przetworzone
            return entity
            
        text_to_match = str(entity)
        for bbox in bboxes:
            if text_to_match in bbox.get('text', ''):
                return {
                    "value": text_to_match,
                    "bbox": [bbox['x'], bbox['y'], bbox['width'], bbox['height']],
                    "page": bbox.get('page_num', 1)
                }
        return {"value": text_to_match, "bbox": None, "page": None}

    def _extract_recipient_name(self, doc):
        """
        Ulepszone wykrywanie nazwisk z obsługą błędów
        """
        if not isinstance(doc, (Doc, Span)):  # Sprawdzenie typu
            return ""
            
        blacklist = {
            'festgesetzt', 'solidaritatszuschlag', 'einkommensteuer',
            'steuer', 'bescheid', 'seite', 'anlage'
        }
        
        try:
            # Inicjalizacja Matcher tylko jeśli doc jest prawidłowy
            matcher = Matcher(self.nlp_de.vocab)
            
            patterns = [
                [{"LOWER": {"IN": ["für", "fur"]}}, 
                {"LOWER": {"IN": ["herrn", "frau", "herr", "hr"]}}, 
                {"POS": "PROPN", "IS_TITLE": True, "LENGTH": {">=": 3}}],
                
                [{"LOWER": "sehr"}, {"LOWER": "geehrte"}, 
                {"POS": "PROPN", "IS_TITLE": True, "LENGTH": {">=": 3}}]
            ]
            
            for pattern in patterns:
                matcher.add("RECIPIENT_PATTERN", [pattern])
            
            matches = matcher(doc)
            for _, start, end in matches:
                candidate = doc[start:end].text
                if (len(candidate) >= 3 and 
                    not any(bad in candidate.lower() for bad in blacklist)):
                    return candidate
                    
            # Rezerwowa heurystyka
            for token in doc:
                if (token.pos_ == "PROPN" and token.is_title and 
                    len(token.text) >= 3 and token.text.isalpha() and
                    not any(bad in token.text.lower() for bad in blacklist)):
                    return token.text
                    
        except Exception as e:
            print(f"Error in name extraction: {str(e)}")
        
        return ""
    
    def _format_entities(self, entities, limit=None):
        """Formatuje encje do spójnej struktury z bboxami"""
        formatted = []
        for entity in entities[:limit] if limit else entities:
            if isinstance(entity, dict):
                formatted.append(entity)  # Zakładając, że już ma bbox
            else:
                formatted.append({
                    "value": entity,
                    "bbox": None,
                    "page": None
                })
        return formatted

    def _filter_organizations(self, doc):
        """
        Zaawansowane filtrowanie z wykorzystaniem cech lingwistycznych
        """
        org_blacklist = {
            'einspruch', 'einkünfte', 'festgesetzt', 'seite', 'anlage',
            'steuer', 'bescheid', 'finanzamt', 'antrag'
        }
        
        valid_orgs = []
        for ent in doc.ents:
            if ent.label_ == "ORG":
                text = ent.text.strip()
                
                # Warunki walidacji z użyciem właściwości tokenów
                tokens = [t for t in ent if not t.is_punct]
                valid = (
                    len(text) >= 4 and
                    not any(t.is_digit for t in tokens) and
                    not any(word in text.lower() for word in org_blacklist) and
                    any(t.is_upper or t.is_title for t in tokens[:2]) and
                    not any(t.text.lower() in {'der', 'die', 'das'} for t in tokens)
                )
                
                if valid:
                    # Korekta typowych błędów OCR
                    corrected = re.sub(r'[^a-zA-ZäöüßÄÖÜ\s-]', '', text)
                    if len(corrected) >= 3:
                        valid_orgs.append(corrected)
        
        return list(set(valid_orgs))[:3]  # Zwracaj max 3 unikalne organizacje

    def _extract_sender(self, organizations):
        """Identyfikuje nadawcę spośród organizacji"""
        for org in organizations:
            if 'finanzamt' in org.lower() or 'tax' in org.lower():
                return org
        return organizations[0] if organizations else ""

    def _detect_document_type(self, text):
        """Heurystyka do detekcji typu dokumentu"""
        text_lower = text.lower()
        if 'steuernummer' in text_lower:
            return 'german_tax_id_assignment'
        elif 'rechnung' in text_lower:
            return 'invoice'
        return 'other'
    
    def _extract_recipient_address(self, doc):
        """
        Wykrywanie adresów z obsługą błędów wejścia
        """
        if not isinstance(doc, (Doc, Span)):
            return None
            
        try:
            plz_pattern = r"\b\d{5}\b"
            matcher = Matcher(self.nlp_de.vocab)
            
            patterns = [
                [{"IS_ALPHA": True}, {"IS_DIGIT": True}, {"IS_PUNCT": True},
                {"TEXT": {"REGEX": plz_pattern}}, {"IS_ALPHA": True}],
                
                [{"TEXT": {"REGEX": plz_pattern}}, {"IS_ALPHA": True}]
            ]
            
            for pattern in patterns:
                matcher.add("ADDRESS_PATTERN", [pattern])
            
            matches = matcher(doc)
            for _, start, end in matches:
                return doc[start:end].text
                
            # Rezerwowe wyszukiwanie
            plz_match = re.search(plz_pattern + r"\s+[A-Za-zäöüßÄÖÜ]+", doc.text)
            return plz_match.group() if plz_match else None
            
        except Exception as e:
            print(f"Error in address extraction: {str(e)}")
            return None

    def generate_annotations(self, data_dir, output_file):
        documents = defaultdict(list)
        for img_path in Path(data_dir).rglob('*.[pj][np]g'):
            stem = img_path.stem
            if '_page' in stem:
                base, page = stem.rsplit('_page', 1)
                documents[base].append((int(page), img_path))
            else:
                documents[stem].append((1, img_path))
        
        def parse_date(date_str):
            """Bezpieczna funkcja parsująca różne formaty dat"""
            if isinstance(date_str, dict):
                date_str = date_str["value"]
            
            # Zamień kropki na myślniki jeśli to konieczne
            normalized_date = date_str.replace('.', '-')
            try:
                return datetime.strptime(normalized_date, "%d-%m-%Y")
            except ValueError:
                try:
                    return datetime.strptime(normalized_date, "%d-%m-%y")
                except ValueError:
                    return None
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for doc_name, pages in tqdm(sorted(documents.items())):
                pages.sort(key=lambda x: x[0])
                page_results = []
                for _, page_path in pages:
                    page_data = self.process_page(page_path)
                    if page_data and "entities" in page_data:  # Ważne sprawdzenie!
                        page_results.append(page_data)
                
                if not page_results:
                    continue
                
                # Agregacja wyników z wielu stron (używamy STAREJ struktury entities)
                combined_entities = {
                    "dates": [],
                    "document_ids": [],
                    "phones": [],
                    "emails": [],
                    "organizations": []
                }
                
                for page in page_results:
                    for key in combined_entities:
                        if key in page["entities"]:  # Dodatkowe zabezpieczenie
                            combined_entities[key].extend(page["entities"][key])
                
                # Usuwanie duplikatów
                def remove_duplicates(entity_list):
                    seen = set()
                    unique_list = []
                    for entity in entity_list:
                        if isinstance(entity, dict):
                            identifier = entity["value"]
                        else:
                            identifier = entity
                        
                        if identifier not in seen:
                            seen.add(identifier)
                            unique_list.append(entity)
                    return unique_list
                
                # Przygotowanie finalnego wyniku
                annotation = {
                    "file_name": pages[0][1].stem.split('_page')[0],
                    "pages": len(pages),
                    "ground_truth": {
                        "gt_parse": {
                            # Używamy nowej struktury content z pierwszego page_result
                            "content": page_results[0]["content"] if "content" in page_results[0] else {},
                            # Zachowujemy starą strukturę entities dla kompatybilności
                            "entities": {
                                "dates": sorted(
                                    [d for d in remove_duplicates(combined_entities["dates"]) if parse_date(d)],
                                    key=lambda x: parse_date(x),
                                    reverse=True
                                )[:5],
                                "document_ids": sorted(
                                    remove_duplicates(combined_entities["document_ids"]),
                                    key=lambda x: len(x["value"]) if isinstance(x, dict) else len(x),
                                    reverse=True
                                )[:3],
                                "phones": remove_duplicates(combined_entities["phones"])[:3],
                                "emails": remove_duplicates(combined_entities["emails"])[:2],
                                "organizations": remove_duplicates(combined_entities["organizations"])[:3],
                                "language": page_results[0]["entities"].get("language", "nl")
                            }
                        },
                        "raw_text": '\n'.join(p["text"] for p in page_results),
                        "metadata": {
                            "document_type": pages[0][1].parent.name,
                            "processing_time": datetime.now().isoformat(),
                            # Dodajemy bboxy jeśli są dostępne
                            "bboxes_available": any("bboxes" in p.get("metadata", {}) for p in page_results)
                        }
                    }
                }
                
                f.write(json.dumps(annotation, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    processor = DocumentProcessor()
    processor.generate_annotations(
        data_dir="data/dokumenty3",
        output_file="donut_annotations.jsonl"
    )