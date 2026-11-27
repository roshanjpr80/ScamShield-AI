"""
OCR Processor for Document Analysis.

Extracts text and structured data from financial documents,
KYC documents, payment screenshots, and other fraud-relevant documents.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)


class OCREngine(Enum):
    """Supported OCR engines."""
    TESSERACT = "tesseract"
    EASYOCR = "easyocr"
    PADDLEOCR = "paddleocr"


@dataclass
class OCRResult:
    """Result of OCR processing."""
    text: str
    confidence: float
    bounding_boxes: List[Dict[str, Any]]
    language: str
    processing_time_ms: float


@dataclass
class StructuredField:
    """A structured field extracted from document."""
    field_name: str
    value: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    field_type: str  # text, number, date, amount, id, vpa


@dataclass
class DocumentExtraction:
    """Complete document extraction result."""
    document_type: str
    raw_text: str
    structured_fields: List[StructuredField]
    key_value_pairs: Dict[str, str]
    tables: List[Dict[str, Any]]
    confidence: float
    warnings: List[str]


class OCRProcessor:
    """
    OCR Processor for financial and KYC documents.

    Supports multiple OCR engines and provides specialized extraction
    for UPI payment screenshots, bank statements, KYC documents, etc.
    """

    def __init__(
        self,
        engine: OCREngine = OCREngine.TESSERACT,
        languages: List[str] = None,
        gpu: bool = False,
    ):
        self.engine = engine
        self.languages = languages or ["eng", "hin"]
        self.gpu = gpu
        self._reader = None
        self._init_engine()

    def _init_engine(self):
        """Initialize the selected OCR engine."""
        if self.engine == OCREngine.TESSERACT:
            try:
                import pytesseract
                self._tesseract_available = True
            except ImportError:
                logger.warning("Tesseract not available")
                self._tesseract_available = False

        elif self.engine == OCREngine.EASYOCR:
            try:
                import easyocr
                self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
                self._easyocr_available = True
            except ImportError:
                logger.warning("EasyOCR not available")
                self._easyocr_available = False

        elif self.engine == OCREngine.PADDLEOCR:
            try:
                from paddleocr import PaddleOCR
                self._reader = PaddleOCR(use_angle_cls=True, lang=self.languages[0], use_gpu=self.gpu)
                self._paddleocr_available = True
            except ImportError:
                logger.warning("PaddleOCR not available")
                self._paddleocr_available = False

    def extract_text(
        self,
        image: np.ndarray,
        preprocess: bool = True,
    ) -> OCRResult:
        """
        Extract text from image using selected OCR engine.

        Args:
            image: Input image (BGR or grayscale)
            preprocess: Whether to apply preprocessing

        Returns:
            OCRResult with extracted text and metadata
        """
        start_time = datetime.now()

        if preprocess:
            image = self._preprocess_image(image)

        if self.engine == OCREngine.TESSERACT and self._tesseract_available:
            result = self._extract_tesseract(image)
        elif self.engine == OCREngine.EASYOCR and hasattr(self, '_easyocr_available') and self._easyocr_available:
            result = self._extract_easyocr(image)
        elif self.engine == OCREngine.PADDLEOCR and hasattr(self, '_paddleocr_available') and self._paddleocr_available:
            result = self._extract_paddleocr(image)
        else:
            # Fallback to basic processing
            result = OCRResult(
                text="",
                confidence=0.0,
                bounding_boxes=[],
                language=self.languages[0],
                processing_time_ms=0.0,
            )

        result.processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        return result

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # Binarize using adaptive threshold
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return cleaned

    def _extract_tesseract(self, image: np.ndarray) -> OCRResult:
        """Extract text using Tesseract."""
        import pytesseract

        # Get detailed data
        data = pytesseract.image_to_data(
            image,
            lang="+".join(self.languages),
            output_type=pytesseract.Output.DICT
        )

        # Extract text and confidence
        texts = []
        confidences = []
        boxes = []

        for i in range(len(data['text'])):
            if data['text'][i].strip():
                texts.append(data['text'][i])
                confidences.append(data['conf'][i])
                boxes.append({
                    'x': data['left'][i],
                    'y': data['top'][i],
                    'w': data['width'][i],
                    'h': data['height'][i],
                    'text': data['text'][i],
                    'conf': data['conf'][i],
                })

        full_text = " ".join(texts)
        avg_conf = np.mean(confidences) if confidences else 0.0

        return OCRResult(
            text=full_text,
            confidence=avg_conf / 100.0,
            bounding_boxes=boxes,
            language=self.languages[0],
            processing_time_ms=0.0,
        )

    def _extract_easyocr(self, image: np.ndarray) -> OCRResult:
        """Extract text using EasyOCR."""
        results = self._reader.readtext(image)

        texts = []
        confidences = []
        boxes = []

        for (bbox, text, conf) in results:
            if text.strip():
                texts.append(text)
                confidences.append(conf)
                # Convert bbox to x,y,w,h
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                boxes.append({
                    'x': int(min(xs)),
                    'y': int(min(ys)),
                    'w': int(max(xs) - min(xs)),
                    'h': int(max(ys) - min(ys)),
                    'text': text,
                    'conf': conf,
                })

        full_text = " ".join(texts)
        avg_conf = np.mean(confidences) if confidences else 0.0

        return OCRResult(
            text=full_text,
            confidence=avg_conf,
            bounding_boxes=boxes,
            language=self.languages[0],
            processing_time_ms=0.0,
        )

    def _extract_paddleocr(self, image: np.ndarray) -> OCRResult:
        """Extract text using PaddleOCR."""
        results = self._reader.ocr(image, cls=True)

        texts = []
        confidences = []
        boxes = []

        if results and results[0]:
            for line in results[0]:
                bbox, (text, conf) = line
                if text.strip():
                    texts.append(text)
                    confidences.append(conf)
                    xs = [p[0] for p in bbox]
                    ys = [p[1] for p in bbox]
                    boxes.append({
                        'x': int(min(xs)),
                        'y': int(min(ys)),
                        'w': int(max(xs) - min(xs)),
                        'h': int(max(ys) - min(ys)),
                        'text': text,
                        'conf': conf,
                    })

        full_text = " ".join(texts)
        avg_conf = np.mean(confidences) if confidences else 0.0

        return OCRResult(
            text=full_text,
            confidence=avg_conf,
            bounding_boxes=boxes,
            language=self.languages[0],
            processing_time_ms=0.0,
        )

    def extract_structured(
        self,
        image: np.ndarray,
        document_type: str = "generic",
    ) -> DocumentExtraction:
        """
        Extract structured data from document based on type.

        Args:
            image: Input image
            document_type: Type of document (upi_payment, bank_statement, aadhaar, pan, etc.)

        Returns:
            DocumentExtraction with structured fields
        """
        # First get raw OCR
        ocr_result = self.extract_text(image)

        # Extract based on document type
        extractors = {
            "upi_payment": self._extract_upi_payment,
            "bank_statement": self._extract_bank_statement,
            "aadhaar": self._extract_aadhaar,
            "pan": self._extract_pan,
            "salary_slip": self._extract_salary_slip,
            "generic": self._extract_generic,
        }

        extractor = extractors.get(document_type, self._extract_generic)
        return extractor(image, ocr_result)

    def _extract_upi_payment(
        self,
        image: np.ndarray,
        ocr_result: OCRResult
    ) -> DocumentExtraction:
        """Extract structured data from UPI payment screenshot."""
        text = ocr_result.text
        boxes = ocr_result.bounding_boxes

        fields = []
        kv_pairs = {}
        warnings = []

        # UPI-specific patterns
        patterns = {
            "amount": [
                r'(?:₹|Rs\.?|INR)\s*([\d,]+\.?\d*)',
                r'Amount[:\s]+(?:₹|Rs\.?|INR)?\s*([\d,]+\.?\d*)',
                r'Paid[:\s]+(?:₹|Rs\.?|INR)?\s*([\d,]+\.?\d*)',
            ],
            "transaction_id": [
                r'(?:TXN|Transaction|Ref|Reference)[:\s#]*([A-Z0-9]{10,})',
                r'\b([A-Z]{2,4}\d{8,})\b',
            ],
            "sender_vpa": [
                r'(?:From|Sender|Payer)[:\s]+([\w.-]+@[\w.-]+)',
                r'([\w.-]+@[\w.-]+)\s+(?:sent|paid)',
            ],
            "receiver_vpa": [
                r'(?:To|Receiver|Payee|Merchant)[:\s]+([\w.-]+@[\w.-]+)',
                r'(?:sent|paid)\s+to\s+([\w.-]+@[\w.-]+)',
            ],
            "date_time": [
                r'(?:Date|Time|On)[:\s]+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}[\s\d:]*)',
                r'(\d{1,2}[-/]\d{1,2}[-/]\d{4}\s+\d{1,2}:\d{2})',
            ],
            "bank_name": [
                r'(?:Bank|via)[:\s]+([A-Za-z\s]+(?:Bank|Ltd))',
            ],
            "status": [
                r'(?:Status|State)[:\s]+(Success|Failed|Pending|Completed)',
                r'\b(Successful|Completed|Failed|Pending)\b',
            ],
        }

        # Extract using patterns
        for field_name, field_patterns in patterns.items():
            for pattern in field_patterns:
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                if matches:
                    match = matches[0]
                    value = match.group(1) if match.groups() else match.group(0)
                    fields.append(StructuredField(
                        field_name=field_name,
                        value=value.strip(),
                        confidence=0.8,
                        bbox=self._find_bbox_for_text(boxes, value),
                        field_type=self._get_field_type(field_name),
                    ))
                    kv_pairs[field_name] = value.strip()
                    break

        # Check for red flags
        if "collect" in text.lower() and "request" in text.lower():
            warnings.append("Collect request detected - verify payment direction")

        if "cashback" in text.lower() or "refund" in text.lower():
            warnings.append("Cashback/refund language detected - potential social engineering")

        return DocumentExtraction(
            document_type="upi_payment",
            raw_text=text,
            structured_fields=fields,
            key_value_pairs=kv_pairs,
            tables=[],
            confidence=ocr_result.confidence,
            warnings=warnings,
        )

    def _extract_bank_statement(
        self,
        image: np.ndarray,
        ocr_result: OCRResult
    ) -> DocumentExtraction:
        """Extract structured data from bank statement."""
        text = ocr_result.text

        fields = []
        kv_pairs = {}
        tables = []
        warnings = []

        # Bank statement patterns
        patterns = {
            "account_number": [
                r'(?:Account|A/c|Acc)[.\s#:]*([\dX*]{8,})',
                r'\b(\d{9,18})\b',
            ],
            "ifsc": [
                r'(?:IFSC|IFS)[:\s]+([A-Z]{4}0[A-Z0-9]{6})',
            ],
            "account_holder": [
                r'(?:Name|Holder)[:\s]+([A-Z\s]{5,})',
            ],
            "opening_balance": [
                r'(?:Opening|Previous)\s*Balance[:\s]+(?:₹|Rs\.?|INR)?\s*([\d,]+\.?\d*)',
            ],
            "closing_balance": [
                r'(?:Closing|Current|Available)\s*Balance[:\s]+(?:₹|Rs\.?|INR)?\s*([\d,]+\.?\d*)',
            ],
            "statement_period": [
                r'(?:Period|From|To)[:\s]+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
            ],
        }

        for field_name, field_patterns in patterns.items():
            for pattern in field_patterns:
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                if matches:
                    match = matches[0]
                    value = match.group(1) if match.groups() else match.group(0)
                    fields.append(StructuredField(
                        field_name=field_name,
                        value=value.strip(),
                        confidence=0.75,
                        bbox=(0, 0, 0, 0),
                        field_type=self._get_field_type(field_name),
                    ))
                    kv_pairs[field_name] = value.strip()
                    break

        # Try to extract transaction table
        table = self._extract_transaction_table(text)
        if table:
            tables.append(table)

        return DocumentExtraction(
            document_type="bank_statement",
            raw_text=text,
            structured_fields=fields,
            key_value_pairs=kv_pairs,
            tables=tables,
            confidence=ocr_result.confidence,
            warnings=warnings,
        )

    def _extract_aadhaar(
        self,
        image: np.ndarray,
        ocr_result: OCRResult
    ) -> DocumentExtraction:
        """Extract structured data from Aadhaar card."""
        text = ocr_result.text

        fields = []
        kv_pairs = {}
        warnings = []

        # Aadhaar patterns
        patterns = {
            "aadhaar_number": [
                r'\b(\d{4}\s\d{4}\s\d{4})\b',
                r'\b(\d{12})\b',
            ],
            "name": [
                r'(?:Name|नाम)[:\s]+([A-Z\s]{3,})',
            ],
            "dob": [
                r'(?:DOB|Date of Birth|जन्म तिथि)[:\s]+(\d{2}/\d{2}/\d{4})',
            ],
            "gender": [
                r'(?:Gender|लिंग)[:\s]+(Male|Female|Male|Female|पुरुष|महिला)',
            ],
            "address": [
                r'(?:Address|पता)[:\s]+(.{10,})',
            ],
        }

        for field_name, field_patterns in patterns.items():
            for pattern in field_patterns:
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                if matches:
                    match = matches[0]
                    value = match.group(1) if match.groups() else match.group(0)
                    fields.append(StructuredField(
                        field_name=field_name,
                        value=value.strip(),
                        confidence=0.8,
                        bbox=(0, 0, 0, 0),
                        field_type=self._get_field_type(field_name),
                    ))
                    kv_pairs[field_name] = value.strip()
                    break

        # Validate Aadhaar number (Verhoeff algorithm)
        if "aadhaar_number" in kv_pairs:
            aadhaar = kv_pairs["aadhaar_number"].replace(" ", "")
            if not self._verify_aadhaar(aadhaar):
                warnings.append("Aadhaar number checksum validation failed")

        return DocumentExtraction(
            document_type="aadhaar",
            raw_text=text,
            structured_fields=fields,
            key_value_pairs=kv_pairs,
            tables=[],
            confidence=ocr_result.confidence,
            warnings=warnings,
        )

    def _extract_pan(
        self,
        image: np.ndarray,
        ocr_result: OCRResult
    ) -> DocumentExtraction:
        """Extract structured data from PAN card."""
        text = ocr_result.text

        fields = []
        kv_pairs = {}
        warnings = []

        patterns = {
            "pan_number": [
                r'\b([A-Z]{5}\d{4}[A-Z])\b',
            ],
            "name": [
                r'(?:Name|नाम)[:\s]+([A-Z\s]{3,})',
            ],
            "father_name": [
                r"(?:Father's Name|पिता का नाम)[:\s]+([A-Z\s]{3,})",
            ],
            "dob": [
                r'(?:DOB|Date of Birth|जन्म तिथि)[:\s]+(\d{2}/\d{2}/\d{4})',
            ],
        }

        for field_name, field_patterns in patterns.items():
            for pattern in field_patterns:
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                if matches:
                    match = matches[0]
                    value = match.group(1) if match.groups() else match.group(0)
                    fields.append(StructuredField(
                        field_name=field_name,
                        value=value.strip(),
                        confidence=0.8,
                        bbox=(0, 0, 0, 0),
                        field_type=self._get_field_type(field_name),
                    ))
                    kv_pairs[field_name] = value.strip()
                    break

        # Validate PAN format
        if "pan_number" in kv_pairs:
            pan = kv_pairs["pan_number"]
            if not re.match(r'^[A-Z]{5}\d{4}[A-Z]$', pan):
                warnings.append("PAN format validation failed")

        return DocumentExtraction(
            document_type="pan",
            raw_text=text,
            structured_fields=fields,
            key_value_pairs=kv_pairs,
            tables=[],
            confidence=ocr_result.confidence,
            warnings=warnings,
        )

    def _extract_salary_slip(
        self,
        image: np.ndarray,
        ocr_result: OCRResult
    ) -> DocumentExtraction:
        """Extract structured data from salary slip."""
        text = ocr_result.text

        fields = []
        kv_pairs = {}
        tables = []
        warnings = []

        patterns = {
            "employee_name": [
                r'(?:Employee|Name)[:\s]+([A-Z\s]{3,})',
            ],
            "employee_id": [
                r'(?:Emp|Employee)\s*(?:ID|Code|No)[:\s#]+([A-Z0-9]{3,})',
            ],
            "company": [
                r'(?:Company|Employer|Organization)[:\s]+([A-Z\s]{3,})',
            ],
            "pan": [
                r'(?:PAN)[:\s]+([A-Z]{5}\d{4}[A-Z])',
            ],
            "basic_salary": [
                r'(?:Basic|Base)\s*(?:Salary|Pay)[:\s]+(?:₹|Rs\.?|INR)?\s*([\d,]+\.?\d*)',
            ],
            "gross_salary": [
                r'(?:Gross|Total)\s*(?:Salary|Earnings)[:\s]+(?:₹|Rs\.?|INR)?\s*([\d,]+\.?\d*)',
            ],
            "net_salary": [
                r'(?:Net|Take Home)\s*(?:Salary|Pay)[:\s]+(?:₹|Rs\.?|INR)?\s*([\d,]+\.?\d*)',
            ],
            "pay_period": [
                r'(?:Period|Month)[:\s]+([A-Za-z]+\s+\d{4})',
            ],
        }

        for field_name, field_patterns in patterns.items():
            for pattern in field_patterns:
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                if matches:
                    match = matches[0]
                    value = match.group(1) if match.groups() else match.group(0)
                    fields.append(StructuredField(
                        field_name=field_name,
                        value=value.strip(),
                        confidence=0.75,
                        bbox=(0, 0, 0, 0),
                        field_type=self._get_field_type(field_name),
                    ))
                    kv_pairs[field_name] = value.strip()
                    break

        # Check for inconsistencies
        if "basic_salary" in kv_pairs and "gross_salary" in kv_pairs:
            basic = float(kv_pairs["basic_salary"].replace(",", ""))
            gross = float(kv_pairs["gross_salary"].replace(",", ""))
            if basic > gross:
                warnings.append("Basic salary exceeds gross salary - possible tampering")

        return DocumentExtraction(
            document_type="salary_slip",
            raw_text=text,
            structured_fields=fields,
            key_value_pairs=kv_pairs,
            tables=tables,
            confidence=ocr_result.confidence,
            warnings=warnings,
        )

    def _extract_generic(
        self,
        image: np.ndarray,
        ocr_result: OCRResult
    ) -> DocumentExtraction:
        """Generic extraction for unknown document types."""
        return DocumentExtraction(
            document_type="generic",
            raw_text=ocr_result.text,
            structured_fields=[],
            key_value_pairs={},
            tables=[],
            confidence=ocr_result.confidence,
            warnings=[],
        )

    def _find_bbox_for_text(
        self,
        boxes: List[Dict],
        text: str
    ) -> Tuple[int, int, int, int]:
        """Find bounding box for specific text in OCR results."""
        for box in boxes:
            if text.lower() in box.get('text', '').lower():
                return (box['x'], box['y'], box['w'], box['h'])
        return (0, 0, 0, 0)

    def _get_field_type(self, field_name: str) -> str:
        """Determine field type from field name."""
        if any(kw in field_name for kw in ["amount", "salary", "balance", "basic", "gross", "net"]):
            return "amount"
        elif any(kw in field_name for kw in ["id", "number", "pan", "aadhaar", "ifsc", "vpa"]):
            return "id"
        elif any(kw in field_name for kw in ["date", "dob", "period", "time"]):
            return "date"
        elif any(kw in field_name for kw in ["name", "holder", "company", "bank"]):
            return "text"
        return "text"

    def _extract_transaction_table(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract transaction table from bank statement text."""
        # Look for transaction lines
        lines = text.split('\n')
        transactions = []

        # Pattern for transaction lines: date, description, debit, credit, balance
        txn_pattern = re.compile(
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\s+(.+?)\s+(?:₹|Rs\.?|INR)?\s*([\d,]+\.?\d*)?\s*(?:₹|Rs\.?|INR)?\s*([\d,]+\.?\d*)?\s*(?:₹|Rs\.?|INR)?\s*([\d,]+\.?\d*)?'
        )

        for line in lines:
            match = txn_pattern.search(line)
            if match:
                transactions.append({
                    "date": match.group(1),
                    "description": match.group(2).strip(),
                    "debit": match.group(3) or "0",
                    "credit": match.group(4) or "0",
                    "balance": match.group(5) or "0",
                })

        if transactions:
            return {
                "type": "transactions",
                "columns": ["date", "description", "debit", "credit", "balance"],
                "rows": transactions,
            }
        return None

    def _verify_aadhaar(self, aadhaar: str) -> bool:
        """Verify Aadhaar number using Verhoeff algorithm."""
        if len(aadhaar) != 12 or not aadhaar.isdigit():
            return False

        # Verhoeff algorithm
        d = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
            [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
            [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
            [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
            [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
            [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
            [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
            [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
            [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        ]

        p = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
            [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
            [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
            [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
            [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
            [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
            [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
        ]

        c = 0
        for i, ch in enumerate(reversed(aadhaar)):
            c = d[c][p[i % 8][int(ch)]]

        return c == 0


def extract_document(
    image_path: str,
    document_type: str = "generic",
    engine: OCREngine = OCREngine.TESSERACT,
) -> DocumentExtraction:
    """
    Convenience function for document extraction.

    Args:
        image_path: Path to document image
        document_type: Type of document
        engine: OCR engine to use

    Returns:
        DocumentExtraction with results
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    processor = OCRProcessor(engine=engine)
    return processor.extract_structured(image, document_type)