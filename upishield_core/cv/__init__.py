"""
Computer Vision Module for ScamShield AI.

Provides document forensics, OCR processing, and tampering detection
for fraud detection in financial and KYC documents.
"""

from .document_forensics import (
    DocumentForensics,
    ForensicResult,
    TamperingType,
    DocumentType,
)

from .ocr_processor import (
    OCRProcessor,
    DocumentExtraction,
    StructuredField,
    OCRResult,
    OCREngine,
    extract_document,
)

from .tampering_detector import (
    TamperingDetector,
    TamperingReport,
    TamperingIndicator,
    CrossValidationResult,
    TamperingVerdict,
    RiskLevel,
    detect_tampering,
)

__all__ = [
    # Document Forensics
    "DocumentForensics",
    "ForensicResult",
    "TamperingType",
    "DocumentType",
    # OCR Processor
    "OCRProcessor",
    "DocumentExtraction",
    "StructuredField",
    "OCRResult",
    "OCREngine",
    "extract_document",
    # Tampering Detector
    "TamperingDetector",
    "TamperingReport",
    "TamperingIndicator",
    "CrossValidationResult",
    "TamperingVerdict",
    "RiskLevel",
    "detect_tampering",
]

__version__ = "1.0.0"