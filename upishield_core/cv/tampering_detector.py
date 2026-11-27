"""
Unified Document Tampering Detector.

Integrates document forensics analysis with OCR-based text extraction
for comprehensive document fraud detection.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
import json
from datetime import datetime

from .document_forensics import (
    DocumentForensics,
    ForensicResult,
    TamperingType,
    DocumentType,
)
from .ocr_processor import (
    OCRProcessor,
    DocumentExtraction,
    OCREngine,
    StructuredField,
)

logger = logging.getLogger(__name__)


class TamperingVerdict(Enum):
    """Final tampering verdict."""
    AUTHENTIC = "authentic"
    SUSPICIOUS = "suspicious"
    TAMPERED = "tampered"
    INCONCLUSIVE = "inconclusive"


class RiskLevel(Enum):
    """Risk level assessment."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TamperingIndicator:
    """Individual tampering indicator."""
    indicator_type: str
    description: str
    severity: RiskLevel
    confidence: float
    evidence: Dict[str, Any]
    source: str  # "forensics", "ocr", "cross_validation"


@dataclass
class CrossValidationResult:
    """Result of cross-validating forensics with OCR."""
    check_name: str
    passed: bool
    details: str
    confidence: float
    forensics_finding: Optional[str] = None
    ocr_finding: Optional[str] = None


@dataclass
class TamperingReport:
    """Complete tampering detection report."""
    document_path: str
    document_type: DocumentType
    verdict: TamperingVerdict
    risk_level: RiskLevel
    overall_confidence: float

    # Forensics results
    forensic_result: Optional[ForensicResult] = None

    # OCR results
    ocr_extraction: Optional[DocumentExtraction] = None

    # Tampering indicators
    indicators: List[TamperingIndicator] = field(default_factory=list)

    # Cross-validation
    cross_validations: List[CrossValidationResult] = field(default_factory=list)

    # Summary
    summary: str = ""
    recommendations: List[str] = field(default_factory=list)

    # Metadata
    processing_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class TamperingDetector:
    """
    Unified tampering detector combining document forensics and OCR analysis.

    Detects:
    - Visual tampering (ELA, copy-move, noise inconsistencies)
    - Textual inconsistencies (OCR extracted data vs visual evidence)
    - Document-specific fraud patterns
    - Cross-validation between visual and textual analysis
    """

    def __init__(
        self,
        forensics_analyzer: Optional[DocumentForensics] = None,
        ocr_processor: Optional[OCRProcessor] = None,
        ocr_engine: OCREngine = OCREngine.TESSERACT,
    ):
        self.forensics = forensics_analyzer or DocumentForensics()
        self.ocr = ocr_processor or OCRProcessor(engine=ocr_engine)

    def analyze(
        self,
        image_path: str,
        document_type: DocumentType = DocumentType.GENERIC,
        expected_data: Optional[Dict[str, Any]] = None,
    ) -> TamperingReport:
        """
        Perform complete tampering analysis.

        Args:
            image_path: Path to document image
            document_type: Type of document for specialized checks
            expected_data: Optional expected data for validation (e.g., known transaction amount)

        Returns:
            Complete TamperingReport
        """
        start_time = datetime.now()
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Run forensics analysis
        logger.info(f"Running forensic analysis on {image_path}")
        forensic_result = self.forensics.analyze(image, document_type)

        # Run OCR extraction
        logger.info(f"Running OCR extraction on {image_path}")
        ocr_extraction = self.ocr.extract_structured(image, document_type.value)

        # Generate tampering indicators
        indicators = []
        indicators.extend(self._generate_forensics_indicators(forensic_result))
        indicators.extend(self._generate_ocr_indicators(ocr_extraction, document_type))
        indicators.extend(self._generate_cross_validation_indicators(
            forensic_result, ocr_extraction, document_type, expected_data
        ))

        # Run cross-validations
        cross_validations = self._run_cross_validations(
            forensic_result, ocr_extraction, document_type, expected_data
        )

        # Determine verdict
        verdict, risk_level, confidence = self._determine_verdict(
            indicators, cross_validations, forensic_result
        )

        # Generate summary and recommendations
        summary = self._generate_summary(verdict, risk_level, indicators)
        recommendations = self._generate_recommendations(verdict, risk_level, indicators)

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return TamperingReport(
            document_path=image_path,
            document_type=document_type,
            verdict=verdict,
            risk_level=risk_level,
            overall_confidence=confidence,
            forensic_result=forensic_result,
            ocr_extraction=ocr_extraction,
            indicators=indicators,
            cross_validations=cross_validations,
            summary=summary,
            recommendations=recommendations,
            processing_time_ms=processing_time,
        )

    def _generate_forensics_indicators(
        self,
        result: ForensicResult
    ) -> List[TamperingIndicator]:
        """Generate indicators from forensics results."""
        indicators = []

        # Map TamperingType to severity
        severity_map = {
            TamperingType.COPY_MOVE: RiskLevel.CRITICAL,
            TamperingType.SPLICING: RiskLevel.CRITICAL,
            TamperingType.RETOUCHING: RiskLevel.HIGH,
            TamperingType.MORPHING: RiskLevel.CRITICAL,
            TamperingType.METADATA_MISMATCH: RiskLevel.LOW,
            TamperingType.FONT_INCONSISTENCY: RiskLevel.HIGH,
            TamperingType.ELA_ANOMALY: RiskLevel.HIGH,
            TamperingType.NOISE_INCONSISTENCY: RiskLevel.MEDIUM,
            TamperingType.COLOR_INCONSISTENCY: RiskLevel.MEDIUM,
            TamperingType.EDGE_INCONSISTENCY: RiskLevel.MEDIUM,
        }

        # Generate indicators from tampering types and anomaly regions
        for tampering_type in result.tampering_types:
            severity = severity_map.get(tampering_type, RiskLevel.MEDIUM)
            confidence = result.confidence_scores.get(tampering_type.value, 0.5)

            # Find related anomaly regions
            related_regions = [r for r in result.anomaly_regions if r.get("type") == tampering_type.value]

            indicators.append(TamperingIndicator(
                indicator_type=tampering_type.value,
                description=f"Detected {tampering_type.value.replace('_', ' ')} in document",
                severity=severity,
                confidence=confidence,
                evidence={
                    "anomaly_regions": related_regions,
                    "confidence_scores": result.confidence_scores,
                },
                source="forensics",
            ))

        return indicators

    def _generate_ocr_indicators(
        self,
        extraction: DocumentExtraction,
        document_type: DocumentType
    ) -> List[TamperingIndicator]:
        """Generate indicators from OCR extraction results."""
        indicators = []

        # Low OCR confidence
        if extraction.confidence < 0.5:
            indicators.append(TamperingIndicator(
                indicator_type="low_ocr_confidence",
                description=f"Low OCR confidence ({extraction.confidence:.1%}) - possible image quality issues or obfuscation",
                severity=RiskLevel.MEDIUM,
                confidence=1.0 - extraction.confidence,
                evidence={"ocr_confidence": extraction.confidence},
                source="ocr",
            ))

        # Warnings from extraction
        for warning in extraction.warnings:
            indicators.append(TamperingIndicator(
                indicator_type="ocr_warning",
                description=warning,
                severity=RiskLevel.HIGH,
                confidence=0.8,
                evidence={},
                source="ocr",
            ))

        # Document-specific checks
        if document_type == DocumentType.UPI_PAYMENT:
            indicators.extend(self._check_upi_payment_ocr(extraction))
        elif document_type == DocumentType.BANK_STATEMENT:
            indicators.extend(self._check_bank_statement_ocr(extraction))
        elif document_type == DocumentType.AADHAAR:
            indicators.extend(self._check_aadhaar_ocr(extraction))
        elif document_type == DocumentType.PAN:
            indicators.extend(self._check_pan_ocr(extraction))

        return indicators

    def _check_upi_payment_ocr(self, extraction: DocumentExtraction) -> List[TamperingIndicator]:
        """Check UPI payment specific OCR indicators."""
        indicators = []
        kv = extraction.key_value_pairs

        # Check for collect request masquerading as payment
        if "status" in kv and kv["status"].lower() in ["success", "completed"]:
            if "collect" in extraction.raw_text.lower() or "request" in extraction.raw_text.lower():
                indicators.append(TamperingIndicator(
                    indicator_type="collect_request_spoof",
                    description="Collect request shown as successful payment",
                    severity=RiskLevel.CRITICAL,
                    confidence=0.9,
                    evidence={"status": kv.get("status"), "raw_text_snippet": extraction.raw_text[:200]},
                    source="ocr",
                ))

        # Check amount consistency
        if "amount" in kv:
            try:
                amount = float(kv["amount"].replace(",", ""))
                if amount > 100000:  # High value transaction
                    indicators.append(TamperingIndicator(
                        indicator_type="high_value_transaction",
                        description=f"High value transaction: ₹{amount:,.2f}",
                        severity=RiskLevel.HIGH,
                        confidence=0.7,
                        evidence={"amount": amount},
                        source="ocr",
                    ))
            except ValueError:
                pass

        # Check for suspicious VPA patterns
        for vpa_field in ["sender_vpa", "receiver_vpa"]:
            if vpa_field in kv:
                vpa = kv[vpa_field]
                if self._is_suspicious_vpa(vpa):
                    indicators.append(TamperingIndicator(
                        indicator_type="suspicious_vpa",
                        description=f"Suspicious VPA pattern: {vpa}",
                        severity=RiskLevel.HIGH,
                        confidence=0.8,
                        evidence={vpa_field: vpa},
                        source="ocr",
                    ))

        return indicators

    def _is_suspicious_vpa(self, vpa: str) -> bool:
        """Check if VPA has suspicious patterns."""
        suspicious_patterns = [
            r'.*@paytm$',
            r'.*@phonepe$',
            r'.*@gpay$',
            r'.*@ybl$',
            r'.*@okicici$',
            r'.*@okaxis$',
            r'.*@oksbi$',
            r'.*@okhdfc$',
            r'.*\d{10}@',  # Phone number as VPA
            r'.*[0-9]{4,}@',  # Many digits
        ]
        vpa_lower = vpa.lower()
        return any(re.search(p, vpa_lower) for p in suspicious_patterns)

    def _check_bank_statement_ocr(self, extraction: DocumentExtraction) -> List[TamperingIndicator]:
        """Check bank statement specific OCR indicators."""
        indicators = []
        kv = extraction.key_value_pairs

        # Check for balance inconsistencies
        if "opening_balance" in kv and "closing_balance" in kv:
            try:
                opening = float(kv["opening_balance"].replace(",", ""))
                closing = float(kv["closing_balance"].replace(",", ""))
                if closing < 0:
                    indicators.append(TamperingIndicator(
                        indicator_type="negative_closing_balance",
                        description="Negative closing balance detected",
                        severity=RiskLevel.HIGH,
                        confidence=0.8,
                        evidence={"opening": opening, "closing": closing},
                        source="ocr",
                    ))
            except ValueError:
                pass

        # Check transaction table for anomalies
        for table in extraction.tables:
            if table.get("type") == "transactions":
                for row in table.get("rows", []):
                    try:
                        debit = float(row.get("debit", "0").replace(",", ""))
                        credit = float(row.get("credit", "0").replace(",", ""))
                        balance = float(row.get("balance", "0").replace(",", ""))

                        # Check for round number transactions (potential fabrication)
                        if debit > 0 and debit % 1000 == 0 and debit > 10000:
                            indicators.append(TamperingIndicator(
                                indicator_type="round_number_transaction",
                                description=f"Suspicious round-number debit: ₹{debit:,.2f}",
                                severity=RiskLevel.MEDIUM,
                                confidence=0.6,
                                evidence=row,
                                source="ocr",
                            ))
                    except ValueError:
                        continue

        return indicators

    def _check_aadhaar_ocr(self, extraction: DocumentExtraction) -> List[TamperingIndicator]:
        """Check Aadhaar specific OCR indicators."""
        indicators = []
        kv = extraction.key_value_pairs

        # Already validated in OCR processor, but double-check
        if "aadhaar_number" in kv:
            aadhaar = kv["aadhaar_number"].replace(" ", "")
            if len(aadhaar) != 12 or not aadhaar.isdigit():
                indicators.append(TamperingIndicator(
                    indicator_type="invalid_aadhaar_format",
                    description="Aadhaar number format invalid",
                    severity=RiskLevel.CRITICAL,
                    confidence=0.95,
                    evidence={"aadhaar": aadhaar},
                    source="ocr",
                ))

        return indicators

    def _check_pan_ocr(self, extraction: DocumentExtraction) -> List[TamperingIndicator]:
        """Check PAN specific OCR indicators."""
        indicators = []
        kv = extraction.key_value_pairs

        if "pan_number" in kv:
            pan = kv["pan_number"]
            if not re.match(r'^[A-Z]{5}\d{4}[A-Z]$', pan):
                indicators.append(TamperingIndicator(
                    indicator_type="invalid_pan_format",
                    description="PAN number format invalid",
                    severity=RiskLevel.CRITICAL,
                    confidence=0.95,
                    evidence={"pan": pan},
                    source="ocr",
                ))

        return indicators

    def _generate_cross_validation_indicators(
        self,
        forensic_result: ForensicResult,
        ocr_extraction: DocumentExtraction,
        document_type: DocumentType,
        expected_data: Optional[Dict[str, Any]]
    ) -> List[TamperingIndicator]:
        """Generate indicators from cross-validating forensics and OCR."""
        indicators = []

        # Check if ELA anomaly exists and document has text
        if TamperingType.ELA_ANOMALY in forensic_result.tampering_types:
            if len(ocr_extraction.raw_text.strip()) > 10:
                indicators.append(TamperingIndicator(
                    indicator_type="ela_text_region_tampering",
                    description="ELA inconsistency detected - possible text modification",
                    severity=RiskLevel.CRITICAL,
                    confidence=0.85,
                    evidence={
                        "ela_confidence": forensic_result.confidence_scores.get("ela", 0),
                        "ocr_text_length": len(ocr_extraction.raw_text),
                    },
                    source="cross_validation",
                ))

        # Check copy-move forgery with text
        if TamperingType.COPY_MOVE in forensic_result.tampering_types:
            if len(ocr_extraction.raw_text.strip()) > 10:
                indicators.append(TamperingIndicator(
                    indicator_type="copy_move_text_duplication",
                    description="Copy-move forgery detected - possible duplicated text regions",
                    severity=RiskLevel.CRITICAL,
                    confidence=0.9,
                    evidence={
                        "copymove_confidence": forensic_result.confidence_scores.get("copymove", 0),
                        "ocr_text_length": len(ocr_extraction.raw_text),
                    },
                    source="cross_validation",
                ))

        # Validate expected data against OCR
        if expected_data:
            for key, expected_value in expected_data.items():
                if key in ocr_extraction.key_value_pairs:
                    ocr_value = ocr_extraction.key_value_pairs[key]
                    if str(expected_value).strip() != str(ocr_value).strip():
                        indicators.append(TamperingIndicator(
                            indicator_type="data_mismatch",
                            description=f"Data mismatch for {key}: expected '{expected_value}', found '{ocr_value}'",
                            severity=RiskLevel.HIGH,
                            confidence=0.9,
                            evidence={"field": key, "expected": expected_value, "found": ocr_value},
                            source="cross_validation",
                        ))

        # Check for font inconsistencies
        if TamperingType.FONT_INCONSISTENCY in forensic_result.tampering_types:
            indicators.append(TamperingIndicator(
                indicator_type="font_inconsistency_confirmed",
                description="Font inconsistency detected in document",
                severity=RiskLevel.HIGH,
                confidence=0.85,
                evidence={"font_confidence": forensic_result.confidence_scores.get("font", 0)},
                source="cross_validation",
            ))

        return indicators

    def _run_cross_validations(
        self,
        forensic_result: ForensicResult,
        ocr_extraction: DocumentExtraction,
        document_type: DocumentType,
        expected_data: Optional[Dict[str, Any]]
    ) -> List[CrossValidationResult]:
        """Run cross-validation checks between forensics and OCR."""
        validations = []

        # Validation 1: Document structure consistency
        validations.append(CrossValidationResult(
            check_name="document_structure_consistency",
            passed=len(ocr_extraction.raw_text.strip()) > 0,
            details="Document has extractable text content" if ocr_extraction.raw_text.strip() else "No text extracted - possible blank or image-only document",
            confidence=0.9 if ocr_extraction.raw_text.strip() else 0.3,
            forensics_finding=f"Tampering types: {len(forensic_result.tampering_types)}",
            ocr_finding=f"OCR confidence: {ocr_extraction.confidence:.2f}",
        ))

        # Validation 2: Visual-textual alignment
        high_conf_types = [t for t in forensic_result.tampering_types
                          if forensic_result.confidence_scores.get(t.value, 0) > 0.7]
        validations.append(CrossValidationResult(
            check_name="visual_textual_alignment",
            passed=len(high_conf_types) == 0,
            details=f"Found {len(high_conf_types)} high-confidence tampering types" if high_conf_types else "No high-confidence tampering detected",
            confidence=0.8,
            forensics_finding=f"{len(high_conf_types)} high-confidence types",
            ocr_finding=f"Text length: {len(ocr_extraction.raw_text)} chars",
        ))

        # Validation 3: Expected data match
        if expected_data:
            matches = 0
            mismatches = 0
            for key, expected in expected_data.items():
                if key in ocr_extraction.key_value_pairs:
                    if str(expected).strip() == str(ocr_extraction.key_value_pairs[key]).strip():
                        matches += 1
                    else:
                        mismatches += 1
            validations.append(CrossValidationResult(
                check_name="expected_data_validation",
                passed=mismatches == 0,
                details=f"{matches} fields match, {mismatches} mismatch" if mismatches else f"All {matches} expected fields match",
                confidence=0.95 if mismatches == 0 else 0.4,
                forensics_finding=None,
                ocr_finding=f"Matches: {matches}, Mismatches: {mismatches}",
            ))

        # Validation 4: Metadata consistency
        metadata_anomaly = TamperingType.METADATA_MISMATCH in forensic_result.tampering_types
        validations.append(CrossValidationResult(
            check_name="metadata_consistency",
            passed=not metadata_anomaly,
            details="Metadata mismatch detected" if metadata_anomaly else "Metadata consistent",
            confidence=0.7,
            forensics_finding=f"Metadata mismatch: {metadata_anomaly}",
            ocr_finding=None,
        ))

        # Validation 5: Document-specific validation
        doc_validation = self._validate_document_specific(forensic_result, ocr_extraction, document_type)
        validations.append(doc_validation)

        return validations

    def _validate_document_specific(
        self,
        forensic_result: ForensicResult,
        ocr_extraction: DocumentExtraction,
        document_type: DocumentType
    ) -> CrossValidationResult:
        """Document-type specific cross-validation."""
        if document_type == DocumentType.UPI_PAYMENT:
            # Check for critical UPI indicators (using tampering types that might be added by doc-specific checks)
            critical_types = [
                TamperingType.RETOUCHING,  # Could indicate amount tampering
                TamperingType.SPLICING,    # Could indicate VPA mismatch
            ]
            critical_count = sum(1 for t in critical_types if t in forensic_result.tampering_types)
            ocr_warnings = len(ocr_extraction.warnings)

            passed = critical_count == 0 and ocr_warnings == 0
            return CrossValidationResult(
                check_name="upi_payment_validation",
                passed=passed,
                details=f"Forensics: {critical_count} critical tampering types, OCR: {ocr_warnings} warnings",
                confidence=0.9 if passed else 0.3,
                forensics_finding=f"Critical tampering types: {critical_count}",
                ocr_finding=f"Warnings: {ocr_warnings}",
            )

        elif document_type == DocumentType.AADHAAR:
            # Check for security feature anomalies
            security_types = [
                TamperingType.RETOUCHING,  # Could indicate hologram issues
            ]
            security_count = sum(1 for t in security_types if t in forensic_result.tampering_types)
            aadhaar_valid = "aadhaar_number" in ocr_extraction.key_value_pairs

            passed = security_count == 0 and aadhaar_valid
            return CrossValidationResult(
                check_name="aadhaar_validation",
                passed=passed,
                details=f"Security feature issues: {security_count}, Aadhaar extracted: {aadhaar_valid}",
                confidence=0.9 if passed else 0.2,
                forensics_finding=f"Security anomalies: {security_count}",
                ocr_finding=f"Aadhaar number found: {aadhaar_valid}",
            )

        elif document_type == DocumentType.PAN:
            pan_valid = "pan_number" in ocr_extraction.key_value_pairs
            passed = pan_valid
            return CrossValidationResult(
                check_name="pan_validation",
                passed=passed,
                details=f"PAN number extracted: {pan_valid}",
                confidence=0.85 if passed else 0.2,
                forensics_finding=None,
                ocr_finding=f"PAN found: {pan_valid}",
            )

        return CrossValidationResult(
            check_name="generic_validation",
            passed=True,
            details="Generic document - basic validation passed",
            confidence=0.7,
        )

    def _determine_verdict(
        self,
        indicators: List[TamperingIndicator],
        cross_validations: List[CrossValidationResult],
        forensic_result: ForensicResult
    ) -> Tuple[TamperingVerdict, RiskLevel, float]:
        """Determine final verdict from all indicators."""
        # Count indicators by severity
        critical_count = sum(1 for i in indicators if i.severity == RiskLevel.CRITICAL)
        high_count = sum(1 for i in indicators if i.severity == RiskLevel.HIGH)
        medium_count = sum(1 for i in indicators if i.severity == RiskLevel.MEDIUM)

        # Count failed cross-validations
        failed_validations = sum(1 for v in cross_validations if not v.passed)

        # Calculate weighted score
        score = 0
        score += critical_count * 10
        score += high_count * 5
        score += medium_count * 2
        score += failed_validations * 3

        # Forensics confidence factor - use average confidence or 0 if no tampering
        if forensic_result.confidence_scores:
            avg_confidence = sum(forensic_result.confidence_scores.values()) / len(forensic_result.confidence_scores)
        else:
            avg_confidence = 1.0  # No tampering detected = high confidence
        forensics_factor = 1.0 - avg_confidence
        score += forensics_factor * 5

        # Determine verdict
        if score >= 20 or critical_count >= 2:
            verdict = TamperingVerdict.TAMPERED
            risk_level = RiskLevel.CRITICAL
            confidence = min(0.95, 0.7 + score * 0.02)
        elif score >= 10 or critical_count >= 1:
            verdict = TamperingVerdict.SUSPICIOUS
            risk_level = RiskLevel.HIGH
            confidence = min(0.9, 0.6 + score * 0.02)
        elif score >= 5 or high_count >= 1 or failed_validations >= 1:
            verdict = TamperingVerdict.SUSPICIOUS
            risk_level = RiskLevel.MEDIUM
            confidence = min(0.8, 0.5 + score * 0.02)
        elif score > 0:
            verdict = TamperingVerdict.SUSPICIOUS
            risk_level = RiskLevel.LOW
            confidence = 0.5 + score * 0.02
        else:
            verdict = TamperingVerdict.AUTHENTIC
            risk_level = RiskLevel.LOW
            confidence = forensic_result.overall_confidence

        return verdict, risk_level, confidence

    def _generate_summary(
        self,
        verdict: TamperingVerdict,
        risk_level: RiskLevel,
        indicators: List[TamperingIndicator]
    ) -> str:
        """Generate human-readable summary."""
        summary_parts = [
            f"Document verdict: {verdict.value.upper()}",
            f"Risk level: {risk_level.value.upper()}",
            f"Total indicators: {len(indicators)}",
        ]

        critical = [i for i in indicators if i.severity == RiskLevel.CRITICAL]
        high = [i for i in indicators if i.severity == RiskLevel.HIGH]

        if critical:
            summary_parts.append(f"CRITICAL: {len(critical)} critical issue(s) - {', '.join(c.indicator_type for c in critical[:3])}")
        if high:
            summary_parts.append(f"HIGH: {len(high)} high-severity issue(s)")

        return ". ".join(summary_parts)

    def _generate_recommendations(
        self,
        verdict: TamperingVerdict,
        risk_level: RiskLevel,
        indicators: List[TamperingIndicator]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        if verdict == TamperingVerdict.TAMPERED:
            recommendations.append("REJECT: Document shows clear signs of tampering - do not accept")
            recommendations.append("Report to fraud investigation team immediately")
            recommendations.append("Request original document from user")
        elif verdict == TamperingVerdict.SUSPICIOUS:
            if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                recommendations.append("HOLD: Document requires manual review before acceptance")
                recommendations.append("Request additional verification (video call, original document)")
            else:
                recommendations.append("REVIEW: Document has minor anomalies - verify with user")
                recommendations.append("Cross-check with transaction records if applicable")
        else:
            recommendations.append("ACCEPT: Document appears authentic")
            recommendations.append("Standard processing recommended")

        # Specific recommendations based on indicators
        indicator_types = {i.indicator_type for i in indicators}
        if "amount_tampering" in indicator_types or "ela_text_region_tampering" in indicator_types:
            recommendations.append("Verify transaction amount with bank/UPI records")
        if "vpa_mismatch" in indicator_types or "suspicious_vpa" in indicator_types:
            recommendations.append("Verify VPA with sender/receiver directly")
        if "collect_request_spoof" in indicator_types:
            recommendations.append("WARN USER: This appears to be a collect request, not a payment received")
        if "copy_move_forgery" in indicator_types or "copy_move_text_duplication" in indicator_types:
            recommendations.append("Document contains duplicated regions - likely fabricated")

        return recommendations


def detect_tampering(
    image_path: str,
    document_type: DocumentType = DocumentType.GENERIC,
    expected_data: Optional[Dict[str, Any]] = None,
    ocr_engine: OCREngine = OCREngine.TESSERACT,
) -> TamperingReport:
    """
    Convenience function for tampering detection.

    Args:
        image_path: Path to document image
        document_type: Type of document
        expected_data: Optional expected data for validation
        ocr_engine: OCR engine to use

    Returns:
        TamperingReport with complete analysis
    """
    detector = TamperingDetector(ocr_engine=ocr_engine)
    return detector.analyze(image_path, document_type, expected_data)


# Add re import for pattern matching
import re