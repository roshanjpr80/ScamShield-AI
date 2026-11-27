"""
Document Forensics Module for Fraud Detection.

Provides computer vision based analysis for detecting document tampering,
forgery, and manipulation in KYC documents, payment screenshots, and other
financial documents.
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

logger = logging.getLogger(__name__)


class TamperingType(Enum):
    """Types of document tampering detected."""
    COPY_MOVE = "copy_move"
    SPLICING = "splicing"
    RETOUCHING = "retouching"
    MORPHING = "morphing"
    METADATA_MISMATCH = "metadata_mismatch"
    FONT_INCONSISTENCY = "font_inconsistency"
    ELA_ANOMALY = "ela_anomaly"
    NOISE_INCONSISTENCY = "noise_inconsistency"
    COLOR_INCONSISTENCY = "color_inconsistency"
    EDGE_INCONSISTENCY = "edge_inconsistency"


class DocumentType(Enum):
    """Supported document types for analysis."""
    AADHAAR = "aadhaar"
    PAN = "pan"
    PASSPORT = "passport"
    BANK_STATEMENT = "bank_statement"
    SALARY_SLIP = "salary_slip"
    PAYMENT_SCREENSHOT = "payment_screenshot"
    GENERIC = "generic"


@dataclass
class ForensicResult:
    """Result of forensic analysis."""
    tampering_detected: bool
    tampering_types: List[TamperingType]
    confidence_scores: Dict[str, float]
    anomaly_regions: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    visualizations: Dict[str, np.ndarray] = field(default_factory=dict)
    analysis_time_ms: float = 0.0


@dataclass
class ELAResult:
    """Error Level Analysis result."""
    ela_image: np.ndarray
    mean_ela: float
    max_ela: float
    suspicious_regions: List[Tuple[int, int, int, int]]  # x, y, w, h
    compression_quality: int


@dataclass
class NoiseAnalysisResult:
    """Noise pattern analysis result."""
    noise_map: np.ndarray
    noise_consistency_score: float
    inconsistent_regions: List[Tuple[int, int, int, int]]
    sensor_noise_profile: Dict[str, float]


@dataclass
class CopyMoveResult:
    """Copy-move forgery detection result."""
    matches: List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]  # (src_x, src_y), (dst_x, dst_y), (w, h)
    similarity_scores: List[float]
    forged_mask: np.ndarray


class DocumentForensics:
    """
    Main class for document forensic analysis.

    Detects various types of document tampering including:
    - Copy-move forgery (cloned regions)
    - Splicing (pasted content from other images)
    - Retouching (local edits)
    - Metadata inconsistencies
    - Compression artifacts (ELA)
    - Noise pattern inconsistencies
    """

    def __init__(
        self,
        ela_quality: int = 90,
        copy_move_block_size: int = 16,
        copy_move_threshold: float = 0.9,
        noise_patch_size: int = 32,
        min_tampering_area: int = 100,
    ):
        self.ela_quality = ela_quality
        self.copy_move_block_size = copy_move_block_size
        self.copy_move_threshold = copy_move_threshold
        self.noise_patch_size = noise_patch_size
        self.min_tampering_area = min_tampering_area

    def analyze(
        self,
        image_path: str,
        document_type: DocumentType = DocumentType.GENERIC,
        expected_metadata: Optional[Dict[str, Any]] = None,
    ) -> ForensicResult:
        """
        Perform comprehensive forensic analysis on a document image.

        Args:
            image_path: Path to the image file
            document_type: Type of document for specialized checks
            expected_metadata: Expected metadata for validation

        Returns:
            ForensicResult with all detected anomalies
        """
        start_time = datetime.now()

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        original = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        tampering_types = []
        confidence_scores = {}
        anomaly_regions = []
        metadata = {}
        visualizations = {}

        # 1. Error Level Analysis (ELA)
        ela_result = self._perform_ela(image)
        visualizations["ela"] = ela_result.ela_image
        if ela_result.suspicious_regions:
            tampering_types.append(TamperingType.ELA_ANOMALY)
            confidence_scores["ela"] = min(ela_result.max_ela / 100.0, 1.0)
            for x, y, w, h in ela_result.suspicious_regions:
                anomaly_regions.append({
                    "type": "ela_anomaly",
                    "bbox": [x, y, w, h],
                    "confidence": confidence_scores["ela"]
                })
        metadata["ela_mean"] = ela_result.mean_ela
        metadata["ela_max"] = ela_result.max_ela
        metadata["compression_quality"] = ela_result.compression_quality

        # 2. Noise Analysis
        noise_result = self._analyze_noise(gray)
        visualizations["noise_map"] = noise_result.noise_map
        if noise_result.inconsistent_regions:
            tampering_types.append(TamperingType.NOISE_INCONSISTENCY)
            confidence_scores["noise"] = 1.0 - noise_result.noise_consistency_score
            for x, y, w, h in noise_result.inconsistent_regions:
                anomaly_regions.append({
                    "type": "noise_inconsistency",
                    "bbox": [x, y, w, h],
                    "confidence": confidence_scores["noise"]
                })
        metadata["noise_consistency"] = noise_result.noise_consistency_score

        # 3. Copy-Move Detection
        copymove_result = self._detect_copy_move(gray)
        visualizations["copymove_mask"] = copymove_result.forged_mask
        if copymove_result.matches:
            tampering_types.append(TamperingType.COPY_MOVE)
            confidence_scores["copymove"] = max(copymove_result.similarity_scores) if copymove_result.similarity_scores else 0.0
            for (src_x, src_y), (dst_x, dst_y), (w, h) in copymove_result.matches:
                anomaly_regions.append({
                    "type": "copy_move",
                    "source_bbox": [src_x, src_y, w, h],
                    "dest_bbox": [dst_x, dst_y, w, h],
                    "confidence": confidence_scores["copymove"]
                })

        # 4. Color Consistency Analysis
        color_result = self._analyze_color_consistency(image)
        visualizations["color_anomaly"] = color_result.get("anomaly_map", np.zeros_like(gray))
        if color_result.get("inconsistent_regions"):
            tampering_types.append(TamperingType.COLOR_INCONSISTENCY)
            confidence_scores["color"] = color_result.get("max_inconsistency", 0.0)
            for x, y, w, h in color_result["inconsistent_regions"]:
                anomaly_regions.append({
                    "type": "color_inconsistency",
                    "bbox": [x, y, w, h],
                    "confidence": confidence_scores["color"]
                })

        # 5. Edge Consistency Analysis
        edge_result = self._analyze_edge_consistency(gray)
        visualizations["edge_anomaly"] = edge_result.get("anomaly_map", np.zeros_like(gray))
        if edge_result.get("inconsistent_regions"):
            tampering_types.append(TamperingType.EDGE_INCONSISTENCY)
            confidence_scores["edge"] = edge_result.get("max_inconsistency", 0.0)
            for x, y, w, h in edge_result["inconsistent_regions"]:
                anomaly_regions.append({
                    "type": "edge_inconsistency",
                    "bbox": [x, y, w, h],
                    "confidence": confidence_scores["edge"]
                })

        # 6. Metadata Analysis
        if expected_metadata:
            meta_result = self._analyze_metadata(image_path, expected_metadata)
            metadata["metadata_analysis"] = meta_result
            if meta_result.get("mismatches"):
                tampering_types.append(TamperingType.METADATA_MISMATCH)
                confidence_scores["metadata"] = len(meta_result["mismatches"]) / max(len(expected_metadata), 1)

        # 7. Document-specific checks
        doc_result = self._document_specific_checks(image, document_type)
        metadata["document_specific"] = doc_result
        for check_name, check_result in doc_result.items():
            if check_result.get("anomaly"):
                tampering_type = TamperingType[check_result.get("tampering_type", "RETOUCHING").upper()]
                if tampering_type not in tampering_types:
                    tampering_types.append(tampering_type)
                confidence_scores[check_name] = check_result.get("confidence", 0.0)

        analysis_time = (datetime.now() - start_time).total_seconds() * 1000

        return ForensicResult(
            tampering_detected=len(tampering_types) > 0,
            tampering_types=tampering_types,
            confidence_scores=confidence_scores,
            anomaly_regions=anomaly_regions,
            metadata=metadata,
            visualizations=visualizations,
            analysis_time_ms=analysis_time,
        )

    def _perform_ela(self, image: np.ndarray) -> ELAResult:
        """Perform Error Level Analysis to detect compression artifacts."""
        # Save as JPEG with specified quality
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, image, [cv2.IMWRITE_JPEG_QUALITY, self.ela_quality])
            compressed = cv2.imread(tmp.name)

        # Calculate difference
        diff = cv2.absdiff(image.astype(np.float32), compressed.astype(np.float32))
        diff = np.mean(diff, axis=2)

        # Normalize for visualization
        if diff.max() > 0:
            ela_normalized = (diff / diff.max() * 255).astype(np.uint8)
        else:
            ela_normalized = np.zeros_like(diff, dtype=np.uint8)

        # Apply colormap for visualization
        ela_colored = cv2.applyColorMap(ela_normalized, cv2.COLORMAP_JET)

        # Find suspicious regions (high ELA values)
        threshold = np.mean(diff) + 2 * np.std(diff)
        _, binary = cv2.threshold(diff.astype(np.uint8), threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        suspicious_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_tampering_area:
                x, y, w, h = cv2.boundingRect(contour)
                suspicious_regions.append((x, y, w, h))

        return ELAResult(
            ela_image=ela_colored,
            mean_ela=float(np.mean(diff)),
            max_ela=float(np.max(diff)),
            suspicious_regions=suspicious_regions,
            compression_quality=self.ela_quality,
        )

    def _analyze_noise(self, gray: np.ndarray) -> NoiseAnalysisResult:
        """Analyze sensor noise patterns for inconsistencies."""
        # Use wavelet-based noise estimation
        # Simplified: use local variance as noise proxy
        kernel_size = self.noise_patch_size
        h, w = gray.shape

        # Calculate local variance in patches
        noise_map = np.zeros((h, w), dtype=np.float32)
        pad = kernel_size // 2
        padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REFLECT)

        for i in range(h):
            for j in range(w):
                patch = padded[i:i+kernel_size, j:j+kernel_size]
                noise_map[i, j] = np.var(patch)

        # Normalize
        noise_map_norm = cv2.normalize(noise_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Check consistency - compare quadrants
        mid_h, mid_w = h // 2, w // 2
        quadrants = [
            noise_map[0:mid_h, 0:mid_w],
            noise_map[0:mid_h, mid_w:w],
            noise_map[mid_h:h, 0:mid_w],
            noise_map[mid_h:h, mid_w:w],
        ]

        quadrant_means = [np.mean(q) for q in quadrants]
        quadrant_stds = [np.std(q) for q in quadrants]

        # Consistency score based on variance of quadrant means
        mean_consistency = 1.0 - (np.std(quadrant_means) / (np.mean(quadrant_means) + 1e-6))
        std_consistency = 1.0 - (np.std(quadrant_stds) / (np.mean(quadrant_stds) + 1e-6))
        consistency_score = (mean_consistency + std_consistency) / 2

        # Find inconsistent regions using local outlier detection
        inconsistent_regions = []
        global_mean = np.mean(noise_map)
        global_std = np.std(noise_map)

        # Slide window and check local statistics
        stride = kernel_size
        for i in range(0, h - kernel_size, stride):
            for j in range(0, w - kernel_size, stride):
                patch = noise_map[i:i+kernel_size, j:j+kernel_size]
                local_mean = np.mean(patch)
                local_std = np.std(patch)

                # Check if significantly different from global
                if abs(local_mean - global_mean) > 2 * global_std:
                    inconsistent_regions.append((j, i, kernel_size, kernel_size))

        return NoiseAnalysisResult(
            noise_map=noise_map_norm,
            noise_consistency_score=max(0.0, min(1.0, consistency_score)),
            inconsistent_regions=inconsistent_regions,
            sensor_noise_profile={
                "global_mean": float(global_mean),
                "global_std": float(global_std),
                "quadrant_means": [float(m) for m in quadrant_means],
            }
        )

    def _detect_copy_move(self, gray: np.ndarray) -> CopyMoveResult:
        """Detect copy-move forgery using block matching."""
        h, w = gray.shape
        block_size = self.copy_move_block_size
        stride = block_size // 2

        # Extract blocks with their positions
        blocks = []
        positions = []

        for i in range(0, h - block_size + 1, stride):
            for j in range(0, w - block_size + 1, stride):
                block = gray[i:i+block_size, j:j+block_size].flatten()
                blocks.append(block)
                positions.append((i, j))

        blocks = np.array(blocks, dtype=np.float32)

        # Normalize blocks
        blocks_norm = (blocks - blocks.mean(axis=1, keepdims=True)) / (blocks.std(axis=1, keepdims=True) + 1e-6)

        # Compute similarity matrix (using dot product for normalized vectors)
        similarity = np.dot(blocks_norm, blocks_norm.T)

        # Find matching blocks (excluding self-matches and nearby blocks)
        matches = []
        similarity_scores = []
        forged_mask = np.zeros((h, w), dtype=np.uint8)

        for idx in range(len(blocks)):
            # Get top matches
            sims = similarity[idx]
            # Exclude self and nearby blocks (within 2*block_size)
            pi, pj = positions[idx]

            for jdx in np.argsort(sims)[::-1][1:10]:  # Top 9 excluding self
                if sims[jdx] < self.copy_move_threshold:
                    break

                qi, qj = positions[jdx]
                dist = np.sqrt((pi - qi)**2 + (pj - qj)**2)

                if dist > 2 * block_size:  # Not nearby
                    matches.append(((pj, pi), (qj, qi), (block_size, block_size)))
                    similarity_scores.append(float(sims[jdx]))

                    # Mark forged region
                    forged_mask[qi:qi+block_size, qj:qj+block_size] = 255

        return CopyMoveResult(
            matches=matches,
            similarity_scores=similarity_scores,
            forged_mask=forged_mask,
        )

    def _analyze_color_consistency(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze color consistency across image regions."""
        h, w = image.shape[:2]
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Split into patches and analyze color statistics
        patch_size = 64
        stride = 32
        inconsistent_regions = []
        max_inconsistency = 0.0
        anomaly_map = np.zeros((h, w), dtype=np.float32)

        # Global color statistics
        global_mean = np.mean(lab.reshape(-1, 3), axis=0)
        global_std = np.std(lab.reshape(-1, 3), axis=0)

        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                patch = lab[i:i+patch_size, j:j+patch_size].reshape(-1, 3)
                local_mean = np.mean(patch, axis=0)
                local_std = np.std(patch, axis=0)

                # Mahalanobis-like distance
                diff_mean = np.abs(local_mean - global_mean) / (global_std + 1e-6)
                diff_std = np.abs(local_std - global_std) / (global_std + 1e-6)
                inconsistency = np.mean(diff_mean) + np.mean(diff_std)

                anomaly_map[i:i+patch_size, j:j+patch_size] = max(
                    anomaly_map[i:i+patch_size, j:j+patch_size], inconsistency
                )

                if inconsistency > 3.0:  # Threshold
                    inconsistent_regions.append((j, i, patch_size, patch_size))
                    max_inconsistency = max(max_inconsistency, inconsistency)

        anomaly_map_norm = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        anomaly_colored = cv2.applyColorMap(anomaly_map_norm, cv2.COLORMAP_HOT)

        return {
            "inconsistent_regions": inconsistent_regions,
            "max_inconsistency": float(max_inconsistency),
            "anomaly_map": anomaly_colored,
        }

    def _analyze_edge_consistency(self, gray: np.ndarray) -> Dict[str, Any]:
        """Analyze edge consistency for detecting splicing."""
        # Compute edges using Canny
        edges = cv2.Canny(gray, 50, 150)

        # Compute edge density in patches
        patch_size = 64
        stride = 32
        h, w = gray.shape
        inconsistent_regions = []
        max_inconsistency = 0.0
        anomaly_map = np.zeros((h, w), dtype=np.float32)

        # Global edge density
        global_density = np.sum(edges > 0) / (h * w)

        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                patch_edges = edges[i:i+patch_size, j:j+patch_size]
                local_density = np.sum(patch_edges > 0) / (patch_size * patch_size)

                inconsistency = abs(local_density - global_density) / (global_density + 1e-6)

                anomaly_map[i:i+patch_size, j:j+patch_size] = max(
                    anomaly_map[i:i+patch_size, j:j+patch_size], inconsistency
                )

                if inconsistency > 2.0:
                    inconsistent_regions.append((j, i, patch_size, patch_size))
                    max_inconsistency = max(max_inconsistency, inconsistency)

        anomaly_map_norm = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        anomaly_colored = cv2.applyColorMap(anomaly_map_norm, cv2.COLORMAP_HOT)

        return {
            "inconsistent_regions": inconsistent_regions,
            "max_inconsistency": float(max_inconsistency),
            "anomaly_map": anomaly_colored,
        }

    def _analyze_metadata(
        self,
        image_path: str,
        expected_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze image metadata for inconsistencies."""
        import piexif
        from PIL import Image

        mismatches = []
        found_metadata = {}

        try:
            img = Image.open(image_path)

            # Extract EXIF
            if "exif" in img.info:
                exif_dict = piexif.load(img.info["exif"])

                # Check common fields
                checks = {
                    "Make": ("0th", piexif.ImageIFD.Make),
                    "Model": ("0th", piexif.ImageIFD.Model),
                    "DateTime": ("0th", piexif.ImageIFD.DateTime),
                    "Software": ("0th", piexif.ImageIFD.Software),
                    "GPSInfo": ("GPS", None),
                }

                for key, (ifd, tag) in checks.items():
                    if key in expected_metadata:
                        if ifd in exif_dict and tag in exif_dict[ifd]:
                            found_value = exif_dict[ifd][tag]
                            if isinstance(found_value, bytes):
                                found_value = found_value.decode('utf-8', errors='ignore')
                            if str(found_value) != str(expected_metadata[key]):
                                mismatches.append({
                                    "field": key,
                                    "expected": expected_metadata[key],
                                    "found": found_value,
                                })
                            found_metadata[key] = found_value
                        else:
                            mismatches.append({
                                "field": key,
                                "expected": expected_metadata[key],
                                "found": "MISSING",
                            })
        except Exception as e:
            logger.warning(f"Metadata extraction failed: {e}")

        return {
            "mismatches": mismatches,
            "extracted": found_metadata,
        }

    def _document_specific_checks(
        self,
        image: np.ndarray,
        doc_type: DocumentType
    ) -> Dict[str, Any]:
        """Perform document-type specific checks."""
        results = {}

        if doc_type == DocumentType.AADHAAR:
            results["qr_code"] = self._check_aadhaar_qr(image)
            results["photo_region"] = self._check_photo_consistency(image)

        elif doc_type == DocumentType.PAN:
            results["pan_format"] = self._check_pan_format(image)
            results["photo_region"] = self._check_photo_consistency(image)

        elif doc_type == DocumentType.PAYMENT_SCREENSHOT:
            results["ui_elements"] = self._check_payment_ui(image)
            results["timestamp_consistency"] = self._check_timestamp_consistency(image)

        elif doc_type == DocumentType.BANK_STATEMENT:
            results["table_structure"] = self._check_table_structure(image)
            results["font_consistency"] = self._check_font_consistency(image)

        return results

    def _check_aadhaar_qr(self, image: np.ndarray) -> Dict[str, Any]:
        """Check Aadhaar QR code validity."""
        # Detect QR codes
        detector = cv2.QRCodeDetector()
        data, bbox, _ = detector.detectAndDecode(image)

        if bbox is not None and data:
            # Check if QR data follows Aadhaar format
            is_valid = len(data) > 100  # Aadhaar QR has substantial data
            return {
                "detected": True,
                "data_length": len(data),
                "valid_format": is_valid,
                "anomaly": not is_valid,
                "confidence": 0.8 if is_valid else 0.3,
            }
        return {
            "detected": False,
            "anomaly": True,
            "confidence": 0.5,
        }

    def _check_pan_format(self, image: np.ndarray) -> Dict[str, Any]:
        """Check PAN card format and security features."""
        # Check for hologram region (typically top-right)
        h, w = image.shape[:2]
        hologram_region = image[0:h//4, 3*w//4:w]

        # Simple check: hologram should have high variance/reflectivity
        hologram_var = np.var(hologram_region)

        return {
            "hologram_detected": hologram_var > 500,
            "anomaly": hologram_var <= 500,
            "confidence": 0.6,
        }

    def _check_photo_consistency(self, image: np.ndarray) -> Dict[str, Any]:
        """Check photo region for morphing/signs of replacement."""
        # Photo typically in specific regions for ID docs
        h, w = image.shape[:2]
        # Common photo region: left side, middle
        photo_region = image[h//5:3*h//5, w//10:3*w//10]

        # Check for inconsistencies in photo region
        gray_photo = cv2.cvtColor(photo_region, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray_photo, cv2.CV_64F).var()

        # Low variance might indicate blurred/replaced photo
        return {
            "photo_sharpness": float(laplacian_var),
            "anomaly": laplacian_var < 100,
            "confidence": 0.5,
        }

    def _check_payment_ui(self, image: np.ndarray) -> Dict[str, Any]:
        """Check payment screenshot for UI element consistency."""
        # Check for standard UPI app UI elements
        # This is a simplified version - real implementation would use template matching
        h, w = image.shape[:2]

        # Check for common UI colors (UPI apps have specific color schemes)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Google Pay: teal/blue, PhonePe: purple, Paytm: blue
        # Check if colors match known schemes
        return {
            "ui_consistency_check": "basic",
            "anomaly": False,
            "confidence": 0.3,
        }

    def _check_timestamp_consistency(self, image: np.ndarray) -> Dict[str, Any]:
        """Check timestamp in screenshot for consistency."""
        # OCR would be needed for real implementation
        return {
            "timestamp_check": "requires_ocr",
            "anomaly": False,
            "confidence": 0.0,
        }

    def _check_table_structure(self, image: np.ndarray) -> Dict[str, Any]:
        """Check bank statement table structure."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)

        h_lines = np.sum(horizontal_lines > 0)
        v_lines = np.sum(vertical_lines > 0)

        return {
            "horizontal_lines": int(h_lines),
            "vertical_lines": int(v_lines),
            "has_table_structure": h_lines > 100 and v_lines > 50,
            "anomaly": not (h_lines > 100 and v_lines > 50),
            "confidence": 0.6,
        }

    def _check_font_consistency(self, image: np.ndarray) -> Dict[str, Any]:
        """Check font consistency in document."""
        # Would require OCR + font analysis
        return {
            "font_check": "requires_ocr",
            "anomaly": False,
            "confidence": 0.0,
        }


def analyze_document(
    image_path: str,
    document_type: DocumentType = DocumentType.GENERIC,
    **kwargs
) -> ForensicResult:
    """
    Convenience function for document forensic analysis.

    Args:
        image_path: Path to document image
        document_type: Type of document
        **kwargs: Additional arguments for DocumentForensics

    Returns:
        ForensicResult with analysis results
    """
    forensics = DocumentForensics(**kwargs)
    return forensics.analyze(image_path, document_type)