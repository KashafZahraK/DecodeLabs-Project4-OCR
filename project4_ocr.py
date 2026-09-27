

import argparse
import sys
import cv2
import numpy as np
import pytesseract
from pytesseract import Output

# ----------------------------------------------------------------------------
# Step 1: Pre-processing (Requirement #2 — Pre-Processing Integrity)
# ----------------------------------------------------------------------------

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Collapse the 3D RGB matrix into a 1D intensity matrix."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(gray: np.ndarray) -> np.ndarray:
    """Gaussian blur to smooth out micro-imperfections and sensor noise."""
    return cv2.GaussianBlur(gray, (5, 5), 0)


def adaptive_threshold(blurred: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Force every pixel to a binary decision using Otsu's method.
    Returns the binary image and the computed cutoff intensity.
    """
    cutoff, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary, cutoff


def deskew(binary: np.ndarray) -> np.ndarray:
    """
    Calculate the rotation angle of the text mass and rotate the image
    back to a horizontal baseline. Falls back gracefully if the page
    is (close to) empty or already straight.
    """
    coords = np.column_stack(np.where(binary > 0))
    if coords.shape[0] < 20:
        return binary

    angle = cv2.minAreaRect(coords)[-1]
    # cv2.minAreaRect returns angles in [-90, 0); normalize to a small
    # rotation correction rather than a full 90-degree flip.
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = binary.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        binary, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def preprocess(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Run the full pre-processing chain and return the cleaned image."""
    gray = to_grayscale(image)
    blurred = denoise(gray)
    binary, cutoff = adaptive_threshold(blurred)
    cleaned = deskew(binary)
    return cleaned, cutoff


# ----------------------------------------------------------------------------
# Step 2: OCR + Confidence Gate (Requirement #1 & #3)
# ----------------------------------------------------------------------------
def run_ocr(cleaned_image: np.ndarray, psm: int = 6) -> dict:
    """
    Run pytesseract on the cleaned image and return the raw word-level
    output dictionary (text, confidence, bounding box per detected word).
    """
    config = f"--oem 3 --psm {psm}"
    data = pytesseract.image_to_data(
        cleaned_image, config=config, output_type=Output.DICT
    )
    return data


def apply_confidence_gate(data: dict, threshold: float = CONFIDENCE_THRESHOLD):
    """
    The 'Gatekeeper Rule': only accept words whose confidence score
    clears the minimum validated standard. Returns a list of
    (text, confidence, (x, y, w, h)) tuples that passed the gate,
    plus everything that was rejected (for transparency).
    """
    accepted, rejected = [], []
    n = len(data["text"])
    for i in range(n):
        word = data["text"][i].strip()
        conf = float(data["conf"][i])
        if not word or conf < 0:  # tesseract uses -1 for non-text regions
            continue
        box = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
        if conf >= threshold:
            accepted.append((word, conf, box))
        else:
            rejected.append((word, conf, box))
    return accepted, rejected


# ----------------------------------------------------------------------------
# Step 3: Visual Confirmation (Requirement #4)
# ----------------------------------------------------------------------------
def annotate(original: np.ndarray, accepted: list, rejected: list) -> np.ndarray:
    """
    Draw bounding boxes over the original image:
      - Green boxes = words that passed the 80% confidence gate
      - Red boxes   = words that were dropped for low confidence
    """
    annotated = original.copy()
    for word, conf, (x, y, w, h) in rejected:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 1)

    for word, conf, (x, y, w, h) in accepted:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 2)
        label = f"{conf:.0f}%"
        cv2.putText(
            annotated, label, (x, max(y - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 1, cv2.LINE_AA,
        )
    return annotated


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Project 4 — OCR Pipeline")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument(
        "--psm", type=int, default=6,
        help="Tesseract Page Segmentation Mode (default 6: uniform block of text)",
    )
    parser.add_argument(
        "--threshold", type=float, default=CONFIDENCE_THRESHOLD,
        help=f"Minimum confidence to accept a word (default {CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--outdir", default=".", help="Directory to save output artifacts"
    )
    args = parser.parse_args()

    # --- Step 1: Ingest -----------------------------------------------------
    image = cv2.imread(args.image_path)
    if image is None:
        print(f"ERROR: could not read image at '{args.image_path}'")
        sys.exit(1)

    print(f"[1/4] Loaded image: {args.image_path}  shape={image.shape}")

    # --- Step 2: Pre-process -------------------------------------------------
    cleaned, cutoff = preprocess(image)
    print(f"[2/4] Pre-processing complete. Otsu cutoff intensity = {cutoff:.1f}")

    # --- Step 3: OCR + Confidence Gate ---------------------------------------
    data = run_ocr(cleaned, psm=args.psm)
    accepted, rejected = apply_confidence_gate(data, threshold=args.threshold)

    if accepted:
        avg_conf = sum(c for _, c, _ in accepted) / len(accepted)
    else:
        avg_conf = 0.0

    print(f"[3/4] OCR complete. "
          f"{len(accepted)} word(s) passed the {args.threshold:.0f}% gate, "
          f"{len(rejected)} rejected. Average accepted confidence: {avg_conf:.1f}%")

    # --- Step 4: Visual Confirmation -----------------------------------------
    annotated = annotate(image, accepted, rejected)

    cleaned_path = f"{args.outdir}/output_cleaned.png"
    annotated_path = f"{args.outdir}/output_annotated.png"
    cv2.imwrite(cleaned_path, cleaned)
    cv2.imwrite(annotated_path, annotated)

    extracted_text = " ".join(word for word, _, _ in accepted)
    text_path = f"{args.outdir}/output_text.txt"
    with open(text_path, "w") as f:
        f.write(extracted_text)

    print(f"[4/4] Saved:")
    print(f"       - Cleaned (pre-processed) image -> {cleaned_path}")
    print(f"       - Annotated image (confidence boxes) -> {annotated_path}")
    print(f"       - Extracted text -> {text_path}")
    print()
    print("=" * 60)
    print("EXTRACTED TEXT (>= {:.0f}% confidence):".format(args.threshold))
    print("=" * 60)
    print(extracted_text if extracted_text else "(no text cleared the confidence gate)")
    print("=" * 60)

    # Gatekeeper validation summary
    passed_gate = avg_conf >= args.threshold and len(accepted) > 0
    print(f"\nMilestone validation: "
          f"{'PASSED ✅' if passed_gate else 'NOT MET ❌'} "
          f"(avg confidence {avg_conf:.1f}% vs required {args.threshold:.0f}%)")


if __name__ == "__main__":
    main()
    
