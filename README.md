# Project 4 — Image or Text Recognition (Basic)

**DecodeLabs AI Internship — Optional Mastery Milestone**

## Overview

This project implements **Path 1: Optical Character Recognition (OCR)**.

It takes a raw image, applies image pre-processing using **OpenCV**, extracts text using the **Tesseract OCR engine**, and accepts only detected words with a confidence score of **80% or higher**.

## How It Works

The project follows the following OCR pipeline:

1. **Ingest** — Loads the input image using OpenCV.
2. **Pre-processing**

   * Converts the image to grayscale.
   * Applies Gaussian blur to reduce noise and artifacts.
   * Uses Otsu thresholding to create a clean binary image.
   * Performs deskewing to correct tilted text.
3. **OCR** — Uses `pytesseract.image_to_data()` to extract detected words along with their confidence scores and bounding boxes.
4. **Confidence Gate** — Rejects words with a confidence score below **80%**.
5. **Visual Confirmation** — Generates an annotated image showing accepted and rejected detections using bounding boxes and confidence labels.

## Requirements

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains:

```text
opencv-python
pytesseract
numpy
Pillow
```

### Tesseract OCR

Tesseract OCR must also be installed separately because `pytesseract` provides the Python interface to the Tesseract OCR engine.

* **Windows:** Install Tesseract from the UB Mannheim distribution.
* **Ubuntu/Debian:**

```bash
sudo apt install tesseract-ocr
```

* **macOS:**

```bash
brew install tesseract
```

## Usage

Run the program with:

```bash
python project4_ocr.py sample_input.png
```

To save generated outputs inside a `results` folder:

```bash
python project4_ocr.py sample_input.png --psm 6 --threshold 80 --outdir ./results
```

### Command-Line Options

* `--psm` — Tesseract Page Segmentation Mode. The default is `6`, which is suitable for a uniform block of text.
* `--threshold` — Minimum confidence percentage required to accept a detected word. The default is `80`.
* `--outdir` — Directory where generated output files are saved. The default is the current directory.

## Project Files

| File                   | Purpose                                                           |
| ---------------------- | ----------------------------------------------------------------- |
| `project4_ocr.py`      | Main OCR pipeline and image-processing code                       |
| `README.md`            | Project documentation and usage instructions                      |
| `requirements.txt`     | Required Python libraries                                         |
| `sample_input.png`     | Sample image used to test the OCR pipeline                        |
| `output_annotated.png` | Example visual result showing detected text and confidence scores |

## Generated Output

When the program runs, it automatically generates the following files in the specified output directory:

* `output_cleaned.png` — Pre-processed binary image.
* `output_annotated.png` — Original image with bounding boxes and confidence labels.
* `output_text.txt` — Final extracted text containing words that passed the confidence gate.

These files are generated automatically by the program. The annotated output may be included in the repository as visual evidence of the OCR result.

## Milestone Validation

The project addresses the main requirements of the OCR milestone:

* **Library Integration** — Uses OpenCV and `pytesseract` for image processing and OCR.
* **Pre-Processing Integrity** — Performs grayscale conversion, Gaussian blur, Otsu thresholding, and deskewing.
* **Confidence Gate** — Accepts only detected words with a confidence score of **80% or higher**.
* **Visual Confirmation** — Produces an annotated image showing detected text, bounding boxes, and confidence scores.

### Sample Test Result

The project was tested successfully using the included sample image.

* **18 words** passed the 80% confidence gate.
* **2 words** were rejected.
* **95.4%** average confidence was achieved among the accepted words.

## Notes

* The program can be tested with any suitable image containing printed text.
* The `--psm` option can be adjusted for different image layouts.
* OCR results and confidence scores may vary depending on image quality, text size, lighting, orientation, and layout.
* Tesseract OCR must be installed separately because it is not included in the Python package requirements.
