# Auggit-OCR

Auggit-OCR is a generative AI-powered document processing system designed to extract structured data from images and PDFs, such as shipping bills and invoices, using Google Gemini and OCR technologies.

## Features
- Extracts key fields (shipping bill, invoice, item details) from scanned documents
- Uses Google Gemini 2.0 Flash for advanced generative AI
- PDF to image conversion and OCR support
- FastAPI backend for easy API integration
- Modular codebase for extensibility

## Setup
1. Clone the repository
2. Create and activate a Python virtual environment
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements1.txt
   pip install -r requirements2.txt
   ```
4. Set up your `.env` file with the required Google API key
5. Run the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

## Usage

The project exposes three main API endpoints:

1. **`/process-pdf-yelo/`**
   - Accepts a PDF file and extracts structured data using the Yelo model pipeline.
   - Returns key fields such as shipping bill number, invoice details, item information, and more.

2. **`/process-image-gemini/`**
   - Accepts a PDF file and uses Google Gemini generative AI to extract structured data image by image.
   - Designed for advanced document understanding and field extraction from scanned images or PDFs.

3. **`/process-pdf-gemini/`**
   - Accepts a PDF file and processes it using Google Gemini generative AI for multi-page document extraction.
   - Returns a comprehensive JSON output with all relevant fields from the document.

Each endpoint returns structured JSON output with extracted fields for easy integration into downstream systems.

## Requirements
- Python 3.12+
- FastAPI, Uvicorn, pdf2image, pytesseract, Pillow, google-generativeai, python-dotenv
