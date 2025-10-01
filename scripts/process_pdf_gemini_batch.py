import os
import csv
import requests

API_URL = "http://127.0.0.1:8000/model/process-pdf-gemini/"  # Correct endpoint for FastAPI route with prefix
INPUT_FOLDER = "/home/fiftyfive/Downloads/sample/sample"  # Folder containing PDF documents
OUTPUT_CSV = "gemini_pdf_results.csv"

# Ensure output folder exists
os.makedirs(INPUT_FOLDER, exist_ok=True)

def process_documents():
    results = []
    for filename in os.listdir(INPUT_FOLDER):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(INPUT_FOLDER, filename)
            with open(file_path, "rb") as f:
                files = {"file": (filename, f, "application/pdf")}
                try:
                    response = requests.post(API_URL, files=files)
                    response.raise_for_status()
                    output = response.json()
                except Exception as e:
                    output = {"error": str(e)}
                results.append({"document": filename, "output": output})
    # Write results to CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["document", "output"])
        writer.writeheader()
        for row in results:
            writer.writerow({"document": row["document"], "output": str(row["output"])})
    print(f"Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    process_documents()
