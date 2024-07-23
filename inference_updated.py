import os
import json
import torch
from pdf2image import convert_from_path
from PIL import Image, ImageDraw
import pytesseract
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

app = FastAPI()

# Step 1: Load the model
model_path = r'runs/train/exp16/weights/best.pt'

def load_model(model_path):
    model = torch.hub.load(os.getcwd(), 'custom', source='local', path=model_path, force_reload=True)
    return model

model = load_model(model_path)

# Step 2: Convert PDF to images
def pdf_to_images(pdf_path):
    images = convert_from_path(pdf_path)
    return images

# Step 3: Run inference on images
def run_inference(model, images):
    results = []
    for i, image in enumerate(images):
        image = image.convert('RGB')  # Ensure image is in RGB mode
        results.append(model(image))
    return results

# Step 4: Clean extracted text
def clean_text(text):
    cleaned_text = text.replace('|', '').strip()  # Remove '|' and trim whitespace
    return cleaned_text

# Function to parse the details field and extract hsCode, brand, and noOfBags
def parse_details(details):
    hs_code = brand = no_of_bags = None
    details = details.lower()
    if 'hs code:' in details:
        hs_code = details.split('hs code:')[1].split()[0]
    if 'brand :' in details:
        brand = details.split('brand :')[1].split('/')[0].strip()
    if 'bags of' in details:
        no_of_bags = details.split('total: ')[1].split(' ')[0]

    return hs_code, brand, no_of_bags

# Step 5: Extract values with labels and OCR
def extract_labels_and_ocr(results, images):
    pages_data = {}
    for i, (result, image) in enumerate(zip(results, images)):
        labels_with_text = {}
        title = None
        
        draw = ImageDraw.Draw(image)  # Initialize the drawing context for bounding boxes
        
        for det in result.xyxy[0]:
            xmin, ymin, xmax, ymax, conf, cls = det
            label = result.names[int(cls)]
            xmin, ymin, xmax, ymax = map(int, (xmin.item(), ymin.item(), xmax.item(), ymax.item()))
            cropped_image = image.crop((xmin, ymin, xmax, ymax))
            text = pytesseract.image_to_string(cropped_image, config='--psm 6')  # OCR to extract text
            cleaned_text = clean_text(text)  # Clean the extracted text
            labels_with_text[label] = cleaned_text
            
            # # Draw the bounding box on the original image
            # draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
            # draw.text((xmin, ymin), label, fill="red")  # Optionally, add label text

            # Check if the label is a title
            if label.lower() == 'title':
                title = cleaned_text

        # Use title if available; otherwise fallback to page number
        page_key = title if title else f'page_{i+1}'
        pages_data[page_key] = labels_with_text

        # Extract hsCode, brand, and noOfBags from details
        if 'details' in labels_with_text:
            hs_code, brand, no_of_bags = parse_details(labels_with_text['details'])
            if hs_code: labels_with_text['hsCode'] = hs_code
            if brand: labels_with_text['brand'] = brand
            if no_of_bags: labels_with_text['noOfBags'] = no_of_bags
            del labels_with_text['details']  # Remove the original 'details' field if necessary

        # Save the modified image with bounding boxes in the detected_images folder
        # image.save(f"detected_images/modified_page_{i+1}.png")

    return pages_data


def format_list_of_items(pages_data):
    for page, labels in pages_data.items():
        if "listOfItem" in labels:
            items = labels["listOfItem"].split('\n\n')  # First attempt to split by double newline
            
            # If only one item, it means no '\n\n' was found, so split by single newline
            if len(items) == 1:
                items = labels["listOfItem"].split('\n')
            
            formatted_items = []
            for item in items:
                elements = item.split()
                if len(elements) == 4:
                    formatted_items.append(elements)
                else:
                    # Handle cases where elements do not have exactly four elements
                    # For now, we'll discard these items, but you can pad or handle them differently
                    continue
            
            labels["listOfItem"] = formatted_items
    return pages_data


# Main function to process the PDF and run inference with OCR
def process_pdf(pdf_path, model):
    images = pdf_to_images(pdf_path)
    results = run_inference(model, images)
    pages_data = extract_labels_and_ocr(results, images)
    pages_data = format_list_of_items(pages_data)

    # Convert the output to JSON format
    json_output = json.dumps(pages_data, indent=4)

    return json_output

@app.post("/process-pdf/")
async def process_pdf_endpoint(file: UploadFile = File(...)):
    # Save the uploaded file to a temporary location
    temp_pdf_path = f"/tmp/{file.filename}"
    with open(temp_pdf_path, "wb") as f:
        f.write(await file.read())

    # Process the PDF and get the JSON output
    detected_data_json = process_pdf(temp_pdf_path, model)

    # Return the JSON response
    return JSONResponse(content=json.loads(detected_data_json))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
