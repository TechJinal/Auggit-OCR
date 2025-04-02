import os
import json
import torch
from pdf2image import convert_from_path
from PIL import Image, ImageDraw
import pytesseract
import re
from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
import os
from dotenv import load_dotenv
import google.generativeai as genai
import json
from fastapi.responses import JSONResponse
from pdf2image import convert_from_path

from test import filter_items, merge_data

load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

genai.configure(api_key=GOOGLE_API_KEY)

MODEL_CONFIG = {
    "temperature": 0.2,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]
app = FastAPI()

def pdf_to_images_shipping(pdf_path):
    try:
        pages = convert_from_path(pdf_path, 300)  # Convert PDF to images with 300 DPI
        image_paths = []

        os.makedirs("temp_files", exist_ok=True)  # Ensure temp directory exists

        for i, page in enumerate(pages):
            img_path = f"temp_files/page_{i + 1}.jpg"
            page.save(img_path, 'JPEG')  # Save each page as a JPEG image
            image_paths.append(img_path)
            
        return image_paths
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error converting PDF to images: {str(e)}")

def image_format(image_path):
    try:
        img = Path(image_path)
        if not img.exists():
            raise FileNotFoundError(f"Could not find image: {img}")

        image_parts = [{"mime_type": "image/jpeg", "data": img.read_bytes()}]  # Adjust MIME type as needed
        return image_parts
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

def gemini_output(image_path, system_prompt, user_prompt):
    try:
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=MODEL_CONFIG,
            safety_settings=safety_settings,
        )
        image_info = image_format(image_path)
        input_prompt = [system_prompt, image_info[0], user_prompt]
        response = model.generate_content(input_prompt)
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response from Gemini: {str(e)}")

def call_gemini_model(full_prompt):
    """Generate structured data using Gemini 1.5 Flash."""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")

        # Generate response using Gemini
        response = model.generate_content(full_prompt)

        # Extract and return the response text
        if response and response.candidates:
            return response.candidates[0].content.parts[0].text.strip()
        else:
            return "No valid response generated."
    except Exception as e:
        print(f"Error while generating content with Gemini: {str(e)}")
        return None

    
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

@app.post("/process-document/")
async def process_document(file: UploadFile = File(...)):

    file_location = f"temp_files/{file.filename}"
    os.makedirs("temp_files", exist_ok=True)

    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())

    try:
        if file.filename.endswith(".pdf"):
            # Convert PDF to images
            image_paths = pdf_to_images_shipping(file_location)  

            json_output = {}
            for image_path in image_paths:
                system_prompt = """
                       You are a specialist in comprehending receipts.
                       Input images in the form of receipts will be provided to you,
                       and your task is to respond to questions based on the content of the input image.
                       """

                user_prompt = f"""

                Extract and convert the image-based document data into a structured JSON format by carefully identifying and capturing each field based on the tags and descriptions provided below. Ensure all data is accurately extracted as per the specified rules, keeping original formatting intact where required. Focus on extracting key fields from all pages of the document, ensuring completeness and accuracy.

                ### **Tags and Extraction Rules:**

                1. **Shipping Bill Number** : key value as = shippingBillNumber  
                - Identify and extract the unique **Shipping Bill Number** from the document.  
                - The Shipping Bill Number is typically a numeric string found near the top of the document, labeled as **"SB No"**.
                - If no value is found, dont return null instead return empty double quotes.

                2. **Invoice Number**  : key value as = invoiceNumber
                - Capture the **Invoice Number** present in the document.  
                - This is typically found next to the label **"Invoice No"**.
                - If no value is found, dont return null instead return empty double quotes.

                3. **Shipping Bill Date**  : key value as = shippingBillDate
                - Extract the date associated with the **Shipping Bill**.  
                - This information may be found near the **Shipping Bill Number** or next to the **Invoice Date**.
                - If no value is found, dont return null instead return empty double quotes.

                4. **Invoice Date**  : 
                - Record the **Invoice Date** mentioned in the document.  
                - Look for a label like **"Invoice Date"** or similar.
                - If no value is found, dont return null instead return empty double quotes.

                5. **Port Code**  : key value as = portCode
                - Identify and capture the **Port Code** from the document. 
                - it is Port of Ldg-Code.
                - If no value is found, dont return null instead return empty double quotes.

                6. **Location**  : key value as = location
                - Extract the **Location** mentioned in the document. 
                - State of Origin is location.
                - If no value is found, dont return null instead return empty double quotes.

                7. **Item Number**  : key value as = itemNumber
                - Fetch the RITC CD .
                - only of 8 digit number. consider them only as item number.
                - If no value is found, return empty list.
                - If more than one value is found, return all values in list format.

                8. **Quantity**  : key value as = quantity
                - Extract the quantity of each item listed in the document.
                - Can be PCS, KGS or NOS.
                - If no value is found, return empty list.
                - If more than one value is found, return all values in list format.

                9. **Item Details**  : key value as = itemDetails
                - Extract the descriptions and quantities of all the items listed under "item details" section in the document in list format. Include every details of every item in single quotes.
                - If no value is found, return empty list.
                - If more than one value is found, return all values in list format.

                Remember only consider items those are present under item details section of the document.
                """

                structured_output = gemini_output(image_path, system_prompt, user_prompt)

                # Convert string to JSON
                # structured_json = json.loads(structured_output.strip("```json").strip("```"))

                # Add to the final output
                json_output[image_path] = structured_output

            
            result = {}
            for file_path, json_str in json_output.items():
                # Replace the code block markers (```json) with an empty string
                clean_json_str = json_str.replace("```json", "").replace("```", "").strip()
                # Convert the cleaned string to a dictionary
                try:
                    parsed_json = json.loads(clean_json_str)
                    result[file_path] = parsed_json
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON for {file_path}: {e}")
            print("\nresult", result)
            final_combined_data = merge_data(result)
            print("\nfinal_combined_data", final_combined_data)

            prompt = f"""
            
            **Task:**  
            You are given a JSON object containing shipping and invoice details along with a list of items. Each item has an `itemNumber`, `quantity`, and `itemDetails`. Your task is to:  

            1. **Extract bill of entry details** (if available) from `itemDetails`.  
            - The bill of entry number is usually indicated by **"BE NO."**, **"B/E NO."**, **"BOE NO."**, or similar terms.  
            - The bill of entry date is typically in **DD/MM/YYYY** or **DD.MM.YYYY** format.  
            2. **Create a `billOfEntry` list** for each item where such details are found.  
            - Each entry in the list should have:  
                - `"billOfEntryNumber": "<extracted number>"`  
                - `"billOfEntryDate": "<extracted date>"`  
            3. **Merge items that have similar `itemDetails`:**  
                - If two or more items have nearly identical `itemDetails`, they should be considered as **one item**.  
                - If one entry has an `itemNumber` while the other does not, keep the valid `itemNumber`.  
                - Ensure the `billOfEntry` field consolidates all relevant details from merged items.  
            4. If no bill of entry details are found in an item’s `itemDetails`, the `billOfEntry` field should be an empty list (`[]`).  
            5. Return the modified JSON with the `billOfEntry` field added to each item.  

            ---

            ### **Input JSON Example 1:**
            ```json
            {{
            "shippingBillNumber": "7143265",
            "invoiceNumber": "MRF/THIL/0036/19",
            "shippingBillDate": "24/09/2019",
            "invoiceDate": "26/08/2019",
            "portCode": "INNSA1",
            "location": "TAMIL NADU",
            "items": [
                {{
                "itemNumber": "73102990",
                "quantity": "2640.000",
                "itemDetails": "'(IMPORTED BE NO.4309572 DT: 01.08.2019 1 4 PLT, 3934667 DT:04.07.2019')"
                }}
            ]
            }}
            ```

            ---

            ### **Expected Output JSON Format:**
            ```json
            {{
            "shippingBillNumber": "7143265",
            "invoiceNumber": "MRF/THIL/0036/19",
            "shippingBillDate": "24/09/2019",
            "invoiceDate": "26/08/2019",
            "portCode": "INNSA1",
            "location": "TAMIL NADU",
            "items": [
                {{
                "itemNumber": "73102990",
                "quantity": "2640.000",
                "itemDetails": "'(IMPORTED BE NO.4309572 DT: 01.08.2019 1 4 PLT, 3934667 DT:04.07.2019')",
                "billOfEntry": [
                    {{
                    "billOfEntryNumber": "4309572",
                    "billOfEntryDate": "01.08.2019"
                    }},
                    {{
                    "billOfEntryNumber": "3934667",
                    "billOfEntryDate": "04.07.2019"
                    }}
                ]
                }}
            ]
            }}
            ```

        ### **Input JSON Example2:**
        ```json
        {{
            "shippingBillNumber": "5000351",
            "invoiceNumber": "MRFCHN/0039/2019",
            "shippingBillDate": "06/03/2019",
            "invoiceDate": "28/02/2019",
            "portCode": "",
            "location": "Tamil Nadu",
            "items": [
                {{
                "itemNumber": "",
                "quantity": "192.000000",
                "itemDetails": "'RETURNABLE METAL PALLETS(MB5)MADE OF GALVANISED STEEL'",
                "billOfEntry": []
                }},
                {{
                "itemNumber": "73102990",
                "quantity": "",
                "itemDetails": "'73102990-RETURNABLE METAL PALLETS (MB5) MADE OF GALVANISED STEEL'",
                "billOfEntry": []
                }}
            ]
            }}
        ```
        Expected Output JSON Format (with merged items):
        ```json
            {{
            "shippingBillNumber": "5000351",
            "invoiceNumber": "MRFCHN/0039/2019",
            "shippingBillDate": "06/03/2019",
            "invoiceDate": "28/02/2019",
            "portCode": "",
            "location": "Tamil Nadu",
            "items": [
                {{
                "itemNumber": "73102990",
                "quantity": "192.000000",
                "billOfEntry": []
                }}
            ]
            }}
        ```
            ---

            ### **Instructions for the Model:**
            - Identify bill of entry numbers and dates within `itemDetails`.  
            - Format the extracted details in the `billOfEntry` field under each item.  
            - Merge items with similar itemDetails into a single entry while:
                - Retaining the valid itemNumber if available.
                - Ensuring billOfEntry remains accurate for the merged item.
            - If no bill of entry details are found, return an empty list for `billOfEntry`.  
            - Preserve the structure and content of the original JSON, only adding the required field.  
            - And i dont want python code, do the conversion and return the json only.

            #Here is the input JSON object:
            {final_combined_data} convert this.
            """
            
            response = call_gemini_model(prompt)
            print("\nresponse", response)
            clean_json_str = response.replace("```json", "").replace("```", "").strip()
            response = json.loads(clean_json_str)
            filtered_data = filter_items(response)
            for item in filtered_data["items"]:
                del item["itemDetails"]
            
            print("response", filtered_data)
            print("\n")
            return JSONResponse(content=filtered_data)

        else:
            raise HTTPException(status_code=400, detail="Invalid file format. Only PDF files are supported.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    finally:
        # Clean up temporary files
        for image_path in image_paths:
            os.remove(image_path)
        os.remove(file_location)