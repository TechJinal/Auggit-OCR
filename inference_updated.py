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
import openai

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

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

# OpenAI utility function
def call_openai_model(system_message: str, prompt: str) -> str:
    """Calls OpenAI's API to generate a response."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")
    
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

                2. **Invoice Number**  : key value as = invoiceNumber
                - Capture the **Invoice Number** present in the document.  
                - This is typically found next to the label **"Invoice No"**.

                3. **Shipping Bill Date**  : key value as = shippingBillDate
                - Extract the date associated with the **Shipping Bill**.  
                - This information may be found near the **Shipping Bill Number** or next to the **Invoice Date**.

                4. **Invoice Date**  : 
                - Record the **Invoice Date** mentioned in the document.  
                - Look for a label like **"Invoice Date"** or similar.

                5. **Port Code**  : key value as = portCode
                - Identify and capture the **Port Code** from the document. 
                - it is Port of Ldg-Code.

                6. **Location**  : key value as = location
                - Extract the **Location** mentioned in the document. 
                - State of Origin is location.

                7. **Item Number**  : key value as = itemNumber
                - Fetch the RITC CD .
                - only of 8 digit number. consider them only as item number.

                8. **Quantity**  : key value as = quantity
                - Extract the quantity of each item listed in the document.
                - Can be PCS, KGS or NOS.

                9. **Item Details**  : key value as = itemDetails
                - Extract the descriptions and quantities of all the items listed under "item details" section in the document. Include every details of every item in single quotes.

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
            # print(json.dumps(result, indent=4))
            # Combine the extracted data and process it further
            
            final_system_prompt = """
                You are an execellent json converter. 
                Your task is to convert the extracted data into a final json format.
            """

            final_user_prompt = f"""
                Combine the extracted data from all pages into a single JSON format.
                Ensure all data is accurately extracted as per the specified rules, keeping original formatting intact where required.
                Focus on extracting key fields from all pages of the document, ensuring completeness and accuracy.

                ### **Tags and Extraction Rules:**

                1. **Shipping Bill Number** : key value as = shippingBillNumber
                2. **Invoice Number** : key value as = invoiceNumber
                3. **Shipping Bill Date** : key value as = shippingBillDate
                4. **Invoice Date** : key value as = invoiceDate
                5. **Port Code** : key value as = portCode
                6. **Location** : key value as = location
                7. **Item Number** : key value as = itemNumber
                8. **Quantity** : key value as = quantity
                9. **Bill of entry Number** : key value as = billOfEntryNumber 
                10. **Bill Of Entry Date** : key value as = billOfEntryDate

                Points to be noted:-
                1. If the tags are not present in the image then set their value as 'null'.
                2. for bill of entry number and bill of entry date, use item details to extract the data. If BE NO is present then set its value and if and date format is present like 06.11.19 then set bill of entry date value as the date.
                    Example:
                    "itemDetails": [
                        "73102990 EMPTY GOODPACK METAL BOXES TYPE MBS BEING RETURNED TO SUPPLIERNO COMMERCIAL VALUE DECLARED FOR CUSTOMS PURPOSE ONLY 120.000PCS",
                        "73102990 IMPORTED VIDE BE NO:3912537 DTD 03.07.2019(QTY:48FCS), 3912769 DID 03.07.2019(QTY: 48PCS),3912536 DTD 03.07.2019(QTY:24PCS 1.000PCS",
                        "73102990 EXPORT VIDE FEMA NOTIFICATION 23/2000-RB /03.05.2000 & FEMA116/2004-RB/25.03.2004 1.000PCS"
                    ]
                    then the output should be:
                    "billOfEntryNumber": [[], ["3912769", "3912769", "3912536"], []],
                    "billOfEntryDate": [[], ["03.07.2019", "03.07.2019", "03.07.2019"], []],
                3. It is possible that in one item of itemdetails there are more than one BE NO or BE Date then mention all of them.
                4. It is also possible that in one item of itemdetails no BE NO or BE Date is present then set their value as 'null'.

                ### **Final JSON Format:**  
                {{
                    "shippingBillNumber": "8763764",
                    "invoiceNumber": "MRFCHN/0107/2019",
                    "shippingBillDate": "06/12/2019",
                    "invoiceDate": "29/11/2019",
                    "portCode": "INTVT6",
                    "location": "TAMIL NADU",
                    "items": 
                    [
                        {{
                        "itemNumber": "73269099",
                        "quantity": "12499.200KGS",
                        "billOfEntry": [
                            {{
                            "billOfEntryNumber": "5565161",
                            "billOfEntryDate": "06.11.19"
                            }},
                            {{
                            "billOfEntryNumber": "5565149",
                            "billOfEntryDate": "06.11.19"
                            }}
                        ]
                        }},
                        {{
                        "itemNumber": "39231090",
                        "quantity": "343.000KGS",
                        "billOfEntry": [
                            {{
                            "billOfEntryNumber": "5565101",
                            "billOfEntryDate": "06.11.19"
                            }},
                            {{
                            "billOfEntryNumber": "5565149",
                            "billOfEntryDate": "06.11.19"
                            }}
                        ]
                        }},
                        {{
                        "itemNumber": "39231090",
                        "quantity": "714.000KGS",
                        "billOfEntry": [
                            {{
                            "billOfEntryNumber": "5565101",
                            "billOfEntryDate": "06.11.19"
                            }},
                            {{
                            "billOfEntryNumber": "5565149",
                            "billOfEntryDate": "06.11.19"
                            }}
                        ]
                        }},...
                    ]
                }}
                5. If the tags are not present in the image then set their value as 'null'.
                6. Item number is only of 8 digit. Consider them only as item number. If any number bigger than 8 digit is there than dont include it.
                7. Length of list itemNumber, quantity, billOfEntryNumber, billOfEntryDate should be equal.
                
                Here is the json you need to work on:
                {json.dumps(result, indent=4)}
            """

            # Call Gemini model with the combined result
            response = call_openai_model(final_system_prompt, final_user_prompt)
            clean_json_str = response.replace("```json", "").replace("```", "").strip()
            response = json.loads(clean_json_str)
            print("response")
            return JSONResponse(content=response)

        else:
            raise HTTPException(status_code=400, detail="Invalid file format. Only PDF files are supported.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    finally:
        # Clean up temporary files
        for image_path in image_paths:
            os.remove(image_path)
        os.remove(file_location)