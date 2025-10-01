import os
import json
from app.config import MODEL_CONFIG, SAFETY_SETTINGS
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

load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

genai.configure(api_key=GOOGLE_API_KEY)

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
            model_name="gemini-2.0-flash",
            generation_config=MODEL_CONFIG,
            safety_settings=SAFETY_SETTINGS,
        )
        image_info = image_format(image_path)
        input_prompt = [system_prompt, image_info[0], user_prompt]
        response = model.generate_content(input_prompt)
        if response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count
            total_tokens = response.usage_metadata.total_token_count

            print(f"Input tokens: {input_tokens}")
            print(f"Output tokens: {output_tokens}")
            print(f"Total tokens for this call: {total_tokens}")
        else:
            print("Usage metadata not available in the response.")
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response from Gemini: {str(e)}")
    

def merge_data(data):
    merged_data = {
        "shippingBillNumber": "",
        "invoiceNumber": "",
        "shippingBillDate": "",
        "invoiceDate": "",
        "portCode": "",
        "location": "",
        "items": []
    }
    
    # Extract common metadata from the first non-empty page
    for page in data.values():
        for key in ["shippingBillNumber", "invoiceNumber", "shippingBillDate", "invoiceDate", "portCode", "location"]:
            if not merged_data[key] and page.get(key):
                merged_data[key] = page[key]
    
    # Collect item details
    all_items = []
    for page in data.values():
        item_numbers = page.get("itemNumber", [])
        quantities = page.get("quantity", [])
        item_details = page.get("itemDetails", [])
        
        # Ensure all lists have the same length by padding with empty strings
        max_length = max(len(item_numbers), len(quantities), len(item_details))
        item_numbers += [""] * (max_length - len(item_numbers))
        quantities += [""] * (max_length - len(quantities))
        item_details += [""] * (max_length - len(item_details))
        
        all_items.extend(
            {"itemNumber": item_numbers[i], "quantity": quantities[i], "itemDetails": item_details[i]}
            for i in range(max_length)
        )
    
    # Filter out items where itemNumber does not have exactly 8 characters
    merged_data["items"] = all_items

    return merged_data

def filter_items(response):
    filtered_items = []
    for item in response.get("items", []):
        item_number = str(item.get("itemNumber", ""))
        item_details = item.get("itemDetails")

        if (re.fullmatch(r"\d{8}", item_number) or item_number == "") and item_details:
            filtered_items.append(item)

    response["items"] = filtered_items
    return response

def call_gemini_model(full_prompt):
    """Generate structured data using Gemini 2.0 Flash."""
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")

        # Generate response using Gemini
        response = model.generate_content(full_prompt)

        if response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count
            total_tokens = response.usage_metadata.total_token_count

            print(f"Input tokens: {input_tokens}")
            print(f"Output tokens: {output_tokens}")
            print(f"Total tokens for this call: {total_tokens}")
        else:
            print("Usage metadata not available in the response.")

        # Extract and return the response text
        if response and response.candidates:
            return response.candidates[0].content.parts[0].text.strip()
        else:
            return "No valid response generated."
    except Exception as e:
        print(f"Error while generating content with Gemini: {str(e)}")
        return None
    
async def process_gemini_image(file: UploadFile):

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
                print(f"Processing image: {image_path}")
                system_prompt = """
                       You are a specialist in comprehending receipts.
                       Input images in the form of receipts will be provided to you,
                       and your task is to respond to questions based on the content of the input image.
                       """

                user_prompt = """

                Extract and convert the image-based document data into a structured JSON format by carefully identifying and capturing each field based on the tags and descriptions provided below. Ensure all data is accurately extracted as per the specified rules, keeping original formatting intact where required. Focus on extracting key fields from all pages of the document, ensuring completeness and accuracy.

                ### **Tags and Extraction Rules:**

                1. **Shipping Bill Number** : key value as = shippingBillNumber  
                - Identify and extract the unique **Shipping Bill Number** from the document.  
                - The Shipping Bill Number is typically a numeric string found near the top of the document, labeled as **"SB No"**.
                - If no value is found, dont return null instead return empty double quotes.

                2. **Invoice Number**  : key value as = invoiceNumber
                - Capture the **Invoice Number** present in the document.  
                - This is typically found next to the label **"Invoice No"** or **"Invoice Number"** or **"Inv. No."**.
                - Do not take any other invoice value like Invoice dt or only Invoice or Inv. val. These will not give you the exact invoice number. 
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
            filter_items(final_combined_data)
            print("\nfinal_combined_data", final_combined_data)

            prompt = f"""
            You are a specialist in comprehending receipts. You are given a list of json objects containing itemNumber, quantity and itemDetails. Your task is to:
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

            For example1 :
            ```json
            [
                 {{
                 "itemNumber": "73102990",
                 "quantity": "2640.000",
                 "itemDetails": "(IMPORTED BE NO.4309572 DT: 01.08.2019 1 4 PLT, 3934667 DT:04.07.2019')"
                 }}
            ]
            ```
            should be converted to:

            ```json
            [
                {{
                "itemNumber": "73102990",
                "quantity": "2640.000",
                "itemDescription": "RETURNABLE METAL PALLETS(MB5)MADE OF GALVANISED STEEL",
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
            ```

            For example2 :

            ```json
            [
                {{
                "itemNumber": "",
                "quantity": "192.000000",
                "itemDetails": "RETURNABLE METAL PALLETS(MB5)MADE OF GALVANISED STEEL",
                "billOfEntry": []
                }},
                {{
                "itemNumber": "73102990",
                "quantity": "",
                "itemDetails": "73102990-RETURNABLE METAL PALLETS (MB5) MADE OF GALVANISED STEEL",
                "billOfEntry": []
                }}
            ]
            ```
            should be converted to:
                
            ```json
            [
                {{
                "itemNumber": "73102990",
                "quantity": "192.000000",
                "itemDescription": "RETURNABLE METAL PALLETS(MB5)MADE OF GALVANISED STEEL",
                "billOfEntry": []
                }}
            ]
            ```

            For example3 :

            ```json
            [
                {{
                'itemNumber': '73102990', 
                'quantity': '24.000NOS', 
                'itemDetails': "EMPTY GOODPACK METAL BOXES (BEING RETURN TO SUPPLIER NO COMMERCIAL VALUE) VALUE DECLARED FOR CUSTOM PURPOSE ONLY) 24.000NOS 70.00000per1 NOS 1680.00000 118440.00"
                }}, 
                {{
                'itemNumber': '73102990', 
                'quantity': '1.000NOS', 
                'itemDetails': "(IMPORTED BE NO.4029154 DT: 11.07.2019 2 1.000NOS 0.00001per1 NOS 0.00000 0.00"
                }}, 
                {{
                'itemNumber': '73102990', 
                'quantity': '1.000NOS', 
                'itemDetails': "THE METAL BOXES WERE RECEIVED FREE OF CH ARGE VIDE THE BELOW MENTIONED INVOICE RE-EXPORT OF 1.000NOS 0.00001per1 NOS 0.00000 0.00"
                }}, 
                {{
                'itemNumber': '73102990', 
                'quantity': '1.000NOS', 
                'itemDetails': "RETURNABLE RETAL BOX TYPE MBS "GOODPACK"D UTY ON IMPORTATION ON THESE BOXES NOT PAID UNDER NTFN.NO.104/94 CUST.DT 16.03.94 1.000NOS 0.00001per1 NOS 0.00000 0.00"
                }}
            ]
            ```
            should be converted to:
                
            ```json
            [
                {{
                'itemNumber': '73102990', 
                'quantity': '24.000NOS', 
                'itemDescription': "EMPTY GOODPACK METAL BOXES (BEING RETURN TO SUPPLIER NO COMMERCIAL VALUE) VALUE DECLARED FOR CUSTOM PURPOSE ONLY) 24.000NOS 70.00000per1 NOS 1680.00000 118440.00",
                'billOfEntry': []
                }},
                {{
                'itemNumber': '73102990', 
                'quantity': '1.000NOS',
                'itemDescription': "(IMPORTED BE NO.4029154 DT: 11.07.2019 2 1.000NOS 0.00001per1 NOS 0.00000 0.00", 
                'billOfEntry': [
                    {{
                    "billOfEntryNumber": "4029154",
                    "billOfEntryDate": "11.07.2019"
                    }}
                ]
                }},
                {{
                'itemNumber': '73102990', 
                'quantity': '1.000NOS', 
                'itemDescription': "THE METAL BOXES WERE RECEIVED FREE OF CH ARGE VIDE THE BELOW MENTIONED INVOICE RE-EXPORT OF 1.000NOS 0.00001per1 NOS 0.00000 0.00",
                'billOfEntry': []
                }},
                {{
                'itemNumber': '73102990', 
                'quantity': '1.000NOS', 
                'itemDescription': "RETURNABLE RETAL BOX TYPE MBS "GOODPACK"D UTY ON IMPORTATION ON THESE BOXES NOT PAID UNDER NTFN.NO.104/94 CUST.DT 16.03.94 1.000NOS 0.00001per1 NOS 0.00000 0.00",
                'billOfEntry': []
                }},
            ]
            ```
            4. If no bill of entry details are found in an item’s `itemDetails`, the `billOfEntry` field should be an empty list (`[]`).
            5. If the `itemNumber` is not present in the values of items, check if an itemNumber of 8 digits is present in the `itemDetails`. If yes, then consider that as the `itemNumber`.
            6. Ensure the JSON structure is maintained, with the new `billOfEntry` field added to each item.
            7. Preserve the structure and content of the original JSON, only adding the required field.
            8. Return the modifies json only. 
            9. No python code, no explanation, no comments.
            10. invoice Date is in dd/mm/yyyy format. reset it if it is in any other format.
            11. shippingBillDate is in dd/mm/yyyy format. reset it if it is in any other format.
            12. itemDescription should be fetched from the `itemDetails` field by removing the itemNumber.
            13. If itemDescription is not present in the input JSON object, then return empty list.
            14. In items dont return fields with same itemdescription.

            #Here is the input list of JSON object:
            # {final_combined_data["items"]} convert this."""

            
            response = call_gemini_model(prompt)
            # print("\nresponse", response)
            clean_json_str = response.replace("```json", "").replace("```", "").strip()
            response = json.loads(clean_json_str)
            print("\nresponse", response)

            final_combined_data["items"] = response
            print("\nfiltered_data", final_combined_data)

            for item in final_combined_data["items"]:
                if "itemDetails" in item:
                    del item["itemDetails"]
                    
            print("\nresponse", final_combined_data)
            return final_combined_data

        else:
            raise HTTPException(status_code=400, detail="Invalid file format. Only PDF files are supported.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    finally:
        # Clean up temporary files
        for image_path in image_paths:
            os.remove(image_path)
        os.remove(file_location)