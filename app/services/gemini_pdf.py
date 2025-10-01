import os
import pathlib
import json
import google.generativeai as genai

from fastapi import HTTPException
from dotenv import load_dotenv

from app.config import EXTRACTION_PROMPT

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is missing in environment variables.")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

async def process_gemini_pdf(file):
    try:
        # Save uploaded file temporarily
        temp_path = pathlib.Path(file.filename)
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Upload PDF to Gemini
        uploaded_file = genai.upload_file(path=temp_path)

        # Initialize model
        model = genai.GenerativeModel("gemini-2.0-flash")

        # Generate structured JSON
        response = model.generate_content([uploaded_file, EXTRACTION_PROMPT])
        if response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count
            total_tokens = response.usage_metadata.total_token_count
            print(f"Input Tokens: {input_tokens}, Output Tokens: {output_tokens}, Total Tokens: {total_tokens}")

        print("Model response:", response.text)

        clean_json = response.text.replace("```json", "").replace("```", "").strip()

        parsed_json = json.loads(clean_json)

        # Delete the uploaded file
        genai.delete_file(uploaded_file)

        # Cleanup temp file
        temp_path.unlink(missing_ok=True)

        if not response.text:
            raise HTTPException(status_code=500, detail="No response generated.")

        return parsed_json

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
