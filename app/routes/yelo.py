import json

from app.services.yelo import process_yelo
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse

router = APIRouter()

@router.post("/process-pdf-yelo/")
async def process_pdf_yelo(file: UploadFile = File(...)):
    # Save the uploaded file to a temporary location
    result = await process_yelo(file)

    # Return the JSON response
    return JSONResponse(content=json.loads(result))