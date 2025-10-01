from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse

from app.services.gemini_pdf import process_gemini_pdf

router = APIRouter()

@router.post("/process-pdf-gemini/")
async def process_document(file: UploadFile = File(...)):

    result = await process_gemini_pdf(file)
    return JSONResponse(content=result)
