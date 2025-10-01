from app.services.gemini_image import process_gemini_image
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse

router = APIRouter()

@router.post("/process-image-gemini/")
async def process_document(file: UploadFile = File(...)):

    result = await process_gemini_image(file)
    return JSONResponse(content=result)
