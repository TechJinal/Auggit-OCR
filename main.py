from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import gemini_image, gemini_pdf, yelo

app = FastAPI(title="Auggit OCR API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(yelo.router, prefix="/model", tags=["Yelo"])
app.include_router(gemini_image.router, prefix="/model", tags=["Gemini PDF"])
app.include_router(gemini_pdf.router, prefix="/model", tags=["Gemini PDF"])