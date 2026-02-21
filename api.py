import os
import io
import base64
import tempfile
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for server

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
from pathlib import Path

# Import our generator function
from vics_ir_generator import generate_guitar_ir

app = FastAPI(title="Vic's IR Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional: Mount a static directory to serve index.html directly
# app.mount("/", StaticFiles(directory="static", html=True), name="static")

def cleanup_temp_dir(dir_path: str):
    try:
        shutil.rmtree(dir_path)
    except Exception as e:
        print(f"Error cleaning up temp directory {dir_path}: {e}")

@app.post("/api/generate")
async def generate_ir(
    background_tasks: BackgroundTasks,
    piezo_file: UploadFile = File(...),
    mic_file: UploadFile = File(...),
    ir_length: int = Form(2048),
    smoothing: float = Form(0.333333333)
):
    # Create a temporary directory for this request
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save uploads to temp files
        piezo_path = os.path.join(temp_dir, piezo_file.filename or "piezo.wav")
        mic_path = os.path.join(temp_dir, mic_file.filename or "mic.wav")
        output_path = os.path.join(temp_dir, "output.wav")
        
        with open(piezo_path, "wb") as f:
            shutil.copyfileobj(piezo_file.file, f)
            
        with open(mic_path, "wb") as f:
            shutil.copyfileobj(mic_file.file, f)
            
        # Run generator
        generate_guitar_ir(
            piezo_path=piezo_path,
            mic_path=mic_path,
            output_path=output_path,
            ir_length=ir_length,
            smoothing=smoothing,
            plot=True # Always generate plot for web UI
        )
        
        # Read the output IR (WAV)
        with open(output_path, "rb") as f:
            wav_data = f.read()
            wav_base64 = base64.b64encode(wav_data).decode("utf-8")
            
        # Read the plot image (PNG)
        plot_path = output_path.rsplit('.', 1)[0] + '.png'
        with open(plot_path, "rb") as f:
            img_data = f.read()
            img_base64 = base64.b64encode(img_data).decode("utf-8")
            
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_dir, temp_dir)
        
        # Return success
        return JSONResponse(content={
            "success": True,
            "wav_base64": wav_base64,
            "img_base64": img_base64,
            "message": "IR generated successfully"
        })
        
    except Exception as e:
        # Ensure cleanup even on failure
        background_tasks.add_task(cleanup_temp_dir, temp_dir)
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
