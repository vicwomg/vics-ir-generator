import os
import io
import base64
import tempfile
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for server

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import uuid
import json
import asyncio
from typing import Dict, Any
from pathlib import Path
from fastapi.responses import StreamingResponse

# Import our generator function
from vics_ir_generator import generate_guitar_ir

app = FastAPI(title="Vic's IR Generator")

# List of allowed origins - update this when you deploy!
ALLOWED_ORIGINS = [
    "http://127.0.0.1:8000",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    # Restrict CORS specifically to our allowed origins instead of "*"
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"], # Only allow GET and POST
    allow_headers=["*"],
)

# Optional: Mount a static directory to serve index.html directly
# app.mount("/", StaticFiles(directory="static", html=True), name="static")

# Global dictionary to store task progress and state
# Key: task_id (str), Value: Dict with keys: 'progress' (str), 'status' (float), 'result' (dict), 'error' (str)
tasks: Dict[str, Any] = {}

def cleanup_temp_dir(dir_path: str):
    try:
        shutil.rmtree(dir_path)
    except Exception as e:
        print(f"Error cleaning up temp directory {dir_path}: {e}")

def run_ir_generation(task_id: str, temp_dir: str, piezo_path: str, mic_path: str, output_path: str, ir_length: int, smoothing: float):
    """Background worker that runs the compute-heavy generator and updates the tasks dict."""
    try:
        # Define the callback that ties the generator to our task dict
        def progress_cb(msg: str, pct: int = None):
            tasks[task_id]["message"] = msg
            if pct is not None:
                tasks[task_id]["progress"] = pct

        generate_guitar_ir(
            piezo_path=piezo_path,
            mic_path=mic_path,
            output_path=output_path,
            ir_length=ir_length,
            smoothing=smoothing,
            plot=True,
            progress_callback=progress_cb
        )
        
        # Generator finished successfully. Read the files into base64.
        # Read the output IR (WAV)
        with open(output_path, "rb") as f:
            wav_base64 = base64.b64encode(f.read()).decode("utf-8")
            
        # Read the plot image (PNG)
        plot_path = output_path.rsplit('.', 1)[0] + '.png'
        with open(plot_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")
            
        # Store results
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["result"] = {
            "wav_base64": wav_base64,
            "img_base64": img_base64
        }
    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
    finally:
        # Always clean up the temp directory when the background worker finishes
        cleanup_temp_dir(temp_dir)

@app.post("/api/generate")
async def generate_ir(
    request: Request,
    background_tasks: BackgroundTasks,
    piezo_file: UploadFile = File(...),
    mic_file: UploadFile = File(...),
    ir_length: int = Form(2048),
    smoothing: float = Form(0.333333333)
):
    # Enforce Origin restriction for non-CORS direct API calls (e.g. cURL)
    origin = request.headers.get("origin") or request.headers.get("referer")
    
    # Get the host, prioritizing X-Forwarded-Host if behind a proxy
    host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    
    # If there is an origin/referer, and it's not in our allowed list AND it's not same-origin, block it
    if origin:
        # Strip trailing slashes that referer often adds
        clean_origin = origin.rstrip("/")
        
        # Check if it's in the hardcoded list
        is_allowed_origin = any(clean_origin.startswith(allowed) for allowed in ALLOWED_ORIGINS)
        
        # Check if it's same-origin (the origin contains the host domain)
        is_same_origin = host and host in clean_origin
        
        if not is_allowed_origin and not is_same_origin:
            raise HTTPException(status_code=403, detail="Access denied. Action must be initiated from the official frontend.")
    elif host and host.split(':')[0] not in ["127.0.0.1", "localhost"]:
        # If no origin, block curl/postman unless the host is strictly localhost testing
        raise HTTPException(status_code=403, detail="API access must originate from a browser.")
        
    # Create a temporary directory for this request
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save uploads to temp files
        piezo_path = os.path.join(temp_dir, "piezo.wav")
        mic_path = os.path.join(temp_dir, "mic.wav")
        output_path = os.path.join(temp_dir, "output.wav")
        
        # SECURITY FIX: Enforce 15MB size limit server-side
        MAX_SIZE = 15 * 1024 * 1024
        
        if piezo_file.size and piezo_file.size > MAX_SIZE:
             raise ValueError("Piezo file exceeds 15MB limit.")
        if mic_file.size and mic_file.size > MAX_SIZE:
             raise ValueError("Mic file exceeds 15MB limit.")
             
        with open(piezo_path, "wb") as f:
            shutil.copyfileobj(piezo_file.file, f)
            
        with open(mic_path, "wb") as f:
            shutil.copyfileobj(mic_file.file, f)
            
        # Initialize task state
        task_id = str(uuid.uuid4())
        tasks[task_id] = {
            "status": "running",
            "progress": 0,
            "message": "Initializing...",
            "result": None,
            "error": None
        }
        
        # Schedule the heavy processing to run in the background
        background_tasks.add_task(
            run_ir_generation, 
            task_id, temp_dir, piezo_path, mic_path, output_path, ir_length, smoothing
        )
        
        # Immediately return the task ID so the frontend can connect to the event stream
        return JSONResponse(content={
            "success": True,
            "task_id": task_id
        })
        
    except Exception as e:
        # Pre-execution failure
        cleanup_temp_dir(temp_dir)
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    """SSE Endpoint to push status updates to the client."""
    async def event_generator():
        while True:
            if task_id not in tasks:
                yield f"data: {json.dumps({'status': 'failed', 'error': 'Invalid task ID'})}\n\n"
                break
                
            task = tasks[task_id]
            # Yield the current state as a JSON string
            yield f"data: {json.dumps(task)}\n\n"
            
            if task["status"] in ["completed", "failed"]:
                # If done, we can optionally clean up the tasks dict here, 
                # or rely on a generic garbage collection later.
                break
                
            # Send updates every half second
            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

app.mount("/", StaticFiles(directory="static", html=True), name="static")

def start_dev_server():
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    start_dev_server()
