import base64
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import traceback
from ultralytics import YOLO

app = FastAPI()

app.mount("/frontend", StaticFiles(directory="frontend"), name="fronted")

model = YOLO("best.pt")

@app.post("/detect", response_class=Response)
async def detect_objects(file: UploadFile = File(...)):
    try:
        # Legge l'immagine dal file caricato
        content = await file.read()
        
        # Converte i bytes in un oggetto immagine
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Esegue l'inferenza con YOLO
        results = model(image)
        
        # Crea l'immagine annotata
        annotated_frame = results[0].plot()
        
        # Codifica l'immagine annotata in formato PNG
        success, buffer = cv2.imencode('.png', annotated_frame)
        
        if not success:
            return Response(content="Errore nella codifica dell'immagine", status_code=500)
        
        # Restituisce l'immagine come risposta
        return Response(content=buffer.tobytes(), media_type="image/png")
    
    except Exception as e:
        print(f"Errore durante l'elaborazione: {str(e)}")
        print(traceback.format_exc())
        return Response(content=f"Si è verificato un errore: {str(e)}", status_code=500)
    
@app.get("/")
def get_index():
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            base64_image = data.split(",")[1]
            image_bytes = base64.b64decode(base64_image)
            np_arr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            results = model(frame)
            annotated_frame = results[0].plot()
            
            _, buffer = cv2.imencode(".jpg", annotated_frame)
            annotated_base64 = base64.b64encode(buffer).decode("utf-8")

            await websocket.send_text("data:image/jpeg;base64," + annotated_base64)

    except Exception as e:
        print(f"Errore WebSocket: {e}")
    finally:
        await websocket.close()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)