from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel
from deepface import DeepFace
import base64
import os
from io import BytesIO
import time

app = FastAPI()

# face verification from image file
@app.post("/faces/")
async def faces(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
):
    try:
        contents1 = await image1.read()
        
        if len(contents1) == 0:
            return {"message": "image 1 is empty"}
 
        contents2 = await image2.read()
        
        if len(contents2) == 0:
            return {"message": "image 2 is empty"}
        
        encode_image1 = base64.b64encode(contents1).decode('utf-8')
        encode_image2 = base64.b64encode(contents2).decode('utf-8')
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # create files
        img_path1 = os.path.join(current_dir, image1.filename)
        with open(img_path1, "wb") as img_file:
            img_file.write(contents1)
        img_path2 = os.path.join(current_dir, image2.filename)
        with open(img_path2, "wb") as img_file:
            img_file.write(contents2)
        
        # face verification
        result = DeepFace.verify(
            img1_path = image1.filename,
            img2_path = image2.filename,
            model_name = "Facenet",
            distance_metric = "cosine",
        )
        # distance adalah value dari hasil face verification
        distance = result['distance']
        # threshold adalah acuan untuk membandingkan hasil face verification
        threshold = result['threshold']
        
        # menghapus file yang sudah berhasil dibuat sebelumnya
        # setelah proses face recognition selesai, file akan dihapus
        os.remove(img_path1)
        os.remove(img_path2)
        
        return {
            "message": "Percobaan DeepFace Verification",
            "image1": image1.filename,
            "image2": image2.filename,
            "distance": distance,
            "threshold": threshold,
            "match": distance <= threshold,
            }
    except Exception as e:
        return {"error": str(e)}
    
# face verification from base64
@app.post("/faces_base64/")
async def faces_base64(
    image1: str = Form(...),
    image2: str = Form(...),
):
    try:
        if(len(image1) == 0 or len(image2) == 0):
            return {"message": "image is empty"}
        
        # untuk mendapatkan direktori saat ini
        # yang nantinya digunakan untuk menyimpan file image sementara
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        contents1 = base64.b64decode(image1)
        contents2 = base64.b64decode(image2)
        
        epoch_time = int(time.time())
        filename1 = str(epoch_time+1)+".jpg"
        filename2 = str(epoch_time+2)+".jpg"
        
        # create files
        img_path1 = os.path.join(current_dir, filename1)
        with open(img_path1, "wb") as img_file:
            img_file.write(contents1)
        img_path2 = os.path.join(current_dir, filename2)
        with open(img_path2, "wb") as img_file:
            img_file.write(contents2)
        
        # face verification
        result = DeepFace.verify(
            img1_path = filename1,
            img2_path = filename2,
            model_name = "Facenet",
            distance_metric = "cosine",
        )
        # distance adalah value dari hasil face verification
        distance = result['distance']
        # threshold adalah acuan untuk membandingkan hasil face verification
        threshold = result['threshold']
        
        # menghapus file yang sudah berhasil dibuat sebelumnya
        # setelah proses face recognition selesai, file akan dihapus
        os.remove(img_path1)
        os.remove(img_path2)
        
        return {
            "message": "Percobaan DeepFace Verification",
            "image1": filename1,
            "image2": filename2,
            "distance": distance,
            "threshold": threshold,
            "match": distance <= threshold,
            }
    except Exception as e:
        return {"error": str(e)}
