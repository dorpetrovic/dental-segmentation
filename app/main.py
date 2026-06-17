"""
Inference API
=============
Lightweight REST API for serving teeth segmentation predictions.
Accepts an image upload and returns detected tooth masks and bounding boxes.

Run locally:
    uvicorn app.main:app --reload --port 8000

Docker:
    docker build -t dental-seg . && docker run -p 8000:8000 dental-seg
"""

import io
import os
import sys
import json
import base64
import numpy as np
from pathlib import Path
import cv2
import torch
import torchvision.transforms.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

try:
    from fastapi import FastAPI, UploadFile, File, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from PIL import Image
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    print("FastAPI not installed. Run: pip install fastapi uvicorn pillow")

from utils.preprocessing import enhance_contrast
from models.teeth_segmentation import predict, load_inference_model
from configs.model_config import BINARY, CONF_THRESHOLD, BINARY_CLASSES, FDI_CLASSES

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
CLASS_NAMES = BINARY_CLASSES if BINARY else FDI_CLASSES
NUM_CLASSES = 2 if BINARY else 36


FDI_TO_QUADRANT = {
    **{str(fdi): (74,  144, 217) for fdi in range(11, 19)},  # UR blue
    **{str(fdi): (232, 112,  64) for fdi in range(21, 29)},  # UL orange
    **{str(fdi): (46,  204, 113) for fdi in range(31, 39)},  # LL green
    **{str(fdi): (155,  89, 182) for fdi in range(41, 49)},  # LR purple
}
 
def get_color(class_name: str) -> tuple:
    if class_name in ('tooth', '__background__'):
        return (74, 144, 217)
    return FDI_TO_QUADRANT.get(class_name, (170, 170, 170))

# App setup

if HAS_FASTAPI:
    app = FastAPI(
        title="Dental Teeth Segmentation API",
        description=(
            "Instance segmentation of individual teeth in dental panoramic X-rays "
            "using a fine-tuned Mask R-CNN model(torchversion ResNET50+FPN)"
        ),
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Lazy-loaded model (loaded on first request)
    _model = None

    def get_model(): # creates local _model that dies when function ends
        global _model #makes model persist between requests
        if _model is None:
            print("Loading model weigths...")
            _model = load_inference_model()
            print("Model ready.")
        return _model

    # Routes
    @app.get("/health")
    def health_check():
        """
        Health check endpoint.
        Returns dict[str,str]

        example: GET http://localhost:8000/health

        """
        return {"status": "ok", "model": "dental-teeth-segmentation-torch"}


    @app.post("/predict")
    async def predict(file: UploadFile=File(...)):
        """
        Segment teeth in an uploaded image.

        Arg:
            file - UploadFile = File(...), image file (JPEG)

        Returns JSONResponse with:
        - n_teeth:      Number of detected teeth instances.
        - boxes:        List of bounding boxes [y1, x1, y2, x2].
        - label:        class index per detection
        - class_name:   FDI tooth label or tooth if in binary mode
        - scores:       Confidence score for each detection.
        - masks_b64:    Base64-encoded PNG masks (one per tooth).
        - overlay_b64:  Base64-encoded overlay image with masks applied.
        """
        # Validate content type
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"Expected an image file, got '{file.content_type}'"
            )
        # Decode
        try:
            raw_bytes = await file.read()
            pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            image = np.array(pil_img)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decode image: {e}")

        # Preprocess
        enhanced = enhance_contrast(image, method="clahe")
        image_t = F.to_tensor(enhanced).unsqueeze(0).to(DEVICE) #(1,3,H,W)

        # Inference
        model = get_model()
        with torch.no_grad():
            outputs = model(image_t)

        output = outputs[0]
        boxes = output['boxes'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        labels = output['labels'].cpu().numpy()
        masks = output['masks'].cpu().numpy()

        #filter by confidence
        keep = scores >= CONF_THRESHOLD
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        masks = masks[keep]
        n = len(boxes)

        # Build overlay image
        overlay_img = enhanced.copy()
        for i in range(n):
            class_name = CLASS_NAMES[labels[i]] if labels[i]< len(CLASS_NAMES) else 'tooth'
            color = get_color(class_name)
            mask = (masks[i,0]>0.5)

            canvas = overlay_img.copy()
            canvas[mask] = color
            overlay_img = cv2.addWeighted(overlay_img,0.55,canvas,0.45,0)

            x1, y1, x2, y2 = map(int, boxes[i])
            cv2.rectangle(overlay_img,(x1,y1),(x2,y2), color, 2)

            # Label at mask centroid
            M = cv2.moments(mask.astype(np.uint8))
            cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else (x1 + x2) // 2
            cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else (y1 + y2) // 2
            cv2.putText(overlay_img, class_name, (cx - 10, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


        # Encode masks as base64 PNGs
        # Each mask is a binary grayscale image, 255-tooth px, 0 background
        masks_b64 = []
        for i in range(n):
            m = (masks[i,0]>0.5).astype(np.uint8) * 255
            buf = io.BytesIO()
            Image.fromarray(m).save(buf, format="PNG")
            masks_b64.append(base64.b64encode(buf.getvalue()).decode())

        # Encode Overlay image as base64 jpeg
        buf = io.BytesIO()
        Image.fromarray(overlay_img).save(buf, format="JPEG", quality=90)
        overlay_b64 = base64.b64encode(buf.getvalue()).decode()

        #Collect class names for respones
        class_names_out = [
            CLASS_NAMES[labels[i]] if labels[i] < len(CLASS_NAMES) else 'tooth'
            for i in range(n)
        ]
        return JSONResponse({
            "n_teeth": n,
            "boxes": boxes.tolist(),
            "labels": labels.tolist(),
            "class_names": class_names_out,
            "scores": scores.tolist(),
            "masks_b64": masks_b64,
            "overlay_b64": overlay_b64,
        })
