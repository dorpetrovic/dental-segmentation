"""
app/gradio_demo.py

Gradio demo for torchvision Mask R-CNN implementation for tooth 
detection and segmentation

"""

import os
import sys
import numpy as np
import gradio as gr
import torch
import torchvision.transforms.functional as F
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))



from utils.preprocessing import enhance_contrast
from models.teeth_segmentation import build_model, load_inference_model
from configs.model_config import BINARY, CONF_THRESHOLD, BINARY_CLASSES,FDI_CLASSES


WEIGHTS_PATH = PROJECT_ROOT/'outputs'/'results'/'maskrcnn_torch'/'best.pth'

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

CLASS_NAMES = BINARY_CLASSES if BINARY else FDI_CLASSES
NUM_CLASSES = 2 if BINARY else 36


FDI_TO_QUADRANT = {
    **{str(fdi): (74,144,217) for fdi in range(11,19)}, # UR blue
    **{str(fdi): (232, 112, 64) for fdi in range(21,29)}, # UL orange
    **{str(fdi): (46, 204, 113) for fdi in range(31,39)}, # LL green
    **{str(fdi): (155, 89, 182) for fdi in range(41,49)}, #LR purple

}
MODEL_METRICS = {
    'mAP@50':     91.6,
    'mAP@50-95':  67.9,
    'Count MAE':  6.2,
}

def get_color(class_name):
    if class_name in ('tooth','__background__'):
        return (74,144,217)
    return FDI_TO_QUADRANT.get(class_name, (170,170,170))


_model = None
def get_model():
    global _model
    if _model is None:
        _model = load_inference_model()
        print("Model loaded")
    return _model

#Main predict function - predicts on imported image

def predict(image):
    if image is None:
        return None, "NO IMAGE!"
    
    #Step1 - preprocess image
    enhanced = enhance_contrast(image, method='clahe')

    #Step2 - convert to tensor 
    image_tensor = F.to_tensor(enhanced).unsqueeze(0).to(DEVICE)
    
    #Step3 - run interface
    model = get_model()
    with torch.no_grad():
        outputs = model(image_tensor)
    
    output  = outputs[0]
    boxes = output['boxes'].cpu().numpy()
    scores = output['scores'].cpu().numpy()
    labels = output['labels'].cpu().numpy()
    masks = output['masks'].cpu().numpy() #(N,1,H,W)
    
    
    #Step4 - filter by confidence
    keep = scores >= CONF_THRESHOLD
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    masks = masks[keep]
    n = len(boxes)

    if n==0:
        return image, "No teeth detected"
    
    #Step5 - draw masks 
    output_img = enhanced.copy()

    for i in range(n):
        class_name = CLASS_NAMES[labels[i]] if labels[i]<len(CLASS_NAMES) else 'tooth'
        color = get_color(class_name)
        mask = (masks[i,0]>0.5)
        overlay = output_img.copy()
        overlay[mask] = color
        output_img = cv2.addWeighted(output_img,0.55,overlay,0.45,0)
        x1,y1,x2,y2 = map(int,boxes[i])
        cv2.rectangle(output_img,(x1,y1),(x2,y2),color,2)

        # Draw class label at mask centroid
        mask_u8 = mask.astype(np.uint8)
        M = cv2.moments(mask_u8)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.putText(output_img, class_name, (cx - 10, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    #Step6 - summary and mAP results
    summary = (
        f"------THIS IMAGE----------------------\n"
        f"Detected: {n} teeth\n"
        f"Mean confidence: {scores.mean():.2f}\n"
        f"Min confidence: {scores.min():.2f}\n"
        f"Max confidence: {scores.max():.2f}"
        f"\n"
        f"------MODEL PERFORMANCE (val set)-----\n"
        f"mAP@50: {MODEL_METRICS['mAP@50']}%\n"
        f"mAP@50-95: {MODEL_METRICS['mAP@50-95']}%\n"
        f"Count MAE: {MODEL_METRICS['Count MAE']} teeth\n"
    )

    return output_img, summary

#Pick 3 images form validation dataset
 
def _get_val_samples(n = 3):
    """Return [[path_str], ...] for the first n annotated val images that exist on disk."""
    ann_path = PROJECT_ROOT / 'data' / 'annotations' / 'val.json'
    img_dir  = PROJECT_ROOT / 'data' / 'processed'
    samples  = []
    if not ann_path.exists():
        return samples
    import json
    with open(ann_path) as f:
        coco = json.load(f)
    ann_ids = {a['image_id'] for a in coco['annotations']}
    for img in coco['images']:
        if img['id'] not in ann_ids:
            continue
        p = img_dir / img['file_name']
        if p.exists():
            samples.append([str(p)])
        if len(samples) == n:
            break
    return samples

def _get_test_samples():
    """
    Get images from test dataset that are images that the model has never
    seen before
    """
    test_dir = PROJECT_ROOT/'data'/'test'
    if not test_dir.exists():
        return []
    exts = {".jpg",'.jpeg','.png','.bmp','.tiff','.tif'}
    return [[str(p)] for p in sorted(test_dir.iterdir()) if p.suffix.lower() in exts]
 
VAL_SAMPLES = _get_val_samples(3)
TEST_SAMPLES = _get_test_samples()

#Gradio interface

with gr.Blocks(title="Dental Tooth Segmentation") as demo:
    gr.Markdown("""
    # Dental Tooth Segmentation using Mask R-CNN (torchvision ResNet50+FPN)
    Upload a panoramic X-ray image to detect individual teeth
""")
    
    with gr.Row():
        input_image = gr.Image(label="Upload image",type='numpy',height=300)
        output_image = gr.Image(label="Segmentated Image", type='numpy',height=300)
        output_text = gr.Textbox(label="Summary",lines=4)
    gr.Button('Detect tooth',variant='primary').click(
        fn = predict,
        inputs = [input_image],
        outputs = [output_image,output_text],
    )


    # Examples gallery — clicking a thumbnail loads it into input_image
    if VAL_SAMPLES:
        gr.Examples(
            examples=VAL_SAMPLES,
            inputs=[input_image],
            label="Sample images from validation set",
            examples_per_page=3,
        )
    if TEST_SAMPLES:
        gr.Examples(
            examples=TEST_SAMPLES,
            inputs=[input_image],
            label="Test images unseen before, new Xray",
            examples_per_page=3,
        )

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=7860,share=False)        