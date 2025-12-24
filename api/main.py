from fastapi import FastAPI, UploadFile
import torch
import cv2
import numpy as np
from PIL import Image
import io
import base64
from src.model import UNet

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model.load_state_dict(torch.load("models/unet.pth", map_location=device))
model.eval()

def preprocess(image):
    image = np.array(image.resize((512, 512))) / 255.0  # Assuming 512x512
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float().to(device)
    return image

@app.post("/segment")
async def segment_image(file: UploadFile):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    tensor = preprocess(image)

    with torch.no_grad():
        mask = model(tensor)[0, 0].cpu().numpy()

    # Encode mask to base64
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    buffer = io.BytesIO()
    mask_img.save(buffer, format="PNG")
    mask_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return {"mask": mask_base64}