# Imports
from PIL import Image
import torch
from torchvision import transforms
from IPython.display import display
import time
from models.birefnet import BiRefNet


# # Option 1: loading BiRefNet with weights:
# from transformers import AutoModelForImageSegmentation
# birefnet = AutoModelForImageSegmentation.from_pretrained('zhengpeng7/BiRefNet', trust_remote_code=True)

# # Option-2: loading weights with BiReNet codes:
# birefnet = BiRefNet.from_pretrained('zhengpeng7/BiRefNet')

# # Option-3: Loading model and weights from local disk:
from utils import check_state_dict
device = "mps" if torch.backends.mps.is_available() else "cpu"

birefnet = BiRefNet(bb_pretrained=False)
state_dict = torch.load('/Users/emmanuelduah/mlai/BiRefNet/BiRefNet-general-epoch_244.pth', map_location=device)
state_dict = check_state_dict(state_dict)
birefnet.load_state_dict(state_dict)


# Load Model
torch.set_float32_matmul_precision(['high', 'highest'][0])

birefnet.to(device)
birefnet.eval()
print('BiRefNet is ready to use.')

# Input Data
transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

import os
from glob import glob

src_dir = '/Users/emmanuelduah/Pictures/Pics'
image_paths = glob(os.path.join(src_dir, '*'))
batch = str(int(time.time()))
masks_dir = f'/Users/emmanuelduah/mlai/BiRefNet/outputs/{batch}/masks'
dst_dir = f'/Users/emmanuelduah/mlai/BiRefNet/outputs/{batch}/images'

os.makedirs(masks_dir, exist_ok=True)
os.makedirs(dst_dir, exist_ok=True)
print('Found {} images to process.'.format(len(image_paths)))
for image_path in image_paths:
    if not image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        continue
    
    filename, extension = os.path.splitext(os.path.basename(image_path))
    
    print('Processing {} ...'.format(image_path))
    image = Image.open(image_path)
    input_images = transform_image(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()

    # Show Results
    pred_pil = transforms.ToPILImage()(pred)
    pred_pil.resize(image.size).save(image_path.replace(src_dir, masks_dir))
    
    # save image without background to images_output folder
    scale_ratio = 1024 / max(image.size)
    scaled_size = (int(image.size[0] * scale_ratio), int(image.size[1] * scale_ratio))
    
    image_masked = image.resize((1024, 1024))
    image_masked.putalpha(pred_pil)
    
    image_masked = image_masked.resize(scaled_size)
    
    filename = os.path.basename(image_path)
    
    result_path = os.path.join(dst_dir, f"{filename}.png")
    
    image_masked.save(result_path, format='PNG')
    

# Visualize the last sample:
# Scale proportionally with max length to 1024 for faster showing
scale_ratio = 1024 / max(image.size)
scaled_size = (int(image.size[0] * scale_ratio), int(image.size[1] * scale_ratio))
image_masked = image.resize((1024, 1024))
image_masked.putalpha(pred_pil)

display(image_masked.resize(scaled_size))
display(image.resize(scaled_size))
display(pred_pil.resize(scaled_size))

# save image without background to images_output folder
# image_masked.save(image_path.replace(src_dir, dst_dir), format='PNG')

# import requests
# from io import BytesIO


# image_url = "https://qph.cf2.quoracdn.net/main-qimg-d89362d538d350a4e4218a366e0d0425-pjlq"
# response = requests.get(image_url)
# image_data = BytesIO(response.content)
# image = Image.open(image_data)
# input_images = transform_image(image).unsqueeze(0).to(device)

# # Prediction
# with torch.no_grad():
#     preds = birefnet(input_images)[-1].sigmoid().cpu()
# pred = preds[0].squeeze()

# # Show Results
# pred_pil = transforms.ToPILImage()(pred)
# # Scale proportionally with max length to 1024 for faster showing
# scale_ratio = 1024 / max(image.size)
# scaled_size = (int(image.size[0] * scale_ratio), int(image.size[1] * scale_ratio))
# image_masked = image.resize((1024, 1024))
# image_masked.putalpha(pred_pil)
# display(image_masked.resize(scaled_size))
# display(image.resize(scaled_size))
# display(pred_pil.resize(scaled_size))

