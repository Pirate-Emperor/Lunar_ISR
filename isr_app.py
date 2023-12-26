import gradio as gr
import requests
from PIL import Image
import os
import torch
import numpy as np
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

torch.hub.download_url_to_file('https://huggingface.co/spaces/jjourney1125/swin2sr/resolve/main/samples/00003.jpg', 'examples/00003.jpg')
torch.hub.download_url_to_file('https://huggingface.co/spaces/jjourney1125/swin2sr/resolve/main/samples/0855.jpg', 'examples/0855.jpg')
torch.hub.download_url_to_file('https://huggingface.co/spaces/jjourney1125/swin2sr/resolve/main/samples/ali_eye.jpg', 'examples/ali_eye.jpg')
torch.hub.download_url_to_file('https://huggingface.co/spaces/jjourney1125/swin2sr/resolve/main/samples/butterfly.jpg', 'examples/butterfly.jpg')
torch.hub.download_url_to_file('https://huggingface.co/spaces/jjourney1125/swin2sr/resolve/main/samples/chain-eye.jpg', 'examples/chain-eye.jpg')
torch.hub.download_url_to_file('https://huggingface.co/spaces/jjourney1125/swin2sr/resolve/main/samples/gojou-eyes.jpg', 'examples/gojou-eyes.jpg')
torch.hub.download_url_to_file('https://huggingface.co/spaces/jjourney1125/swin2sr/resolve/main/samples/shanghai.jpg', 'examples/shanghai.jpg')
torch.hub.download_url_to_file('https://huggingface.co/spaces/jjourney1125/swin2sr/resolve/main/samples/vagabond.jpg', 'examples/vagabond.jpg')

processor = AutoImageProcessor.from_pretrained("caidas/swin2SR-classical-sr-x2-64")
model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x2-64")

def enhance(image):
    # prepare image for the model
    inputs = processor(image, return_tensors="pt")

    # forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # postprocess
    output = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.moveaxis(output, source=0, destination=-1)
    output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
    
    return Image.fromarray(output)

title = "Lunar Image Super-Resolution"
description = ''''''
article = ""

# examples = [['00003.jpg'], ['0855.jpg'], ['ali_eye.jpg'], ['butterfly.jpg'], ['chain-eye.jpg'], ['gojou-eyes.jpg'], ['shanghai.jpg'], ['vagabond.jpg']]
examples = [f'examples/{name}' for name in sorted(os.listdir('examples'))]

gr.Interface(
    enhance, 
    gr.inputs.Image(type="pil", label="Input").style(height=260),
    gr.inputs.Image(type="pil", label="Ouput").style(height=240),
    title=title,
    description=description,
    article=article,
    examples=examples,
    ).launch(enable_queue=True)