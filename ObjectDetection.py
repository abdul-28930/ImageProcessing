import gradio as gr
from PIL import Image
from transformers import pipeline
from PIL import ImageDraw

object_detector = pipeline("object-detection", model="facebook/detr-resnet-50")

# model_path = "C:\\Users\\abdul\\Documents\\genaiproj\\genai\\Models\\models--facebook--detr-resnet-50\\snapshots\\1d5f47bd3bdd2c4bbfa585418ffe6da5028b4c0b"
# object_detector = pipeline("object-detection", model=model_path)

def draw_bounding_boxes(image, detections):
    draw = ImageDraw.Draw(image)
    
    for detection in detections:
        # Extract bounding box coordinates and label
        box = detection['box']
        label = detection['label']
        xmin, ymin, xmax, ymax = box['xmin'], box['ymin'], box['xmax'], box['ymax']
        
        # Draw rectangle and label
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
        draw.text((xmin, ymin - 10), label, fill="red")
    
    return image


def detect_object(image):
    raw_image = Image.open(image)
    output = object_detector(raw_image)
    processed_image = draw_bounding_boxes(raw_image, output)
    return processed_image

# print(output)

gr.close_all()

# demo = gr.Interface(fn=summary, inputs="text", outputs="text")

demo = gr.Interface(
    fn=detect_object, 
    inputs=[gr.Image(label="Select Image", type="filepath")], 
    outputs=[gr.Image(label="Image with Bounding Box", type="pil")], 
    title="Object Detector", 
    theme="soft",
    description="This is an object detection model that detects objects in an image and draws bounding boxes around them.")
    
demo.launch(share=True)