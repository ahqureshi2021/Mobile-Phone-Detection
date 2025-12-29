import gradio as gr
from ultralytics import YOLO
import numpy as np


# 1. Load the model
model = YOLO("https://github.com/ahqureshi2021/Mobile-Phone-Detection/raw/refs/heads/main/best.pt")
 

def predict_mobile(input_img):
    """
    Takes an image, runs YOLOv11 inference, and returns an annotated image.
    """
    if input_img is None:
        return None
        

    results = model.predict(source=input_img, conf=0.25)
    

    annotated_img_bgr = results[0].plot()
    

    annotated_img_rgb = annotated_img_bgr[..., ::-1]
    
    return annotated_img_rgb

# 2. Define the Gradio Interface
demo = gr.Interface(
    fn=predict_mobile,
    inputs=gr.Image(type="numpy", label="Upload Image for Mobile Phone Detection"),
    outputs=gr.Image(type="numpy", label="Detection Result"),
    title="Mobile Phone Detection System",
    description="Detecting mobile phone in real-time using YOLOv11.",
    flagging_mode="never"
)

# 3. Launch the app
if __name__ == "__main__":
    
    demo.launch(inbrowser=True, share=True)
    