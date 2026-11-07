import argparse
import io
import os
from typing import Union
import numpy as np
import onnxruntime as ort
from PIL import Image
from pydantic import BaseModel

INPUT_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASSES = ['blackspot', 'canker', 'fresh', 'grenning']

class Prediction(BaseModel):
    prediction_class: str | None = None
    prediction_result: dict[str, float] | None = None

def preprocess_image(image_input: Union[str, Image.Image, bytes]) -> np.ndarray:
    # 1. Open image and convert to RGB
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"No se encontró la imagen en: {image_input}")
        img = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, Image.Image):
        img = image_input.convert('RGB')
    elif isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input)).convert('RGB')
    else:
        raise TypeError("Parameter 'image_input' must be str (ruta), bytes, or object PIL.Image")
    # 2. Resize
    img = img.resize((INPUT_SIZE, INPUT_SIZE), Image.NEAREST)
    # 3. Convert to array and range to [0.0, 1.0]
    img_np = np.array(img, dtype=np.float32) / 255.0
    # 4. Normalize
    img_np = (img_np - MEAN) / STD
    # 5. Reorder (Height, Width, Channels) to (Channels, Height, Width)
    img_np = np.transpose(img_np, (2, 0, 1))
    # 6. Add Batch dimension -> (1, Channels, Height, Width)
    img_np = np.expand_dims(img_np, axis=0)
    return img_np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

def predict(image_input: Union[str, Image.Image], model_path: str):
    # 1. Prepare input
    input_data = preprocess_image(image_input)
    # 2. Load model
    session = ort.InferenceSession(model_path)
    # 3. Get input and output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    # 4. Inference
    raw_outputs = session.run([output_name], {input_name: input_data})
    logits = raw_outputs[0]
    # 5. Final prediction
    probabilities = softmax(logits)[0]
    predicted_class_idx = np.argmax(probabilities)
    # Print results
    img_identifier = image_input if isinstance(image_input, str) else "UploadFile (Bytes)"
    print(f"\n Results for the image: {img_identifier}")
    print("-" * 50)
    prediction_result = {}
    for idx, prob in enumerate(probabilities):
        class_name = CLASSES[idx] if idx < len(CLASSES) else f"Clase {idx}"
        print(f"  {class_name}: {prob*100:.2f}%")
        prediction_result[class_name] = round(float(prob * 100), 2)
    print("-" * 50)
    final_class = CLASSES[predicted_class_idx] if predicted_class_idx < len(CLASSES) else predicted_class_idx
    print(f"Prediction: **{final_class}** ({probabilities[predicted_class_idx]*100:.2f}%)")
    return Prediction(prediction_class=final_class, prediction_result=prediction_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferencia con modelo ONNX")
    parser.add_argument("--image", type=str, default="examples/example_blakspot.png", help="Path to local image (.jpg, .png)")
    parser.add_argument("--model", type=str, default="models/orange_diseases_model_1.onnx", help="Path to model .onnx")
    args = parser.parse_args()
    predict(args.image, args.model)