from flask import Flask, request, jsonify
import pickle
import numpy as np
import cv2
from transformers import BertTokenizer, BertModel
import torch
from flask_cors import CORS
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend JS

# Load models
with open('final_model', 'rb') as f:
    output_model = pickle.load(f)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
yolo = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

def preprocess(text, image_bytes):
    # Text features
    text_inputs = tokenizer(text, return_tensors="pt")
    text_outputs = bert_model(**text_inputs)
    text_hidden = text_outputs.last_hidden_state[0].cpu().detach().numpy()

    # Image (YOLO object detection)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    blob = cv2.dnn.blobFromImage(image, 1/255, (320, 320), (0,0,0), swapRB=True, crop=False)
    yolo.setInput(blob)
    output_layers_name = yolo.getUnconnectedOutLayersNames()
    layer_outputs = yolo.forward(output_layers_name)

    class_ids = []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.7:
                class_ids.append(class_id)

    object_labels = list({classes[i] for i in class_ids})
    object_text = " ".join(object_labels)

    object_inputs = tokenizer(object_text, return_tensors="pt")
    object_outputs = bert_model(**object_inputs)
    object_hidden = object_outputs.last_hidden_state[0].cpu().detach().numpy()

    max_len = max(len(text_hidden), len(object_hidden))
    text_hidden_padded = np.pad(text_hidden, ((0, max_len - len(text_hidden)), (0, 0)), 'constant')
    object_hidden_padded = np.pad(object_hidden, ((0, max_len - len(object_hidden)), (0, 0)), 'constant')

    attn_output = torch.nn.functional.softmax(
        torch.matmul(torch.tensor(object_hidden_padded), torch.tensor(text_hidden_padded).transpose(0, 1)) / np.sqrt(text_hidden_padded.shape[1]), dim=-1
    ) @ torch.tensor(text_hidden_padded)

    final_output = attn_output.numpy().flatten()
    if len(final_output) < 23040:
        final_output = np.pad(final_output, (0, 23040 - len(final_output)), 'constant')
    else:
        final_output = final_output[:23040]

    return np.expand_dims(final_output, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get("text", "")
    image = request.files.get("image", None)

    if not text or not image:
        return jsonify({"error": "Missing input"}), 400

    try:
        processed_input = preprocess(text, image.read())
        prediction = output_model.predict(processed_input)
        label = int(prediction[0][1] > prediction[0][0])  # 1 = real, 0 = fake

        return jsonify({
            "label": "real" if label == 1 else "fake",
            "softmax": prediction[0].tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
