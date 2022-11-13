"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""
import argparse
import re
from PIL import Image
import os.path
import sys
import cv2
import torch
import numpy as np
from flask import Flask, render_template, request, redirect, Response, json
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from numpy import random
from flask import jsonify
from detecting_object import letterbox
from operator import attrgetter
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from search_request import request_search
import base64
from os import environ as env
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

opt = {
    # Path to weights file default weights are for nano model
    "weights": "train_model/last.pt",
    "img-size": 640,  # default image size
    "conf-thres": 0.1,  # confidence threshold for inference.
    "iou-thres": 0.45,  # NMS IoU threshold for inference.
    "device": '0',  # device to run our model i.e. 0 or 0,1,2,3 or cpu
    "classes": "",  # list of classes to filter or None
    "precision": 0
}


db = firestore.client()
collection = db.collection("product_key_search")
app = Flask(__name__)

# load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load(opt['weights'], map_location=device)['model']
names = model.module.names if hasattr(model, 'module') else model.names
# Put in inference mode
model.float().eval()

if torch.cuda.is_available():
    # half() turns predictions into float16 tensors
    # which significantly lowers inference time
    model.half().to(device)



def pose_model(img_base64):
    np_arr =  np.fromstring(base64.b64decode(img_base64), np.uint8)
    # cv2.IMREAD_COLOR in OpenCV 3.1
    img0 = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    half = device.type != 'cpu'
    img_sz = opt['img-size']
    stride = int(model.stride.max())  # model stride
    img = letterbox(img0, img_sz, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=False)[0]
    # Apply NMS
    classes = None
    if opt['classes']:
        classes = []
        for class_name in opt['classes']:
            classes.append(opt['classes'].index(class_name))
    pred = non_max_suppression(
        pred, opt['conf-thres'], opt['iou-thres'], classes=classes, agnostic=False)
    res_detected = []
    gn = torch.tensor(img.shape)[[1, 0, 1, 0]]
    for i, det in enumerate(pred):
        s = ''
        s += '%gx%g ' % img.shape[2:]  # print string
        if len(det):
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], img0.shape).round()

        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class
            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

        for *xyxy, conf, cls in reversed(det):
            box = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist() 
            res_detected.append({
                "label": f'{names[int(cls)]}',
                "value": float(f'{conf:.2f}'),
                "box": box
            })
    if len(res_detected) > 0:
        max_precision = res_detected[0]
        for node in res_detected:
            if node["value"] > max_precision["value"]:
                max_precision = node
    else:
        max_precision = None
    return {"max": max_precision, "detected": res_detected}


def query_firebase(val):
    if(val["max"] != None): 
        max_precision_label = re.sub(r"_[0-9]+", '', val["max"]["label"])
    # Create a query against the collection
        query_ref = collection.where('Key', '==', max_precision_label)
        docs = query_ref.get()
        if (len(docs) > 0):
            return docs[0]._data
    return None
# Health check route
@app.route("/isalive")
def is_alive():
    print("/isalive request")
    status_code = Response(status=200)
    return status_code


@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        file = request.form.get("file")
        val = pose_model(file)
        text = query_firebase(val)
        products = request_search(text)
        return Response(
        response=json.dumps({
            "productsSuggested": products,
            "productsName": text,
            "objectDetected": val,
        }),
        status=200,
        mimetype="application/json"
    )
    return render_template("index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flask app exposing yolov7 models")
    parser.add_argument("--port", default=8080, type=int, help="port number")
    args = parser.parse_args()

    # debug=True causes Restarting with stat
    app.run(debug=True, host="0.0.0.0", port=8082)
