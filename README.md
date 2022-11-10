# Yolov7 object detection model deployment using flask

## Web app

Simple app consisting of a form where you can upload an image, and see the inference result of the model in the browser. Run:

`$ python3 webapp.py --port 2808`

then visit http://localhost:2808/ in your browser:

## Rest API

Simple rest API exposing the model for consumption by another service. Run:

`$ python3 app.py --port 2808`

Then use [curl](https://curl.se/) to perform a request:

`$ curl -X POST -F image=@tests/zidane.jpg 'http://localhost:2808/v1/object-detection/yolov5s'`

The model inference results are returned:

```
[{'class': 0,
  'confidence': 0.8197850585,
  'name': 'person',
  'xmax': 1159.1403808594,
  'xmin': 750.912902832,
  'ymax': 711.2583007812,
  'ymin': 44.0350036621},
 {'class': 0,
  'confidence': 0.5667674541,
  'name': 'person',
  'xmax': 1065.5523681641,
  'xmin': 116.0448303223,
  'ymax': 713.8904418945,
  'ymin': 198.4603881836},
 {'class': 27,
  'confidence': 0.5661227107,
  'name': 'tie',
  'xmax': 516.7975463867,
  'xmin': 416.6880187988,
  'ymax': 717.0524902344,
  'ymin': 429.2020568848}]
```

## Run & Develop locally

Run locally for dev, requirements mostly originate from [yolov5](https://github.com/ultralytics/yolov5/blob/master/requirements.txt):

- `python3 -m venv venv`
- `source venv/bin/activate`
- `(venv) $ pip install -r requirements.txt`
- `(venv) $ python3 app.py --port 2808`

An example python script to perform inference using [requests](https://docs.python-requests.org/en/master/) is given in `tests/test_request.py`

## Docker

The example dockerfile shows how to expose the rest API:

```
# Build
docker build -t yolov5-flask .
# Run
docker run -p 2808:2808 yolov5-flask:latest
```