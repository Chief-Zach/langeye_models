import time
from imaplib import Int2AP

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated
from PIL import Image

start = time.time()
model = YOLO("yolov8x-oiv7.pt")
print("Model Load Time", time.time() - start)
image = Image.open('bus.jpg')

start = time.time()
results = model.predict(image)
print("Prediction time", time.time() - start)

start = time.time()
for r in results:
    annotator = Annotator(image)
    boxes = r.boxes
    for box in boxes:
        b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
        c = box.cls
        annotator.box_label(b, model.names[int(c)])

img = annotator.result()

print("Annotation time", time.time() - start)
annotated = Image.fromarray(img)

annotated.show()

