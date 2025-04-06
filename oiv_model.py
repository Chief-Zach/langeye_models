import os
import time
from imaplib import Int2AP

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated
from PIL import Image

start = time.time()
model = YOLO("yolov8x-oiv7.pt")
print("Model Load Time", time.time() - start)

for filename in os.listdir("images"):
    image = Image.open(f'images/{filename}')

    start = time.time()
    results = model.predict(image, conf=0.65)
    print("Prediction time", time.time() - start)

    start = time.time()
    for r in results:
        annotator = Annotator(image)
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
            c = box.cls
            if model.names[int(c)] == "Goggles": continue
            d = box.conf
            annotator.box_label(b, f"{model.names[int(c)]} {round(float(d) * 100, 2)}%")

    img = annotator.result()

    print("Annotation time", time.time() - start)
    annotated = Image.fromarray(img)
    annotated.save(f"oiv_annotated/{filename}")
    # annotated.show()

