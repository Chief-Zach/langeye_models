import os
import time
import pandas as pd
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated
from PIL import Image

if __name__ == '__main__':

    models = []
    images = []
    model_stats = {}

    start = time.time()

    models.append(YOLO("yolov8x-oiv7.pt"))
    model_stats[0] = {"name": models[0].model_name, "load_time": time.time() - start}

    start = time.time()
    models.append(YOLO("yolo11x.pt"))
    model_stats[1] = {"name": models[1].model_name, "load_time": time.time() - start}

    for filename in os.listdir("images"):
        images.append(Image.open(f'images/{filename}'))

    for count, model in enumerate(models):
        prediction_average = 0
        annotation_average = 0
        total_results = 0
        for image in images:
            start = time.time()
            results = model.predict(image, conf=0.75)
            total_results += len(results)

            prediction_average += time.time() - start

            for r in results:
                start = time.time()
                annotator = Annotator(image)
                boxes = r.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                    c = box.cls
                    d = box.conf
                    annotator.box_label(b, f"{model.names[int(c)]} {float(d) * 100}%")
                annotation_average += time.time() - start


        model_stats[count]["annotation_average"] = annotation_average / total_results
        model_stats[count]["prediction_average"] = prediction_average/len(images)

    df = pd.DataFrame(model_stats)
    # print(df)

    df.columns = df.iloc[0]

    # Drop the first row since it's now the header
    df = df[1:]

    # Print the cleaned DataFrame
    df.to_csv("model_stats.csv")