import os
import time
from imaplib import Int2AP

from ultralytics import YOLOE
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated
from PIL import Image

start = time.time()
model = YOLOE("yoloe-11l-seg.pt")

classes = [
    "Sneakers", "Chair", "Hat", "Lamp", "Glasses", "Bottle", "Desk", "Cup",
    "Cabinet/shelf", "Handbag/Satchel", "Bracelet", "Plate", "Picture/Frame", "Helmet",
    "Book", "Gloves", "Storage box", "Leather Shoes", "Potted Plant", "Bowl/Basin",
    "Pillow", "Vase", "Microphone", "Necklace", "Ring", "Wine Glass", "Belt",
    "Monitor/TV", "Backpack", "Speaker", "Watch", "Tie", "Trash bin Can", "Slippers",
    "Stool", "Couch", "Sandals", "Basket", "Drum", "Pen/Pencil", "Carpet",
    "Cell Phone", "Bread", "Camera", "Towel", "Stuffed Toy", "Candle", "Laptop",
    "Bed", "Mirror", "Power outlet", "Sink", "Apple", "Air Conditioner", "Knife",
    "Fork", "Spoon", "Clock", "Pot", "Cake", "Hanger",
    "Blackboard/Whiteboard", "Napkin", "Toiletry", "Keyboard", "Tomato",
    "Lantern", "Fan", "Green Vegetables", "Banana", "Pumpkin", "Nightstand",
    "Tea pot", "Telephone", "Remote", "Refrigerator", "Oven", "Lemon",
    "Piano", "Pizza", "Gas stove", "Donut", "Bow Tie", "Carrot", "Toilet",
    "Strawberry", "Shovel", "Pepper", "Computer Box", "Toilet Paper",
    "Cleaning Products", "Chopsticks", "Microwave", "Cutting/chopping Board",
    "Coffee Table", "Side Table", "Scissors", "Marker", "Pie", "Ladder",
    "Cookies", "Radiator", "Grape", "Potato", "Sausage", "Violin", "Egg",
    "Candy", "Bathtub", "Wheelchair", "Golf Club", "Briefcase", "Cucumber",
    "Cigar/Cigarette", "Paint Brush", "Pear", "Hamburger", "Extractor",
    "Extension Cord", "Tong", "Folder", "Mask", "Kettle", "Coffee Machine",
    "Onion", "Green beans", "Projector", "Printer", "Watermelon", "Saxophone",
    "Tissue", "Toothbrush", "Ice cream", "Cello", "French Fries", "Scale",
    "Trophy", "Cabbage", "Hot dog", "Blender", "Peach", "Rice", "Wallet/Purse",
    "Tape", "Tablet", "Cosmetics", "Trumpet", "Key", "Fishing Rod",
    "Medal", "Flute", "Brush", "Corn", "Lettuce", "Garlic", "Green Onion",
    "Sandwich", "Nuts", "Induction Cooker", "Broom", "Plum", "Goldfish",
    "Kiwi fruit", "Router/modem", "Toaster", "Shrimp", "Sushi", "Cheese",
    "Notepaper", "Cherry", "Pliers", "CD", "Pasta", "Hammer", "Avocado",
    "Hami melon", "Flask", "Mushroom", "Screwdriver", "Soap", "Recorder",
    "Eggplant", "Board Eraser", "Coconut", "Tape Measure/Ruler", "Showerhead",
    "Globe", "Chips", "Steak", "Stapler", "Pomegranate", "Dishwasher",
    "Meatball", "Rice Cooker", "Calculator", "Papaya", "Game board", "Mop",
    "Radish", "Baozi", "Spring Rolls", "Pencil Case", "Red Cabbage", "Asparagus",
    "Noodles", "Comb", "Dumpling", "Table Tennis paddle", "Eraser", "Lipstick",
    "Cosmetics Mirror", "Table Tennis", "Mouse"
]


model.set_classes(classes, model.get_text_pe(classes))

print("Model Load Time", time.time() - start)
times = []
for filename in os.listdir("images"):
    image = Image.open(f'images/{filename}')

    start = time.time()
    results = model.predict(image, conf=0.65)
    times.append(time.time() - start)

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

    annotated = Image.fromarray(img)
    annotated.save(f"yoloe_annotated/{filename}")
    # annotated.show()

print("Average prediction time", sum(times[1:])/len(times))