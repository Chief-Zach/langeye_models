import time

from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-03-27",
    trust_remote_code=True,
    # Uncomment to run on GPU.
    device_map={"": "cuda"}
)

image = Image.open("images/image7.jpeg")

# # Captioning
# print("Short caption:")
# print(model.caption(image, length="short")["caption"])
#
# print("\nNormal caption:")
# for t in model.caption(image, length="normal", stream=True)["caption"]:
#     # Streaming generation example, supported for caption() and detect()
#     print(t, end="", flush=True)
# print(model.caption(image, length="normal"))
#
# # Visual Querying
# print("\nVisual query: 'How many people are in the image?'")
# print(model.query(image, "How many people are in the image?")["answer"])

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
    "Cosmetics Mirror", "Table Tennis"
]
# Object Detection
average = 0
count = 0
for things in classes:
    start = time.time()
    objects = model.detect(image, things)["objects"]
    print(f"Found {len(objects)} {things}(s)")
    average += time.time() - start
    count += 1
    print(f"Running average: {average / count}")
# # Pointing
# print("\nPointing: 'bottle'")
# points = model.point(image, "bottle")["points"]
# print(points)
# print(f"Found {len(points)} bottle(s)")
