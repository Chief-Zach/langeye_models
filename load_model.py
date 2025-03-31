import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
from PIL import Image

loader = transforms.Compose([transforms.ToTensor()])

model = torch.load("/Users/zach/PycharmProjects/csci4220/runs/detect/train3/weights/best.torchscript", weights_only=False)

image = Image.open("bus.jpg")
image = loader(image).float()
image = image.unsqueeze(0)

model.eval()
print(model(image))