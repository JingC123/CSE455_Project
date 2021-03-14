import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import PIL.Image as Image
from typing import Tuple

from IPython.core.display import display
from Simple_DLA import *
from torch.backends import cudnn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def save_model(model, file_name: str, path: str = "/content/drive/MyDrive/CSE_455_Project/") -> None:
    model_save_name = file_name
    path += file_name
    torch.save(model.state_dict(), path)
    print("model saved")


def load_simple_dla_model(file_name: str, path: str = "/content/drive/MyDrive/CSE_455_Project/") -> SimpleDLA:
    model = SimpleDLA()
    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    path += file_name
    model.load_state_dict(torch.load(path))
    return model


def test_image(model, loader, image_name, classes,
               image_path="/content/drive/MyDrive/CSE_455_Project/testing_image/") -> Tuple[str, int]:
    model_evl = model.eval()
    image_path += image_name
    image = Image.open(image_path)
    image = loader(image).float()
    image = torch.autograd.Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    image = image.cuda()
    output = model_evl(image)
    display(Image.open(image_path))
    conf, predicted = torch.max(output.data, 1)
    return classes[predicted.item()], conf.item()


