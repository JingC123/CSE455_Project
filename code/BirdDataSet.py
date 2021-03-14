import h5py
import helpers
import numpy as np
from pathlib import Path
import torch
import glob

import torchvision
from torch.utils import data
import matplotlib.pyplot as plt
from torchvision import transforms


class BirdDataset(data.Dataset):
    """
    Load image and label of birds stored in h5 format.
    The file is processed locally, which has the size of 8GB
    Data will be loaded to RAM and stored in a dict
    """

    def __init__(self, file_path, transform=None):
        super().__init__()
        self.folder_path = file_path
        self.transform = transform
        h5_file_names = glob.glob(self.folder_path)
        self.images = {}  # stores (name: image, label)
        self.image_index = []  # stores the name in a labeled list
        self.class_to_idx = {}
        classes = sorted([str(i) for i in range(555)])
        for i in range(len(classes)):
            self.class_to_idx[classes[i]] = i
        for name in h5_file_names:
            with h5py.File(name, "r") as f:
                images = f["birds"]
                names = f["file_name"]
                label = f["labels"]
                for i in range(len(f["birds"])):
                    k = names[i]
                    im = images[i].reshape(128, 128, 3)
                    im = np.transpose(im, (2, 0, 1))
                    v = (im, label[i])
                    self.images[k] = v
                    self.image_index.append(k)

    def __getitem__(self, index):
        target_name = self.image_index[index]
        image, label = self.images[target_name]
        image = torch.from_numpy(image)
        return image, label

    def __len__(self):
        return len(self.images)


def get_bird_data(augmentation=0):
    transform_train = transforms.Compose([
        # transforms.Resize(128),
        transforms.RandomCrop(128, padding=8, padding_mode='edge'),  # Take 128x128 crops from padded images
        transforms.RandomHorizontalFlip(),  # 50% of time flip image along y-axis
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
    ])
    trainset = BirdDataset("D:/Course/CSE_455/Final_Project/h5files/training/1.h5", transform=transform_train)
    # trainset = torchvision.datasets.ImageFolder(root='birds21wi/train', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.ImageFolder(root='birds21wi/test', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
    classes = open("birds21wi/names.txt").read().strip().split("\n")
    class_to_idx = trainset.class_to_idx
    idx_to_class = {int(v): int(k) for k, v in class_to_idx.items()}
    idx_to_name = {k: classes[v] for k, v in idx_to_class.items()}
    return {'train': trainloader, 'test': testloader, 'to_class': idx_to_class, 'to_name': idx_to_name}


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    num_epochs = 50
    loader_params = {'batch_size': 100, 'shuffle': True, 'num_workers': 6}
    birds = BirdDataset("D:/Course/CSE_455/Final_Project/h5files/training/*.h5")
    # data_loader = data.DataLoader(birds, **loader_params)

    image, label = birds[10]
    imshow(image)
    print(label)

    data = get_bird_data()

    dataiter = iter(data['train'])
    images, labels = dataiter.next()
    labels = labels.type(torch.LongTensor)
    print(images.shape)
    images = images[:8]
    print(labels.type())
    print(" ")
    print(len(labels))

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print("Labels:" + ', '.join('%9s' % data['to_name'][labels[j].item()] for j in range(8)))
    print([labels[j].item() for j in range(8)])
    print([data['to_class'][labels[j].item()] for j in range(8)])
    print([data['to_name'][labels[j].item()] for j in range(8)])