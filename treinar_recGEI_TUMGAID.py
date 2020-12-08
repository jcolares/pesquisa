from facenet_pytorch import training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import os
import torchvision.models as models
import matplotlib.pyplot as plt
import torchvision
import modelos

# Parãmetros
data_dir = '/projects/jeff/TUMGAIDimage_50_GEI'
batch_size = 4
epochs = 100
workers = 8

# Transformações aplicadas ao dataset
trans = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(240),
    transforms.RandomCrop(240, pad_if_needed=True),  # padding=2),
    transforms.RandomHorizontalFlip(),
    np.float32,
    transforms.ToTensor()
])

# Dataset
dataset = datasets.ImageFolder(data_dir, transform=trans)
img_inds = np.arange(len(dataset))
np.random.shuffle(img_inds)
train_inds = img_inds[:int(0.8 * len(img_inds))]
val_inds = img_inds[int(0.8 * len(img_inds)):]

# Dataloaders
train_loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(train_inds)
)
val_loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(val_inds)
)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# Treimanento completo
net = modelos.min2019()
net = net.to(device)

#optimizer = optim.Adam(net.parameters(), lr=0.0005)
optimizer = optim.SGD(net.parameters(), lr=0.001)
scheduler = MultiStepLR(optimizer, [5, 10])
# loss_fn =
#loss_fn = torch.nn.BCELoss()
loss_fn = torch.nn.CrossEntropyLoss()
metrics = {
    'fps': training.BatchTimer(),
    'acc': training.accuracy
}

# Train
writer = SummaryWriter()
writer.iteration, writer.interval = 0, 10

print('\n\nInitial')
print('-' * 10)
net.eval()
training.pass_epoch(
    net, loss_fn, val_loader,
    batch_metrics=metrics, show_running=True, device=device,
    writer=writer
)

for epoch in range(epochs):
    print('\nEpoch {}/{}'.format(epoch + 1, epochs))
    print('-' * 10)

    net.train()
    training.pass_epoch(
        net, loss_fn, train_loader, optimizer, scheduler,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

    net.eval()
    training.pass_epoch(
        net, loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

torch.save(net.state_dict(), '/home/jeff/github/pesquisa/modelos/GEI_min2019_model_dict.pth')
writer.close()
