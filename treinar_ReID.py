import modelos
#import torchvision
#import matplotlib.pyplot as plt
#import torchvision.models as models
import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from torch import optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from facenet_pytorch import training
import torch
np.random.seed(23)
torch.manual_seed(23)

# Parâmetros
data_dir = '/projects/jeff/TUMGAIDfeatures_FULL'
batch_size = 4
epochs = 50
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
dataset = datasets.DatasetFolder(data_dir, loader=modelos.features_loader, extensions=('feat'), transform=None)
img_inds = np.arange(len(dataset))
np.random.shuffle(img_inds)
train_inds = img_inds[:int(0.8 * len(img_inds))]
val_inds = np.setdiff1d(img_inds, train_inds)

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

# Usar CUDA
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# Configura a rede
net = modelos.ReID()
net = net.to(device)

optimizer = optim.SGD(net.parameters(), lr=0.001)
scheduler = MultiStepLR(optimizer, [5, 10])
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

torch.save(net.state_dict(), '/home/jeff/github/pesquisa/modelos/ReID_model_dict.pth')
writer.close()
