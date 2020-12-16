import modelos
import torchvision
import matplotlib.pyplot as plt
import torchvision.models as models
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
data_dir = '/projects/jeff/TUMGAIDimage_50_GEI_normal_32'
batch_size = 4
epochs = 60
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
#train_inds = img_inds[:int(0.8 * len(img_inds))]
#val_inds = np.setdiff1d(img_inds, train_inds)
train_inds = img_inds[:int(0.7 * len(img_inds))]
val_test_inds = np.setdiff1d(img_inds, train_inds)
val_inds = img_inds[:int(0.5 * len(val_test_inds))]
test_inds = img_inds[int(0.5 * len(val_test_inds)):]

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
# DEBUG ONLY: device = torch.device('cpu')
print('Running on device: {}'.format(device))

# Configura a rede
num_classes = None  # None ou numero
net = modelos.min2019b(classify=True, num_classes=32)
#net = modelos.gaitRec(classify=True, num_classes=num_classes)
net = net.to(device)

optimizer = optim.SGD(net.parameters(), lr=0.0001)
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
if num_classes is None:
    fill = ''
else:
    fill = '_'+str(num_classes)

#torch.save(net.state_dict(), '/home/jeff/github/pesquisa/modelos/GEI_min2019b{}_model_dict.pth'.format(fill))
torch.save(net.state_dict(), '/home/jeff/github/pesquisa/modelos/GEI_min2019b_32{}_model_dict.pth'.format(fill))
#torch.save(net.state_dict(), '/home/jeff/github/pesquisa/modelos/GEI_gaitRec_normal{}_model_dict.pth'.format(fill))
writer.close()
