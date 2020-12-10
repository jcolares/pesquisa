import torch
#from facenet_pytorch import training
import training
from torch.utils.data import DataLoader, SubsetRandomSampler, BatchSampler
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
import modelos
import os
import numpy as np
np.random.seed(23)
torch.manual_seed(23)

# Parâmetros
data_dir = '/projects/jeff/TUMGAIDimage_50_GEI'
batch_size = 30
epochs = 5
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
val_inds = np.setdiff1d(img_inds, train_inds)

# Dataloaders
val_loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(val_inds)
    #sampler=BatchSampler(val_inds, False)
)

# Usar CUDA
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# Modelo
net = modelos.min2019()
net = net.to(device)
net.load_state_dict(torch.load('/home/jeff/github/pesquisa/modelos/GEI_min2019_model_dict.pth'))


loss_fn = torch.nn.CrossEntropyLoss()
metrics = {
    'r1': training.accuracy,
    'r5': training.rank5,
    'r10': training.rank10,
    'r20': training.rank20
}
net.eval()

for epoch in range(epochs):
    print('\nEpoch {}/{}'.format(epoch + 1, epochs))
    print('-' * 10)

    net.eval()
    training.pass_epoch(
        net, loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=device
    )
