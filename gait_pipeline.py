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
from datetime import datetime
import training as test

np.random.seed(23)
torch.manual_seed(23)

# Parâmetros
data_dir = '/projects/jeff/TUMGAIDimage_50_GEI_normal'
data_dir_LT = '/projects/jeff/TUMGAIDimage_LT_50_GEI_normal'
num_classes = None
model = modelos.min2019b(num_classes=num_classes, classify=False)
model_name = model.__class__.__name__
dict_version = 'todasAsIDsNorm'
model_dict = '/home/jeff/github/pesquisa/modelos/GEI_{}_{}_{}_model_dict.pth'.format(
    model_name, '305' if num_classes == None else num_classes, dict_version)
train_epochs = 50
test_epochs = 3
batch_size = 4
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
# train_inds = img_inds[:int(0.8 * len(img_inds))]
# val_inds = np.setdiff1d(img_inds, train_inds)
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

# Configura o modelo
# net = modelos.model_name(classify=True, num_classes=32)
# net = model(classify=classify, num_classes=num_classes)
# net = modelos.gaitRec(classify=True, num_classes=num_classes)
model = model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.005)
scheduler = MultiStepLR(optimizer, [5, 10])
loss_fn = torch.nn.CrossEntropyLoss()
metrics = {
    'fps': training.BatchTimer(),
    'acc': training.accuracy
}
###########
# Train
#############
print('\n\n *** TREINAMENTO DO MODELO *** ')
print('modelo: {} '.format(model_name))
print('numero de classes: {}'.format(num_classes))
print('dataset treino: {}'.format(data_dir))
print('dataset teste - LP: {}'.format(data_dir_LT))
print('data: {}'.format(datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")))
print('dados do modelo: {}  -  versão: {}'.format(model_dict, dict_version))
writer = SummaryWriter()
writer.iteration, writer.interval = 0, 10

print('\n\nInitial')
print('-' * 10)


model.eval()
training.pass_epoch(
    model, loss_fn, val_loader,
    batch_metrics=metrics, show_running=True, device=device,
    writer=writer
)

for epoch in range(train_epochs):
    print('\nEpoch {}/{}'.format(epoch + 1, train_epochs))
    print('-' * 10)

    model.train()
    training.pass_epoch(
        model, loss_fn, train_loader, optimizer, scheduler,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

    model.eval()
    training.pass_epoch(
        model, loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )
if num_classes is None:
    fill = ''
else:
    fill = '_'+str(num_classes)

# torch.save(net.state_dict(), '/home/jeff/github/pesquisa/modelos/GEI_min2019b{}_model_dict.pth'.format(fill))
torch.save(model.state_dict(), model_dict)
# torch.save(net.state_dict(), '/home/jeff/github/pesquisa/modelos/GEI_gaitRec_normal{}_model_dict.pth'.format(fill))
writer.close()


############
# TESTE Curto Prazo
###########
model.classify = True
model.eval()

print('\n\n *** AVALIAÇÂO DO MODELO *** ')
print(' *** 1 - CURTO PRAZO')

# Transformações aplicadas ao dataset
trans_test = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(240),
    np.float32,
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(data_dir, transform=trans_test)

batch_size = 1

test_loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(test_inds)
)

metrics = {
    'r1': test.accuracy,
    'r5': test.rank5,
    'r10': test.rank10,
    'r20': test.rank20
}
for epoch in range(test_epochs):
    print('\nTeste {}/{}'.format(epoch + 1, test_epochs))
    print('-' * 10)

    test.pass_epoch(
        model, loss_fn, test_loader,
        batch_metrics=metrics, show_running=True, device=device
    )


############
# TESTE LONGO Prazo
###########
print('\n\n *** AVALIAÇÂO DO MODELO *** ')
print('*** 1 - LONGO PRAZO')
model.classify = True
model.eval()

dataset = datasets.ImageFolder(data_dir_LT, transform=trans_test)

img_inds = np.arange(len(dataset))

batch_size = 1

# Dataloader
test_LP_loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(img_inds)
)

metrics = {
    'r1': test.accuracy,
    'r5': test.rank5,
    'r10': test.rank10,
    'r20': test.rank20
}
for epoch in range(test_epochs):
    print('\nTeste {}/{}'.format(epoch + 1, test_epochs))
    print('-' * 10)

    test.pass_epoch(
        model, loss_fn, test_LP_loader,
        batch_metrics=metrics, show_running=True, device=device
    )
