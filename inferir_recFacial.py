from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
#import pandas as pd
import os
import random
from PIL import Image
from torchvision import transforms


def listar_imagens(basedir):
    # ObtÃ©m lista de imagens armazenadas no baseDir
    fname = []
    dname = []
    for root, d_names, f_names in os.walk(basedir):
        for f in f_names:
            fname.append(os.path.join(root, f))
        for d in d_names:
            dname.append(os.path.join(root, d))
    fname = sorted(fname)
    dname = sorted(dname)
    return fname, dname


if __name__ == "__main__":

    # Define transformações que serão aplicadas às imagens
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    # Parâmetros
    basedir = '/projects/jeff/TUMGAIDimage'

    # Checar se hÃ¡ GPU disponÃ­vel
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    fname, dname = listar_imagens(basedir)
    # MOdelo
    #resnet = InceptionResnetV1(pretrained='vggface2', classify=True).eval().to(device)
    #resnet = InceptionResnetV1(num_classes=300, classify=True).eval().to(device)
    resnet = InceptionResnetV1(
        pretrained='vggface2', num_classes=305, classify=True).eval().to(device)
    resnet.load_state_dict(torch.load(
        '/home/jeff/github/pesquisa/modelos/model_dict.pth'))

    # carregar imagem aleatória
    image_file = random.choice(fname)
    img = Image.open(image_file)
    #img_tensor = torch.tensor(img)
    img_tensor = transforms.ToTensor()(img)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    real_id = image_file[len(basedir)+2:][:3]
    print('Imagem correspondente à ID ' + real_id)

    #embedding= resnet(aligned).detach().cpu()
    img_probs = resnet(img_tensor).detach().cpu().numpy()[0]
    classe = np.argmax(img_probs)
    print("Predição: " + str(classe))
