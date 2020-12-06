#from facenet_pytorch import MTCNN, InceptionResnetV1
from facenet_pytorch import fixed_image_standardization
import torch
import numpy as np
import os
import random
from PIL import Image
from torchvision import transforms
import torchvision.models as models
import modelos


def listar_imagens(basedir):
    # Obtem lista de imagens armazenadas no baseDir
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

    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(240),
        np.float32,
        transforms.ToTensor()
    ])

    # Create an inception net (in eval mode):
    #net = models.resnet18()
    net = modelos.min2019()
    # num_ftrs = net.fc.in_features  # num de features de entrada na FC
    # net.fc = torch.nn.Linear(num_ftrs, 305)  # nova FC com novo num de features de saida
    #net = net.to(device)
    net.load_state_dict(torch.load('/home/jeff/github/pesquisa/modelos/GEI_min2019_model_dict.pth'))

    # Imagens
    #basedir = '/projects/jeff/TUMGAIDimage_50_GEI'
    basedir = '/projects/jeff/TUMGAIDimage_LT_50_GEI'

    # carregar imagem aleatória
    fname, dname = listar_imagens(basedir)
    image_file = random.choice(fname)
    img = Image.open(image_file).convert('RGB')
    img = preprocess(img)

    real_id = image_file[len(basedir)+2:][:3]
    print('GEI correspondente à ID ' + real_id)

    img_probs = net(img.unsqueeze(0)).detach().cpu().numpy()[0]
    classe = np.argmax(img_probs)+1
    print("Predição: " + str(classe)+'/'+str(len(img_probs)))
