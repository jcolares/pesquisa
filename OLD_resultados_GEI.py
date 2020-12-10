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


def obter_ranking(img_probs):
    ranking = []
    a = img_probs.tolist()
    for i in range(len(a)):
        menor = min(a)
        maior = max(a)
        pos_maior = a.index(maior)
        ranking.append(pos_maior+1)
        a[pos_maior] = menor
    return(ranking)


if __name__ == "__main__":

    preprocess = transforms.Compose([
        # transforms.Grayscale(num_output_channels=1),
        transforms.Resize(240),
        np.float32,
        transforms.ToTensor()
    ])

    # Create a neural net (in eval mode):
    net = modelos.min2019()
    net.load_state_dict(torch.load('/home/jeff/github/pesquisa/modelos/GEI_min2019_model_dict.pth'))

    # Imagens
    #basedir = '/projects/jeff/TUMGAIDimage_50_GEI'
    basedir = '/projects/jeff/TUMGAIDimage_LT_50_GEI'

    results = []
    execs = 100
    rank1 = 0
    rank5 = 0
    rank10 = 0
    rank20 = 0

    for i in range(execs):
        preds = []
        # carregar imagem aleatória
        fname, dname = listar_imagens(basedir)
        image_file = random.choice(fname)
        img = Image.open(image_file)  # .convert('RGB') #para a resnet, é necessário converter p 3 canais
        img = preprocess(img)

        real_id = int(image_file[len(basedir)+2:][:3])
        img_probs = net(img.unsqueeze(0)).detach().cpu().numpy()[0]
        ranking = obter_ranking(img_probs)
        rank = ranking.index(real_id)
        # print('Real: {}  Predição: {}  Rank 1:{} 5:{} 10:{}'.format(
        #    real_id, ranking[:5], rank == 0, rank < 5, rank < 10))
        # Coleta resultados em rank 1, 5, 10 e 20
        if rank == 0:
            rank1 += 1
        if rank < 5:
            rank5 += 1
        if rank < 10:
            rank10 += 1
        if rank < 20:
            rank20 += 1

    print('Teste executado com {} amostras'.format(execs))
    print('Acertos em rank-1:{},  rank-5:{},  rank-10:{},  rank-20:{}'.format(rank1, rank5, rank10, rank20))
    print('Acurácia em rank-1:{},  rank-5:{},  rank-10:{},  rank-20:{}'.format(rank1/execs,
                                                                               rank5/execs, rank10/execs, rank20/execs))
