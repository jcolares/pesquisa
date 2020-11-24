# Cópia  de inference_pretrained.py modificada para o dataset TUM GAID
import torch
import segmentation_models_pytorch as smp
import numpy as np
from torchvision import transforms
from PIL import Image
import os
from multiprocessing import Process
import time


def listar_imagens(basedir):
    # Obtém lista de imagens armazenadas no baseDir
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


def gerar_silhuetas(fnamex, dev):
    fname = fnamex
    # Define o path para salvar os arquivos
    silhouettes = [it.replace(basedir, basedir+'_silhouettes_101')
                   for it in fname]

    # Configura o modelo de CNN
    device = torch.device(dev)
    model = torch.hub.load(
        'pytorch/vision', 'deeplabv3_resnet101', pretrained=True)
    model.to(device)
    model.eval()

    for f, filename in enumerate(fname):
        # Carrega e preprocessa uma imagem
        input_image = Image.open(filename)
        input_tensor = preprocess(input_image)

        # Cria um minibatch contendo a imagem e envia para a GPU
        input_batch = input_tensor.unsqueeze(0).to(device)

        # Executa a predição da silhueta
        with torch.no_grad():
            output = model(input_batch)['out'][0]
        output_silhouette = output.argmax(0)
        output_silhouette[output_silhouette == 15] = 255

        # Converte a predição obtida em imagem e copia para a CPU
        silhouette_image = Image.fromarray(output_silhouette.byte().cpu().numpy()
                                           ).resize(input_image.size)

        # Salva silhueta no disco
        os.makedirs(os.path.dirname(silhouettes[f]) + "/", exist_ok=True)
        silhouette_image.save(silhouettes[f])


if __name__ == '__main__':

    basedir = '/home/jeff/github/pesquisa/data/TUMGaid'
    fname, _ = listar_imagens(basedir)

    # Define transformações que serão aplicadas às imagens
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Divide os dados em 3 partes
    fname1, fname2, fname3 = np.array_split(fname, 3)

    # Prepara e inicia 3 processos paralelos (para cada GPU)
    inicio = time.time()
    print('Processamento iniciado')
    print(time.strftime('%H:%M:%S', time.localtime()))
    p1 = Process(target=gerar_silhuetas, args=(fname1, 'cuda:0'))
    p1.start()
    p2 = Process(target=gerar_silhuetas, args=(fname2, 'cuda:1'))
    p2.start()
    p3 = Process(target=gerar_silhuetas, args=(fname3, 'cuda:2'))
    p3.start()

    # Tempo decorrido
    p1.join()
    p2.join()
    p3.join()
    print('Processamento concluído')
    print(time.strftime('%H:%M:%S', time.localtime()))
    tempo_total = time.time() - inicio
    print("Tempo total: %02dm:%02ds" % divmod(tempo_total, 60))
