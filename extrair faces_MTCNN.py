import os
import torch
from facenet_pytorch import MTCNN, extract_face
from PIL import Image
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


if __name__ == "__main__":

    # Parâmetros
    basedir = '/home/jeff/github/pesquisa/data/TUMGaid'

    # Checar se há GPU disponível
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    # Definir parâmetros do módulo MTCNN
    mtcnn = MTCNN(keep_all=False, device=device, post_process=False)

    # Obter lista de arquivos e diretórios
    fname, dname = listar_imagens(basedir)

    # Detectar faces e salvar na pasta facecrops
    inicio = time.time()
    print('Processamento iniciado')
    facecrop = [it.replace(basedir, basedir+'_faces') for it in fname]
    for f, filename in enumerate(fname):
        img = Image.open(filename)
        box, prob = mtcnn.detect(img)
        if prob[0] and prob[0] >= 0.5:
            extract_face(img, box[0], save_path=facecrop[f])
    print('Processamento concluído')
    print(time.strftime('%H:%M:%S', time.localtime()))
    tempo_total = time.time() - inicio
    print("Tempo total: %02dm:%02ds" % divmod(tempo_total, 60))
