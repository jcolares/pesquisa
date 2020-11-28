import os
import torch
from facenet_pytorch import MTCNN, extract_face
from PIL import Image

# Parâmetros
basedir = '/home/jeff/github/pesquisa/data/TUMGaid'

# Checar se há GPU disponível
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# Definir parâmetros do módulo MTCNN
mtcnn = MTCNN(keep_all=False, device=device, post_process=False)

# Obter lista de arquivos e diretórios
fname = []
dname = []
for root, d_names, f_names in os.walk(basedir):
    for f in f_names:
        fname.append(os.path.join(root, f))
    for d in d_names:
        dname.append(os.path.join(root, d))
fname = sorted(fname)
dname = sorted(dname)

# Detectar faces e salvar na pasta facecrops
facecrop = [it.replace(basedir, basedir+'_faces') for it in fname]
for f, filename in enumerate(fname):
    img = Image.open(filename)
    box, prob = mtcnn.detect(img)
    if prob[0] and prob[0] >= 0.5:
        extract_face(img, box[0], save_path=facecrop[f])

# Gerar silhuetas e salvar na pasta silhouettes
silhouette = [it.replace(basedir, basedir+'_silhouette') for it in fname]

# Gerar GEIs e salvar na pasta de GEIs
GEI_path = [it+'_GEI' for it in dname]
