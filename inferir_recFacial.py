from facenet_pytorch import MTCNN, InceptionResnetV1
from facenet_pytorch import fixed_image_standardization
import torch
import numpy as np
import os
import random
from PIL import Image
from torchvision import transforms


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
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])

    # Create an inception resnet (in eval mode):
    resnet = InceptionResnetV1(pretrained='vggface2', num_classes=305).eval()
    resnet.load_state_dict(torch.load('/home/jeff/github/pesquisa/modelos/faces_inceptionResnetV1_model_dict.pth'))
    # If using for VGGFace2 classification
    resnet.classify = True

    # Imagens
    #basedir = '/projects/jeff/TUMGAIDimage_facecrops'
    basedir = '/projects/jeff/TUMGAIDimage_LT_facecrops'

    # carregar imagem aleatória
    fname, dname = listar_imagens(basedir)
    image_file = random.choice(fname)
    img = Image.open(image_file)
    img = preprocess(img)

    real_id = image_file[len(basedir)+2:][:3]
    print('Imagem correspondente à ID ' + real_id)

    img_probs = resnet(img.unsqueeze(0)).detach().cpu().numpy()[0]
    classe = np.argmax(img_probs)+1
    print("Predição: " + str(classe)+'/'+str(len(img_probs)))
