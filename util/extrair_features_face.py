from facenet_pytorch import MTCNN, InceptionResnetV1
from facenet_pytorch import fixed_image_standardization
import torch
import numpy as np
import os
import random
from PIL import Image
from torchvision import transforms


def listar_imagens(source_dir):
    # Obtem lista de imagens armazenadas no source_dir
    fname = []
    dname = []
    for root, d_names, f_names in os.walk(source_dir):
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
    resnet.classify = False

    # Arquivo
    #data_file = '/projects/jeff/TUMGAIDfeatures/face_features.dat'
    #os.makedirs(os.path.dirname(data_file) + "/", exist_ok=True)

    # Imagens
    #source_dir = '/projects/jeff/TUMGAIDimage_facecrops'
    #dest_dir = '/projects/jeff/TUMGAIDfeatures'
    source_dir = '/projects/jeff/TUMGAIDimage_LT_facecrops'
    dest_dir = '/projects/jeff/TUMGAIDfeatures_LT'

    fname, dname = listar_imagens(source_dir)
    gname = fname
    i = 0
    for filename in fname:
        face_img = Image.open(filename)
        face_img = preprocess(face_img)
        face_features = resnet(face_img.unsqueeze(0)).detach().cpu().numpy()[0]
        data_file = filename.replace(source_dir, dest_dir)[:-3]+'fac'
        os.makedirs(os.path.dirname(data_file) + "/", exist_ok=True)
        # save to csv file
        np.savetxt(data_file, face_features, delimiter=',')

        i += 1
        if i % 10000 == 0:
            print('Processados {} arquivos de {}.'.format(i, len(fname)))
