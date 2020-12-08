import torch
import numpy as np
import os
import random
from PIL import Image
from torchvision import transforms
import modelos


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


def append_new_line(file_name, text_to_append):
    """Append given text as a new line at the end of file"""
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)


if __name__ == "__main__":

    preprocess = transforms.Compose([
        # transforms.Grayscale(num_output_channels=1),
        transforms.Resize(240),
        np.float32,
        transforms.ToTensor()
    ])

    # Create a neural net (in feature extraction mode):
    net = modelos.min2019feat()
    net.load_state_dict(torch.load('/home/jeff/github/pesquisa/modelos/GEI_min2019_model_dict.pth'))

    # Imagens
    #source_dir = '/projects/jeff/TUMGAIDimage_50_GEI'
    #dest_dir = '/projects/jeff/TUMGAIDfeatures'
    source_dir = '/projects/jeff/TUMGAIDimage_LT_50_GEI'
    dest_dir = '/projects/jeff/TUMGAIDfeatures_LT'

    fname, dname = listar_imagens(source_dir)

    i = 0
    for filename in fname:
        gait_img = Image.open(filename)
        gait_img = preprocess(gait_img)
        gait_features = net(gait_img.unsqueeze(0)).detach().cpu().numpy()[0]
        data_file = filename.replace(source_dir, dest_dir)[:-3]+'gei'
        os.makedirs(os.path.dirname(data_file) + "/", exist_ok=True)
        # save to csv file
        np.savetxt(data_file, gait_features, delimiter=',')
        i += 1
        if i % 1000 == 0:
            print('Processados {} arquivos de {}.'.format(i, len(fname)))
