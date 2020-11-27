# Geração de GEIs
import os
import cv2
import imutils
import math
import random
import numpy as np


def extractFeatures(path_imagem):
    if path_imagem[-3:] == 'jpg':
        sample_image = cv2.imread(path_imagem, 0)
        thresh = cv2.threshold(sample_image, 45, 255, cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        if len(cnts) == 0:
            distance = 0
            w = 0
            hgt = 0
        else:
            c = max(cnts, key=cv2.contourArea)

            extLeft = tuple(c[c[:, :, 0].argmin()][0])
            extRight = tuple(c[c[:, :, 0].argmax()][0])
            extTop = tuple(c[c[:, :, 1].argmin()][0])
            extBot = tuple(c[c[:, :, 1].argmax()][0])

            maximum_y_r = -999
            maximum_x_r = -999

            for i in cnts[0]:
                for k in i:
                    if k[1] > 150:
                        if maximum_x_r < k[0]:
                            maximum_x_r = k[0]
                            maximum_y_r = k[1]

            maximum_x_l = maximum_x_r
            maximum_y_l = -999
            for i in cnts[0]:
                for k in i:
                    if k[1] > maximum_y_r - 10:
                        if maximum_x_l > k[0]:
                            maximum_x_l = k[0]
                            maximum_y_l = k[1]

            if maximum_y_l > maximum_y_r:
                b = maximum_y_l
            else:
                b = maximum_y_r
            dx2 = (maximum_x_r - extBot[0]) ** 2
            dy2 = (maximum_y_r - extBot[1]) ** 2
            distance = math.sqrt(dx2 + dy2)
            hgt = b - extTop[1]
            w = extRight[0] - extLeft[0]
        return distance, w, hgt


def image_resize(image, width=None, height=None, inter=cv2.INTER_NEAREST):
    # or cv2.INTER_AREA
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def getCenter(image):
    sample_image = image[:(image.shape[0]//2)]
    thresh = cv2.threshold(sample_image, 45, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    M = cv2.moments(c)
    centroid_x = int(M['m10']/M['m00'])
    return centroid_x


def getCrop(sample_image, out_h, out_w):

    # Resize to desired height
    # AQUI NAO sample_image = image_resize(sample_image1, height=out_h)
    out_img = np.zeros((out_h, out_w))
    thresh = cv2.threshold(sample_image, 45, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        if extLeft[1] > 0 and extRight[1] < sample_image.shape[1]:
            new_img = sample_image[extTop[1]:extBot[1]]
            try:
                new_img = image_resize(new_img, height=out_h)
            except:
                return out_img
            thresh = cv2.threshold(new_img, 45, 255, cv2.THRESH_BINARY)[1]
            cnts = cv2.findContours(
                thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key=cv2.contourArea)

            extLeft = tuple(c[c[:, :, 0].argmin()][0])
            extRight = tuple(c[c[:, :, 0].argmax()][0])

            centroid_x = getCenter(new_img)
            newExtLeft = int(centroid_x - out_w/2)
            newExtRight = int(out_w/2 + centroid_x)

            # sample_image.shape[1]:
            if newExtLeft+50 > 0 and newExtRight < 700:

                new_img = new_img[0:len(new_img), newExtLeft:newExtRight]

                pad_h = out_w - new_img.shape[1]
                pad_l = pad_h//2
                pad_r = pad_h//2 + pad_h % 2
                out_img = cv2.copyMakeBorder(
                    new_img, 0, 0, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)

    return out_img


def getGEI(paths):
    newPathsuffix = '_GEI'
    print("Starting to save GEIs ...")
    image_stack = []
    for i in range(len(paths)):
        print("ID "+"{0:03}".format(i))
        for j in range(len(paths[i])):
            gei_image_num = 0
            count = 1
            for k in range(len(paths[i][j])):
                path_imagem = paths[i][j][k]
                image_og = cv2.imread(path_imagem, cv2.IMREAD_GRAYSCALE)
                image = getCrop(image_og, 300, 300)
                # try:
                #     image = getCrop(image_og, 240, 240)
                # except:
                #     continue

                test_image_name = paths[i][j][k].replace(
                    basedir, basedir+'_teste')
                os.makedirs(os.path.dirname(
                    test_image_name) + "/", exist_ok=True)
                cv2.imwrite(test_image_name, image)

                if np.sum(image) > 0:
                    image_stack.append(image)
                    count += 1
            if np.sum(image_stack) == 0:
                continue
            gei_image = np.zeros((300, 300))  # , dtype=np.int)
            gei_image = np.mean(image_stack, axis=0)
            gei_image = gei_image.astype(np.int)
            gei_image_name = paths[i][j][k][:-7].replace(
                basedir, basedir+newPathsuffix)+"{0:03}".format(gei_image_num) + '.jpg'
            os.makedirs(os.path.dirname(
                gei_image_name) + "/", exist_ok=True)
            cv2.imwrite(gei_image_name, gei_image)
            gei_image_num += 1
            count += 1
            image_stack = []
    print("Geração de GEIs finalizada ...")


if __name__ == "__main__":

    basedir = '/projects/jeff/TUMGAIDimage_silhouettes_50'
    path_list, path_sub, paths = [], [], []
    pessoas = next(os.walk(basedir))[1]
    pessoas = sorted(pessoas)
    for pessoa in pessoas:
        dir_pessoa = basedir+'/'+pessoa
        capturas = next(os.walk(dir_pessoa))[1]
        capturas = sorted(capturas)
        for captura in capturas:
            dir_captura = basedir+'/'+pessoa+'/'+captura
            imagens = next(os.walk(dir_captura))[2]
            imagens = sorted(imagens)
            for imagem in imagens:
                path_imagem = basedir+'/'+pessoa+'/'+captura+'/'+imagem
                path_sub.append(path_imagem)
            path_list.append(path_sub)
            path_sub = []
        paths.append(path_list)
        path_list = []
    getGEI(paths)
