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


def getcycleLenght(distance_list):
    index_list = []
    for sub_list in distance_list:
        min_ = sub_list[0]
        trimmed_list = sub_list[:20]
        m = 9999
        index = 0
        for i in range(1, len(trimmed_list)):
            diff = abs(trimmed_list[0] - trimmed_list[i])
            if diff < m:
                m = diff
                index = i
        index *= 2
        if index > 24:
            index = random.randrange(20, 25)
        if index < 20:
            index = random.randrange(20, 25)
        index_list.append(index)
    return index_list


def getCroppedImage(sample_image, width_list_max, height_list_max):
    thresh = cv2.threshold(sample_image, 45, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)

        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        x = extLeft[0]
        y = extTop[1]

        mid_point = x + width_list_max // 2
        if (extTop[0] < mid_point):
            diff = mid_point - extTop[0]
            x = x - diff
        elif extTop[0] > mid_point:
            diff = extTop[0] - mid_point
            x += diff
        if x < 0:
            x = 0
        elif x > 240:
            x = 240
        cropped = sample_image[y:y + height_list_max,
                               x:x + width_list_max]  # [y_min:y_max,x_min:x_max]
    else:
        cropped = sample_image[width_list_max, height_list_max]
    return cropped


def getGEI(paths, widths, heights, cycles):
    newPathsuffix = '_GEI'
    print("Starting to save GEIs ...")
    image_stack = []
    for i in range(len(paths)):
        for j in range(len(paths[i])):
            gei_image_num = 0
            count = 1
            for k in range(len(paths[i][j])):
                cycle = cycles[i][j]
                path_imagem = paths[i][j][k]
                image_og = cv2.imread(path_imagem)
                max_w = max(max(widths[i]))
                max_h = max(max(heights[i]))
                try:
                    image = getCroppedImage(cv2.cvtColor(
                        image_og, cv2.COLOR_RGB2GRAY), max_w, max_h)
                except:
                    continue
                test_image_name = paths[i][j][k].replace(
                    basedir, basedir+'_teste')
                os.makedirs(os.path.dirname(
                    test_image_name) + "/", exist_ok=True)
                cv2.imwrite(test_image_name, image)

                image = cv2.resize(image, (240, 240))
                if count % cycle != 0:
                    image_stack.append(image)
                    count += 1
                else:
                    gei_image = np.zeros(image.shape, dtype=np.int)
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

    basedir = '/home/jeff/github/pesquisa/data/TUMGaid_silhouettes_101'
    distance_list, width_list, height_list, path_list = [], [], [], []
    cycleLength = []
    dist_sub, wid_sub, hgt_sub, path_sub = [], [], [], []
    distances, widths, heights, paths, cycles = [], [], [], [], []
    pessoas = next(os.walk(basedir))[1]
    pessoas = sorted(pessoas)
    for pessoa in pessoas:
        print(pessoa)
        dir_pessoa = basedir+'/'+pessoa
        capturas = next(os.walk(dir_pessoa))[1]
        capturas = sorted(capturas)
        for captura in capturas:
            dir_captura = basedir+'/'+pessoa+'/'+captura
            imagens = next(os.walk(dir_captura))[2]
            imagens = sorted(imagens)
            for imagem in imagens:
                path_imagem = basedir+'/'+pessoa+'/'+captura+'/'+imagem
                dst, wid, hgt = extractFeatures(path_imagem)
                dist_sub.append(dst)
                wid_sub.append(wid)
                hgt_sub.append(hgt)
                path_sub.append(path_imagem)
            distance_list.append(dist_sub)
            width_list.append(wid_sub)
            height_list.append(hgt_sub)
            path_list.append(path_sub)
            dist_sub, wid_sub, hgt_sub, path_sub = [], [], [], []
        distances.append(distance_list)
        widths.append(width_list)
        heights.append(height_list)
        paths.append(path_list)
        cycles.append(getcycleLenght(distance_list))
        distance_list, width_list, height_list, path_list = [], [], [], []
    getGEI(paths, widths, heights, cycles)
