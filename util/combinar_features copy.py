import os
import numpy as np

if __name__ == "__main__":
    #src_dir = '/projects/jeff/TUMGAIDfeatures'
    #dst_dir = '/projects/jeff/TUMGAIDfeatures_COMB'
    src_dir = '/projects/jeff/TUMGAIDfeatures_LT'
    dst_dir = '/projects/jeff/TUMGAIDfeatures_LT_COMB'

    dir_list = next(os.walk(src_dir))[1]
    k = 0
    for diret in dir_list:
        k += 1
        geis = []
        faces = []
        for file in os.listdir(src_dir+'/'+diret):
            if file.endswith(".gei"):
                gei_file = src_dir + '/' + diret + '/' + file
                geis.append(gei_file)
            if file.endswith(".fac"):
                face_file = src_dir + '/' + diret + '/' + file
                faces.append(face_file)
        for i, gei in enumerate(geis):
            gei_feat = np.loadtxt(gei, delimiter=',')
            for j, face in enumerate(faces):
                try:
                    face_feat = np.loadtxt(faces[i], delimiter=',')
                except:
                    continue
                reid_feat = np.concatenate((gei_feat, face_feat))
                new_filename = gei.replace(src_dir, dst_dir).replace('.gei', '.feat')
                new_filename = new_filename[:-5] + '-'+str(i)+'x'+str(j) + new_filename[-5:]
                os.makedirs(os.path.dirname(new_filename) + "/", exist_ok=True)
                np.savetxt(new_filename, reid_feat, delimiter=',')
        if k % 10 == 0:
            print('Diretórios processados: {}/{}.'.format(k, len(dir_list)))

    print('Finalizado')
