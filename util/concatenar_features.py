import os
import numpy as np

if __name__ == "__main__":
    src_dir = '/projects/jeff/TUMGAIDfeatures'
    dst_dir = '/projects/jeff/TUMGAIDfeatures_FULL'
    #src_dir = '/projects/jeff/TUMGAIDfeatures_LT'
    #dst_dir = '/projects/jeff/TUMGAIDfeatures_LT_FULL'

    dir_list = next(os.walk(src_dir))[1]

    for diret in dir_list:
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
            try:
                face_feat = np.loadtxt(faces[i], delimiter=',')
            except:
                print('Diretório {} tem mais GEIs que facecrops disponíveis.'.format(diret))
                continue
            reid_feat = np.concatenate((gei_feat, face_feat))
            new_filename = gei.replace(src_dir, dst_dir).replace('.gei', '.feat')
            os.makedirs(os.path.dirname(new_filename) + "/", exist_ok=True)
            np.savetxt(new_filename, reid_feat, delimiter=',')
    print('Finalizado')
