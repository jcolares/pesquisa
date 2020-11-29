# Para poder utilizar os pesos da rede já treinada com o dataset vggface2
# é preciso colocar os arquivos do TUMGAID em uma uma estrutura similar.
# Este arquivo faz esta conversão de estrutura dos diretórios, sem modificar o
# conteúdo dos arquivos.

import os


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

    source_root = '/projects/jeff/TUMGAIDimage_faces'
    dest_root = '/projects/jeff/TUMGAIDimage_facecrops'

    fname, _ = listar_imagens(source_root)
    for filepath in fname:
        src = filepath
        dst = dest_root + ''+os.path.dirname(filepath)[-9:]+'-'+os.path.basename(filepath)
        os.makedirs(os.path.dirname(dst) + "/", exist_ok=True)
        os.system('cp {} {}'.format(src, dst))
