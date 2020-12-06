# Este programa separa o dataset TUMGAID em 2 partes:
# no diretório original, mantém apenas os arquivos para treinamento/validação normal
# no segundo diretótio, os arquivos da 2º captura, simulando o longo-prazo (Long-Term)

import os

src = '/projects/jeff/TUMGAIDimage'
dst = '/projects/jeff/TUMGAIDimage_LT'

LT_list = ['b03', 'b04', 'c01', 'c02', 'n07', 'n08', 'n09', 'n10', 'n11', 'n12', 's03', 's04']
fname = []
dname = []
for root, d_names, f_names in os.walk(src):
    for f in f_names:
        fname.append(os.path.join(root, f))
    for d in d_names:
        dname.append(os.path.join(root, d))
fname = sorted(fname)
dname = sorted(dname)

'''
for filename in fname:
    if filename[len(src)+6:][:3] in LT_list:
        new_filename = filename.replace(src, dst)
        os.makedirs(os.path.dirname(new_filename) + "/", exist_ok=True)
        os.system('mv {} {}'.format(filename, new_filename))
        print("{} - {}".format(filename[len(src)+6:][:3], new_filename))
'''
for dirname in dname:
    if dirname[-3:] in LT_list:
        new_dirname = dirname.replace(src, dst)
        os.makedirs(os.path.dirname(new_dirname) + "/", exist_ok=True)
        print(new_dirname)
        os.system('mv {} {}'.format(dirname, new_dirname))
