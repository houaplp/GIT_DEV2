import os

import fftToTriangleV7 as tri

#pour avoir les graphes pdf, aller dans fftToTriangleV7 ligne 607

filelist = []            

for root, dirs, files in os.walk("./Sons", topdown=False):
    print(root)
    filelist = []
    for name in files: 
        if 'wav' in name :
            filename = os.path.join(root, name)
            filelist.append(filename)
            #tri.analyse_complete_fichier(0, 2500, 5000, filename, "V7")

print("nombre de fichiers trouvés :" , len(filelist))

filelist.sort()

for file in filelist:
    tri.analyse_complete_fichier(1000, 2500, 10000, file, "25")

