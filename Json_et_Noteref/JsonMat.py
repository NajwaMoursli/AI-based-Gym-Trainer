import os
import numpy as np
import json

def json_to_matrix(path):
    PATH = path
    files=os.listdir(PATH) #files c un tableau qui contiendra le nom de tout les fichiers dans le répértoire
    files.sort() #on trie les fichiers
    #print(files)

    matrix=[]
    matrixx=[]
    matrixy=[]

    for f in files:
        with open(PATH+f) as json_file:
            data = json.load(json_file)
            d = data['people']
            #print(d[0])
            tab=d[0]["pose_keypoints_2d"]

            i = 0
            x_vect = []
            y_vect = []
            while i < len(tab)-1:
                x_vect.append(tab[i])
                y_vect.append(tab[i+1])
                i=i+3
        matrixx.append(x_vect)
        matrixy.append(y_vect)

    matrix.append(matrixx)
    matrix.append(matrixy)

    #print(matrix)
    matrix = np.array(matrix)
    print(matrix.shape)
    np.save(PATH[2:-1]+'.npy', matrix)

json_to_matrix('./P05_10/')
