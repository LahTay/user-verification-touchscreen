import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import json
import random_data_generator

def open_file(file_name: str):
    data = []
    with open(file_name) as f:
        data = json.load(f)
    return data


"""
OPIS DANYCH:
size - miejsce, w którym rysujemy (kwadrat wielkości size*size)
gyro - dane z żyroskopu
acc - dane z akcelerometru
x - lista współrzędnych x wzoru
y - lista współrzędnych y wzoru
time - czas mierzony pomiędzy punktami w *nanosekundach*, czyli np. [0, 24337500, 14337500 ...]
rawTime - aktualny czas pomierzony w trakcie rysowania, czyli np. [131094225706037, 131094249974787, 131094266429995, ...]
aproperties - lista właściwości akcelerometru, kolejno [resolution, maxRange, minDelay, maxDelay]
gproperties - lista właściwości żyroskopu, kolejno [resolution, maxRange, minDelay, maxDelay]

Czas na preprocessing.

"""
def read_data(limit):

    # file_data = open_file(filename)
    #
    # print(file_data.keys())
    file_data = random_data_generator.generate(np.random.randint(300,900))
    XX=[]
    YY=[]
    ZZ=[]
    rot = np.array([0,0,0])
    mov = np.array([0,0,0])
    acc0 = np.array(file_data['acc'][0])
    for i in range(1,len(file_data['x'])):
        rot = rot + (file_data['time'][i]*(10**(-9)))**2*np.array(file_data['gyro'][i])/2
        r = R.from_euler('xyz', rot)
        mov = mov + (np.array(file_data['acc'][i])-r.apply(acc0))
        movr = (file_data['time'][i]*(10**(-9)))**2*mov/2
        X = file_data['x'][:i]
        Y = file_data['y'][:i]
        Z = np.zeros(i)
        points = np.array([X,Y,Z]).transpose()
        pointsscipy = r.apply(points)
        points = pointsscipy.transpose()
        points[0]=points[0]+movr[0]
        points[1] = points[1] + movr[1]
        points[2] = points[2] + movr[2]
        XX.append(points[0])
        YY.append(points[1])
        ZZ.append(points[2])
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(X,Y,Z)
    XX=np.hstack(XX)
    YY=np.hstack(YY)
    ZZ=np.hstack(ZZ)
    ZZ = ZZ-np.min(ZZ)
    ZZ = ZZ/(np.max(ZZ)/limit)
    plt.show()
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(XX,YY,ZZ)
    plt.show()
    return XX,YY,ZZ

def normalize(X,Y,Z,sizelimit):
    X = np.mod(X,sizelimit).astype(int)
    Y = np.mod(Y,sizelimit).astype(int)
    Z = np.mod(Z,sizelimit).astype(int)
    points = np.vstack((X,Y,Z))
    points = np.swapaxes(points, 0, 1)
    P0 = np.zeros((sizelimit,sizelimit))
    P1 = np.zeros((sizelimit, sizelimit))
    P2 = np.zeros((sizelimit, sizelimit))
    unq, cnt = np.unique(points, axis=0, return_counts=True)
    for i,u in enumerate(unq):
        P0[u[0],u[1]]=P0[u[0],u[1]]+cnt[i]
        P1[u[1], u[2]] = P1[u[1], u[2]] + cnt[i]
        P2[u[0], u[2]] = P2[u[0], u[2]] + cnt[i]
    P2 = P2 / np.max(P2)
    P1 = P1 / np.max(P1)
    P0 = P0 / np.max(P0)
    P3 = (P0+P1+P2)/3

    P01 = np.hstack((P0,P1))
    P23 = np.hstack((P2, P3))
    out = np.vstack((P01,P23))
    # fig, axs = plt.subplots(2, 2)
    # axs[0, 0].pcolor(P0)
    # axs[0, 0].set_title('P0')
    # axs[0, 1].pcolor(P1)
    # axs[0, 1].set_title('P1')
    # axs[1, 0].pcolor(P2)
    # axs[1, 0].set_title('P2')
    # axs[1, 1].pcolor(P3)
    # axs[1, 1].set_title('P3')
    # plt.show()
    plt.pcolor(out)
    plt.show()
    return out

def normalize2(X,Y,Z,sizelimit):
    X = np.mod(X,sizelimit).astype(int)
    Y = np.mod(Y,sizelimit).astype(int)
    Z = np.mod(Z,sizelimit).astype(int)
    points = np.vstack((X,Y,Z))
    points = np.swapaxes(points, 0, 1)
    P0 = np.zeros((sizelimit,sizelimit))
    P1 = np.zeros((sizelimit, sizelimit))
    P2 = np.zeros((sizelimit, sizelimit))
    unq, cnt = np.unique(points, axis=0, return_counts=True)
    for i,u in enumerate(unq):
        P0[u[0],u[1]]=P0[u[0],u[1]]+cnt[i]
        P1[u[2], u[1]] = P1[u[2], u[1]] + cnt[i]
        P2[u[2], u[0]] = P2[u[2], u[0]] + cnt[i]
    P2 = P2 / np.max(P2)
    P1 = P1 / np.max(P1)
    P0 = P0 / np.max(P0)
    out = np.array((P0,P1,P2))
    out = np.swapaxes(out, 0, 2)
    # fig, axs = plt.subplots(2, 2)
    # axs[0, 0].pcolor(P0)
    # axs[0, 0].set_title('P0')
    # axs[0, 1].pcolor(P1)
    # axs[0, 1].set_title('P1')
    # axs[1, 0].pcolor(P2)
    # axs[1, 0].set_title('P2')
    # plt.show()
    plt.imshow(out)
    plt.show()
    return out

def generate(i):
    X, Y, Z = read_data(i)
    out = normalize2(X, Y, Z, i)
    return out

