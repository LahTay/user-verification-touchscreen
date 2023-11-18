import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import json
import cupy as cp
import machine_learning.random_data_generator as random_data_generator
import os

debug = False


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


def random_from_directory(
    size, directory, amount
):  # ignores original/false and other parameters
    if directory != "":
        images = []
        files = os.listdir(directory)
        test = list(np.random.choice(len(files), amount))
        files_chosen = []
        for i in test:
            files_chosen.append(files[i])
        for path in files_chosen:
            images.append(generate(size, directory + "/" + path))
        return images
    else:
        return []


def load_from_files(size, directory):  # ignores original/false and other parameters
    if directory != "":
        images = []
        files = os.listdir(directory)
        for path in files:
            images.append(generate(size, directory + "/" + path))
        return images
    else:
        return []


def read_data(limit, filename=""):
    if filename != "":
        file_data = open_file(filename)
    else:
        file_data = random_data_generator.generate(np.random.randint(300, 900))
    XX = []
    YY = []
    ZZ = []
    # x = file_data['x']
    # y = file_data['y']
    # file_data['x'] = list(np.array(file_data['x'])-min(x))
    # file_data['y'] = list(np.array(file_data['y']) - min(y))
    rot = np.array([0, 0, 0])
    mov = np.array([0, 0, 0])
    acc0 = np.array(file_data["acc"][0])
    for i in range(1, len(file_data["x"])):
        rot = (
            rot
            + (file_data["time"][i] * (10 ** (-9))) ** 2
            * np.array(file_data["gyro"][i])
            / 2
        )
        r = R.from_euler("xyz", rot)
        mov = mov + (np.array(file_data["acc"][i]) - r.apply(acc0))
        movr = (file_data["time"][i] * (10 ** (-9))) ** 2 * mov / 2
        X = file_data["x"][:i]
        Y = file_data["y"][:i]
        Z = np.zeros(i)
        points = np.array([X, Y, Z]).transpose()
        pointsscipy = r.apply(points)
        points = pointsscipy.transpose()
        points[0] = points[0] + movr[0]
        points[1] = points[1] + movr[1]
        points[2] = points[2] + movr[2]
        XX.append(points[0])
        YY.append(points[1])
        ZZ.append(points[2])
    if debug:
        plt.plot(X, Y)
        plt.show()
    XX = np.hstack(XX)
    XX = XX - np.min(XX)
    # XX = XX / (np.max(XX) / (limit-1))

    YY = np.hstack(YY)
    YY = YY - np.min(YY)
    # YY = YY / (np.max(YY) / (limit-1))

    ZZ = np.hstack(ZZ)
    ZZ = ZZ - np.min(ZZ)
    ZZ = ZZ / (np.max(ZZ) / (limit - 1))
    if debug:
        ax = plt.figure().add_subplot(projection="3d")
        ax.plot(XX, YY, ZZ)
        plt.show()

    data = XX, YY, ZZ, file_data["type"], file_data["position"], filename

    return data


def normalize(X, Y, Z, sizelimit):
    X = np.mod(X, sizelimit).astype(int)
    Y = np.mod(Y, sizelimit).astype(int)
    Z = np.mod(Z, sizelimit).astype(int)
    points = np.vstack((X, Y, Z))
    points = np.swapaxes(points, 0, 1)
    P0 = np.zeros((sizelimit, sizelimit))
    P1 = np.zeros((sizelimit, sizelimit))
    P2 = np.zeros((sizelimit, sizelimit))
    unq, cnt = np.unique(points, axis=0, return_counts=True)
    for i, u in enumerate(unq):
        P0[u[0], u[1]] = P0[u[0], u[1]] + cnt[i]
        P1[u[1], u[2]] = P1[u[1], u[2]] + cnt[i]
        P2[u[0], u[2]] = P2[u[0], u[2]] + cnt[i]
    P2 = P2 / np.max(P2)
    P1 = P1 / np.max(P1)
    P0 = P0 / np.max(P0)
    P3 = (P0 + P1 + P2) / 3

    P01 = np.hstack((P0, P1))
    P23 = np.hstack((P2, P3))
    out = np.vstack((P01, P23))
    if debug:
        plt.pcolor(out)
        plt.show()
    return out


def normalize2(X, Y, Z, sizelimit):
    X = np.mod(X, sizelimit).astype(int)
    Y = np.mod(Y, sizelimit).astype(int)
    Z = np.mod(Z, sizelimit).astype(int)
    points = np.vstack((X, Y, Z))
    points = np.swapaxes(points, 0, 1)
    P0 = np.zeros((sizelimit, sizelimit))
    P1 = np.zeros((sizelimit, sizelimit))
    P2 = np.zeros((sizelimit, sizelimit))
    unq, cnt = np.unique(points, axis=0, return_counts=True)
    for i, u in enumerate(unq):
        P0[u[0], u[1]] = P0[u[0], u[1]] + cnt[i]
        P1[u[2], u[1]] = P1[u[2], u[1]] + cnt[i]
        P2[u[2], u[0]] = P2[u[2], u[0]] + cnt[i]
    P2 = P2 / np.max(P2)
    P1 = P1 / np.max(P1)
    P0 = P0 / np.max(P0)
    out = np.array((P0, P1, P2))
    out = np.swapaxes(out, 0, 2)
    if debug:
        plt.imshow(out)
        plt.show()
    return out


def normalizehybrid(X, Y, Z, sizelimit):
    X = np.mod(X, sizelimit).astype(int)
    Y = np.mod(Y, sizelimit).astype(int)
    Z = np.mod(Z, sizelimit).astype(int)
    points = np.vstack((X, Y, Z))
    points = np.swapaxes(points, 0, 1)
    P0 = np.zeros((sizelimit, sizelimit))
    P1 = np.zeros((sizelimit, sizelimit))
    P2 = np.zeros((sizelimit, sizelimit))
    unq, cnt = np.unique(points, axis=0, return_counts=True)
    P0[unq[:, 0], unq[:, 1]] = cnt
    P1[unq[:, 2], unq[:, 1]] = cnt
    P2[unq[:, 2], unq[:, 0]] = cnt
    P2 = P2 / np.max(P2)
    P1 = P1 / np.max(P1)
    P0 = P0 / np.max(P0)
    out = np.array((P0, P1, P2))
    out = np.swapaxes(out, 0, 2)
    # Check for NaN and Infinity values
    nan_mask = np.isnan(out)
    inf_mask = np.isinf(out)

    # Replace NaN values with 0 and Infinity values with 1
    out[nan_mask] = 0
    out[inf_mask] = 1
    if debug:
        plt.imshow(out)
        plt.show()
    return out


def normalizegpu(X, Y, Z, sizelimit, stream):
    X = cp.mod(X, sizelimit).astype(int)
    Y = cp.mod(Y, sizelimit).astype(int)
    Z = cp.mod(Z, sizelimit).astype(int)
    points = cp.vstack((X, Y, Z))
    points = cp.swapaxes(points, 0, 1)
    P0 = cp.zeros((sizelimit, sizelimit))
    P1 = cp.zeros((sizelimit, sizelimit))
    P2 = cp.zeros((sizelimit, sizelimit))
    points_np = points.get()
    unq, cnt = np.unique(points_np, axis=0, return_counts=True)
    unq = cp.array(unq)
    cnt = cp.array(cnt)
    P0[unq[:, 0], unq[:, 1]] = cnt
    P1[unq[:, 2], unq[:, 1]] = cnt
    P2[unq[:, 2], unq[:, 0]] = cnt
    P2 = P2 / cp.max(P2)
    P1 = P1 / cp.max(P1)
    P0 = P0 / cp.max(P0)
    out = cp.array((P0, P1, P2))
    out = cp.swapaxes(out, 0, 2)
    # Check for NaN and Infinity values
    nan_mask = cp.isnan(out)
    inf_mask = cp.isinf(out)

    # Replace NaN values with 0 and Infinity values with 1
    out[nan_mask] = 0
    out[inf_mask] = 1
    return out


def generate(i, filename=""):
    X, Y, Z, _, _, _ = read_data(i, filename)
    out = normalizehybrid(X, Y, Z, i)
    return out


def normalizegpt(X, Y, Z, sizelimit):
    # Ensure X, Y, Z are numpy arrays
    X, Y, Z = (
        np.mod(X, sizelimit).astype(int),
        np.mod(Y, sizelimit).astype(int),
        np.mod(Z, sizelimit).astype(int),
    )

    # Stack X, Y, Z into a single array
    points = np.column_stack((X, Y, Z))

    # Count unique points and create P0, P1, P2 arrays
    unq, cnt = np.unique(points, axis=0, return_counts=True)
    P0, P1, P2 = (
        np.zeros((sizelimit, sizelimit)),
        np.zeros((sizelimit, sizelimit)),
        np.zeros((sizelimit, sizelimit)),
    )

    P0[unq[:, 0], unq[:, 1]] = cnt
    P1[unq[:, 2], unq[:, 1]] = cnt
    P2[unq[:, 2], unq[:, 0]] = cnt

    # Normalize P0, P1, P2
    P0 /= np.max(P0)
    P1 /= np.max(P1)
    P2 /= np.max(P2)

    # Stack P0, P1, P2 and swap axes
    out = np.dstack((P0, P1, P2))

    if debug:
        import matplotlib.pyplot as plt

        plt.imshow(out)
        plt.show()

    return out


def apply_function_concurrently_on_gpu(X_list, Y_list, Z_list, image_size):
    # Create a CUDA stream for each set of inputs
    streams = [cp.cuda.Stream() for _ in X_list]
    # Apply the function concurrently on the GPU using streams
    results = []
    for pinned_X_, pinned_Y_, pinned_Z_, stream in zip(X_list, Y_list, Z_list, streams):
        with stream:
            result = normalizegpu(pinned_X_, pinned_Y_, pinned_Z_, image_size, stream)
            results.append(result)

    # Synchronize the streams to ensure all operations are completed
    for stream in streams:
        stream.synchronize()

    return results


def generategpu(image_size, files):
    # List comprehension to read data from files and process each element
    arrays = [read_data(image_size, filepath) for filepath in files]
    # Splitting the array into 'position' (XX, YY, ZZ) and 'rest' (type, position, filename)
    rest = [(type, pos, fname) for _, _, _, type, pos, fname in arrays]
    arrays = [(XX, YY, ZZ) for XX, YY, ZZ, _, _, _ in arrays]

    print("data loaded")
    X_cp = [cp.array(X) for X, Y, Z in arrays]
    Y_cp = [cp.array(Y) for X, Y, Z in arrays]
    Z_cp = [cp.array(Z) for X, Y, Z in arrays]
    results = apply_function_concurrently_on_gpu(X_cp, Y_cp, Z_cp, image_size)
    # results = [normalizegpu(cp.array(X), cp.array(Y), cp.array(Z),i) for X, Y, Z in arrays]

    results_cpu = [result.get() for result in results]
    return results_cpu, rest


# random_from_directory(128,"./przebiegi",5)
