import json
import matplotlib.pyplot as plt


def open_file(file_name: str):
    data = []
    with open(file_name) as f:
        data = json.load(f)
    return data


if __name__ == "__main__":

    file_data = open_file("data.json")

    print(file_data.keys())

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

    plt.xlim([0,file_data["size"]])
    plt.ylim([0,file_data["size"]])
    plt.plot(file_data["x"],file_data["y"])
    plt.show()
