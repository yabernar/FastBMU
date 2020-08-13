import os
import numpy as np
from PIL import Image
from scipy.io import wavfile
from scipy.signal import resample


# Random data in [0, 0.8]^d
def uniform(nbr_elements, dimension):
    data = np.array([np.random.random(dimension) * 0.8 for i in range(nbr_elements)])
    return data


def open_image(image_name):
    path = os.path.join("data", image_name)
    image = Image.open(path)
    size = np.flip(image.size, 0)  # For some strange reason the data isn't ordered in the same way as the size says
    pixels = np.array(image.getdata(), 'uint8')
    pixels = pixels.reshape(size)
    return pixels, size


# Random generated 2D data in a shape defined by an image
def shape(nbr_elements, image_name):
    pixels, size = open_image(image_name)
    data = []
    for i in range(nbr_elements):
        element = np.random.random(2)
        while pixels[tuple(np.multiply(element, size).astype(int))] <= 20:  # Not ideal, but works
            element = np.random.random(2)
        data.append(element)
    return np.asarray(data)


# Randomly samples pixels colors from an image
def pixels_colors(nbr_elements, image_name):
    pixels, size = open_image(image_name)
    data = np.array(pixels, 'double') / 255
    np.random.shuffle(data)
    data.resize((nbr_elements, 3))
    return data


# Images as mosaic
def mosaic_image(image_name, pictures_dim):
    pixels, size = open_image(image_name)
    data = []

    if len(pixels.shape) == 2:  # File has RGB colours
        size[1] *= 3
        pictures_dim[1] *= 3
        color = True
    pixels = pixels.reshape(size)
    nb_pictures = np.array(np.divide(size, pictures_dim), dtype=int)
    pixels = pixels[0:nb_pictures[0] * pictures_dim[0],
             0:nb_pictures[1] * pictures_dim[1]]  # Cropping the image to make it fit
    px = np.vsplit(pixels, nb_pictures[0])
    for i in px:
        j = np.hsplit(i, nb_pictures[1])
        for picture in j:
            data.append(picture.flatten())
    data = np.array(data) / 255
    return data


# Spoken digits dataset
def spoken_digits(recording_folder, length=200):
    data = []
    path = os.path.join("data", recording_folder, "recordings")
    files = sorted([d for d in os.listdir(path) if os.path.isfile(os.path.join(path, d))], key=str.lower)
    for f in files:
        fs, element = wavfile.read(os.path.join(path, f))
        element = resample(element, length)
        element = (element / 65536) + 0.5
        element = element - min(element)
        element = element / max(element)
        data.append(element)
    data = np.asarray(data)
    return data