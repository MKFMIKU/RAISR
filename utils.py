def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', 'bmp'])


def mod_crop(im, modulo):
    sz = im.shape
    h = sz[0] // modulo * modulo
    w = sz[1] // modulo * modulo
    ims = im[0:h, 0:w, ...]
    return ims