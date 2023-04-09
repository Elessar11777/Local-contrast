import os
import cv2
import glob
import numpy as np
from contextlib import contextmanager

@contextmanager
def open_image(item):
    print(f"item {item}")
    img = cv2.imread(item)
    try:
        yield cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)
    finally:
        cv2.destroyAllWindows()

def load_images(dir, img_ext):
    iter_items = glob.iglob(dir + img_ext)
    images = []
    exposures = []
    for item in iter_items:
        with open_image(item) as img:
            images.append(img)
            fname = os.path.basename(item)
            num = int(fname[:-4].split('_')[-1])
            den = 1000
            exposures.append(int(num / den))
    iter_tuple = zip(images, exposures)
    sorted_iter_tuple = sorted(iter_tuple, key=lambda pair: pair[1], reverse=True)
    sorted_images = [img for img, times in sorted_iter_tuple]
    sorted_exposures = sorted(exposures, reverse=True)
    sorted_log_exposures = np.log(sorted_exposures)

    return [sorted_images, sorted_log_exposures]



#print(load_images("./test_images/", "*.bmp"))