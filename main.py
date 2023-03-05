"""import localtonemap.tonemap as tm
import cv2

hdr = cv2.imread('1.hdr', cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)

tonemapped_img = tm.tonemap(hdr, l_remap=(0, 1), saturation=1., gamma_=1.5, numtiles=(16, 16))

tonemapped_img = cv2.cvtColor(tonemapped_img, cv2.COLOR_RGB2BGR)

cv2.imwrite('tonemapped_image.bmp', tonemapped_img)
"""
import sys
from load_images import load_images
import matplotlib.pyplot as mp_plt
import numpy as np
from hdr_debevec import hdr_debevec
from irradiance import compute_irradiance
from tonemap import reinhard_tonemap, plot_and_save, local_tonemap
import cv2


def run_hdr(image_name, image_ext, root_dir, COMPUTE_CRF, kwargs):
    if (len(kwargs) > 0):
        lambda_ = kwargs['lambda_']
        num_px = kwargs['num_px']
        gamma = kwargs['gamma']
        alpha = kwargs['alpha']
        gamma_local = kwargs['gamma_local']
        saturation_local = kwargs['saturation_local']

    [images, B] = load_images(root_dir, image_ext)

    plot_idx = np.random.choice(len(images), (2,), replace=False)
    mp_plt.figure(figsize=(16, 16))
    mp_plt.subplot(1, 2, 1)
    mp_plt.imshow(images[plot_idx[0]])
    mp_plt.title("Exposure time: {} secs".format(np.exp(B[plot_idx[0]])))
    mp_plt.subplot(1, 2, 2)
    mp_plt.imshow(images[plot_idx[1]])
    mp_plt.title("Exposure time: {} secs".format(np.exp(B[plot_idx[1]])))

    if (COMPUTE_CRF):
        [crf_channel, log_irrad_channel, w] = hdr_debevec(images, B, lambda_=lambda_, num_px=num_px)

        #print(np.array(crf_channel).shape)
        #print(np.array(log_irrad_channel).shape)
        #print(np.array(w).shape)
        #print(w)

        #p.save(root_dir + "crf.npy", np.array([crf_channel, log_irrad_channel, w]))
    else:
        hdr_loc = kwargs['hdr_loc']
        [crf_channel, log_irrad_channel, w] = np.load(hdr_loc)
    irradiance_map = compute_irradiance(crf_channel, w, images, B)
    tonemapped_img = reinhard_tonemap(irradiance_map, gamma=gamma, alpha=alpha)
    #plot_and_save(tonemapped_img, root_dir, "Globally Tonemapped Image")
    local_tonemap(irradiance_map, root_dir + image_name, saturation=saturation_local, gamma=gamma_local)
    return [tonemapped_img, irradiance_map]


if __name__ == "__main__":
    ROOT_DIR = "./test_images/"
    IMAGE_DIR = ""
    IMAGE_EXT = "*.bmp"
    COMPUTE_CRF = True

    kwargs = {'lambda_': 50, 'num_px': 150, 'gamma': 1 / 2.2, 'alpha': 0.35, 'hdr_loc': ROOT_DIR + "crf.npy",
          'gamma_local': 1.0, 'saturation_local': 2.5}
    hdr_image, irmap = run_hdr("0.0", IMAGE_EXT, ROOT_DIR, COMPUTE_CRF, kwargs)
