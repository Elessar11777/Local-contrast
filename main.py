"""import localtonemap.tonemap as tm
import cv2

hdr = cv2.imread('1.hdr', cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)

tonemapped_img = tm.tonemap(hdr, l_remap=(0, 1), saturation=1., gamma_=1.5, numtiles=(16, 16))

tonemapped_img = cv2.cvtColor(tonemapped_img, cv2.COLOR_RGB2BGR)

cv2.imwrite('tonemapped_image.bmp', tonemapped_img)
"""
import multiprocessing as mp
from load_images import load_images
import numpy as np
from hdr_debevec import hdr_debevec
from irradiance import compute_irradiance
from tonemap import process_local_tonemap
import os
from memory_profiler import profile
import gc


def run_hdr(image_name, image_ext, root_dir, COMPUTE_CRF, kwargs):
    if (len(kwargs) > 0):
        lambda_ = kwargs['lambda_']
        num_px = kwargs['num_px']
        gamma = kwargs['gamma']
        alpha = kwargs['alpha']
        gamma_local = kwargs['gamma_local']
        saturation_local = kwargs['saturation_local']

    [images, B] = load_images(root_dir, image_ext)
    if not images:
        print(f"root_dir {root_dir} is empty")
        return

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
    del images
    #plot_and_save(tonemapped_img, root_dir, "Globally Tonemapped Image")
    process_local_tonemap(irradiance_map, image_name, saturation=saturation_local, gamma=gamma_local)
    return [None, irradiance_map]



# def process_folder(foldername, image_ext, root_dir, compute_crf, kwargs):
#     folder_path = os.path.join(root_dir, foldername)
#     if os.path.isdir(folder_path):
#         return run_hdr(foldername, image_ext, folder_path + "/", compute_crf, kwargs)

def process_folder(foldername, image_ext, root_dir, compute_crf, kwargs):
    folder_path = os.path.join(root_dir, foldername)
    if os.path.isdir(folder_path):
        # Check if a result image is present
        result_image_path = os.path.join("./result/", f"{foldername}.png")
        print(result_image_path)
        if not os.path.exists(result_image_path):
            return run_hdr(foldername, image_ext, folder_path + "/", compute_crf, kwargs)


if __name__ == "__main__":
    ROOT_DIR = "./test_images/"
    IMAGE_DIR = ""
    IMAGE_EXT = "*.bmp"
    COMPUTE_CRF = True

    kwargs = {'lambda_': 50, 'num_px': 150, 'gamma': 1 / 2.2, 'alpha': 0.35, 'hdr_loc': ROOT_DIR + "crf.npy",
          'gamma_local': 1.0, 'saturation_local': 2.5}

    pool = mp.Pool(6, maxtasksperchild=1)
    results = []
    for foldername in os.listdir(ROOT_DIR):
        results.append(pool.apply_async(process_folder, args=(foldername, IMAGE_EXT, ROOT_DIR, COMPUTE_CRF, kwargs)))
    pool.close()
    pool.join()
    for res in results:
        res.get()
