import numpy as np
import cv2
import matplotlib.pyplot as mp_plt
from localtonemap.tonemap import tonemap


def plot_and_save(img, img_name, img_title):
    params = [cv2.IMWRITE_JPEG_QUALITY, 80]
    mp_plt.figure(figsize=(8, 8))
    mp_plt.imshow(img)
    mp_plt.title(img_title)
    image_save = img * 255
    cv2.imwrite(img_name + ".jpg", image_save[:, :, [2, 1, 0]], params)


def reinhard_tonemap(irradiance_map, gamma=1 / 2.2, alpha=0.25):
    C = 3  # num of channels

    # tone mapping parameters
    E_map = np.empty_like(irradiance_map)

    # normalize irradiance map
    for ch in range(C):
        map_channel = irradiance_map[:, :, ch]
        E_min = map_channel.min()
        E_max = map_channel.max()
        E_map[:, :, ch] = (map_channel - E_min) / (E_max - E_min)

    # gamma correction
    E_map = E_map ** gamma

    # convert to grayscale and apply Reinhart Tone Mapping
    L = cv2.cvtColor(E_map.astype('float32'), cv2.COLOR_RGB2GRAY)
    L_avg = np.exp(np.mean(np.log(L)))  # average normalized grayscale irradiance
    T = alpha / L_avg * L
    L_tone = T * (1 + (T / (T.max()) ** 2)) / (1 + T)
    M = L_tone / L

    # apply scaling to each channel
    tonemapped_img = np.empty_like(E_map)
    for ch in range(C):
        tonemapped_img[:, :, ch] = E_map[:, :, ch] * M

    return np.clip(tonemapped_img, 0.0, 1.0)

def local_tonemap(irradiance_map, img_name, saturation=1., gamma=1 / 2.2):
    """# tonemap using Opencv's Durand Tonemap algorithm
    tonemap_obj = cv2.createTonemapDurand(gamma=4, sigma_color = 5.0)
    hdr_local = tonemap_obj.process(irmap.astype('float32'))
    mp_plt.figure(figsize=(16,16))
    mp_plt.imshow(hdr_local)"""

    # compute tonemapped image
    local_tonemap = tonemap(irradiance_map, saturation=saturation, gamma_=gamma, numtiles=(36, 36))

    cv2.imwrite("1.png", local_tonemap)
    img = local_tonemap[:, :, [2, 1, 0]]

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    #img = unsharp_mask(img)
    #img = cv2.detailEnhance(img, sigma_s=20, sigma_r=0.1)
    cv2.imwrite(img_name + ".jpg", img, encode_param)

    return local_tonemap