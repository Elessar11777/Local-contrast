import cv2
from localtonemap import local_tonemap


def process_local_tonemap(irradiance_map, img_name, saturation=1., gamma=1 / 2.2):
    """# tonemap using Opencv's Durand Tonemap algorithm
    tonemap_obj = cv2.createTonemapDurand(gamma=4, sigma_color = 5.0)
    hdr_local = tonemap_obj.process(irmap.astype('float32'))
    mp_plt.figure(figsize=(16,16))
    mp_plt.imshow(hdr_local)"""

    # compute tonemapped image
    local_tonemap_1 = local_tonemap(irradiance_map, saturation=saturation, gamma_=gamma, numtiles=(24, 24))

    cv2.imwrite("1.png", local_tonemap_1)
    img = local_tonemap_1[:, :, [2, 1, 0]]

    path = "./result/" + img_name + ".png"
    print(path)
    cv2.imwrite(path, img)

    return local_tonemap_1