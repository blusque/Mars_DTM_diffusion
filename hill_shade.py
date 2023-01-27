import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import time

pi = np.pi
half_round = 180.


class HillShadeParam:

    def __init__(self, z_factor, altitude=45.0, azimuth=315.0):
        azimuth = azimuth if azimuth < 360 else azimuth - 360
        self.z_factor = z_factor
        self.zenith_rad = (90 - altitude) * pi / half_round
        self.azimuth_rad = (360 - azimuth + 90) * pi / half_round


def hill_shade_alg(dx, dy, param: HillShadeParam):
    slope_rad = np.arctan(param.z_factor * np.sqrt(dx ** 2 + dy ** 2))
    aspect_rad = np.arctan2(dy, -dx)
    print(slope_rad.max(), slope_rad.min(), slope_rad.mean(), slope_rad.var())

    row = aspect_rad.shape[0]
    col = aspect_rad.shape[1]

    aspect_rad_dy = np.where(dy > 0, pi / 2, 2 * pi - pi / 2)
    aspect_rad = np.where(aspect_rad < 0, 2 * pi + aspect_rad, aspect_rad)
    aspect_rad = np.where(aspect_rad == 0, aspect_rad_dy, aspect_rad)
    # for i in range(row):
    #     for j in range(col):
    #         if aspect_rad[i, j] < 0:
    #             aspect_rad[i, j] = 2 * pi + aspect_rad[i, j]
    #         elif aspect_rad[i, j] == 0:
    #             if dy[i, j] > 0:
    #                 aspect_rad[i, j] = pi / 2
    #             elif dy[i, j] < 0:
    #                 aspect_rad[i, j] = 2 * pi - pi / 2

    azimuth_rad_matrix = np.ones_like(aspect_rad) * param.azimuth_rad
    # print(np.cos(param.zenith_rad) * np.cos(slope_rad))
    # print(np.sin(param.zenith_rad) * np.sin(slope_rad))
    shade = (np.cos(param.zenith_rad) * np.cos(slope_rad)) \
            + (np.sin(param.zenith_rad) * np.sin(slope_rad) * np.cos(azimuth_rad_matrix - aspect_rad))
    print(shade.mean(), shade.std(), shade.min(), shade.max())
    
    return shade


def hill_shade(input_data, z_factor, altitude=45.0, azimuth=315.0):
    shade_param = HillShadeParam(z_factor, altitude, azimuth)

    dx, dy = np.gradient(input_data)

    relief = hill_shade_alg(dx, dy, shade_param)

    return relief


def main():
    filename = 'G:/dem_image/dtm/001462_2015-DTM_11.tif'
    dtm = io.imread(filename, plugin='pil')
    dtm = np.array(dtm, dtype=np.float32)
    dtm /= 255.
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(dtm, cmap='gray')
    # io.imshow(dtm)
    # plt.show()
    start = time.time()
    relief = hill_shade(dtm, z_factor=10.0)
    end = time.time()
    print("relief cost: ", end - start)
    ax[1].imshow(relief, cmap='gray')
    plt.show()


if __name__ == "__main__":
    main()

