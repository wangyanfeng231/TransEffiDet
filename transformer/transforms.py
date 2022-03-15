#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#   map测试请看get_dr_txt.py、get_gt_txt.py
#   和get_map.py
#--------------------------------------------#

from skimage import exposure, util
from PIL import Image, ImageEnhance
import numpy as np

def aj_contrast(image):  # 调整对比度 两种方式 gamma/log
    image = np.array(image)
    random_factor = np.random.randint(3, 31) / 10.
    gam = exposure.adjust_gamma(image, random_factor)
    # log = exposure.adjust_log(image, 0.3)
    gam = Image.fromarray(gam)
    return gam


def randomGaussian(image):  # 高斯噪声
    im = np.array(image)
    im = im/255
    random_factor = np.random.random()/10
    im = im + util.random_noise(im, mode='gaussian', var=random_factor)
    im = np.clip(im, 0.0, 1.0)
    gaussian_image = Image.fromarray(np.uint8(im*255))
    return gaussian_image


def randomColor(image):  # 随机颜色

    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    redu_image = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度

    return redu_image

def process (image):
    if np.random.random() >0.2:
        if np.random.choice(('aj_contrast','randomGaussian','randomColor'))=='aj_contrast':
            return aj_contrast(image)
        elif np.random.choice(('aj_contrast','randomGaussian','randomColor'))=='randomGaussian':
            return randomGaussian(image)
        elif np.random.choice(('aj_contrast','randomGaussian','randomColor'))=='randomColor':
            return aj_contrast(image)
    return image


if __name__ == '__main__':
    image = Image.open('/home/jy/programepy/other/efficientdet-pytorch-master/img/1.jpeg')
    img = process(image)
    img.show()
    a = 1