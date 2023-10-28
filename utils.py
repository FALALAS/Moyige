import numpy as np
from numpy.linalg import norm
# from skimage.measure import compare_psnr, compare_ssim, compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse
import numpy as np
import logging


def initialize_logger(file_dir):
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger


def mpsnr(x_true, x_pred):
    """
    :param x_true: 高光谱图像：格式：(H, W, C)#(C,H,W)
    :param x_pred: 高光谱图像：格式：(H, W, C)
    :return: 计算原始高光谱数据与重构高光谱数据的均方误差
    """
    n_bands = x_true.shape[0]
    p = [compare_psnr(x_true[k, :, :], x_pred[k, :, :], data_range=1) for k in range(n_bands)]
    return np.mean(p)


def sam(x_true, x_pred):
    """
    :param x_true: 高光谱图像：格式：(H, W, C)
    :param x_pred: 高光谱图像：格式：(H, W, C)
    :return: 计算原始高光谱数据与重构高光谱数据的光谱角相似度
    """
    assert x_true.ndim == 3 and x_true.shape == x_pred.shape
    sam_rad = np.zeros(x_pred.shape[:2])
    for x in range(x_true.shape[0]):
        for y in range(x_true.shape[1]):
            tmp_pred = x_pred[x, y].ravel()
            tmp_true = x_true[x, y].ravel()
            sam_rad[x, y] = np.arccos(np.dot(tmp_pred, tmp_true) / (norm(tmp_pred) * norm(tmp_true)))
    sam_deg = sam_rad.mean() * 180 / np.pi
    return sam_deg


def mssim(x_true, x_pred):
    """
        :param x_true: 高光谱图像：格式：(H, W, C)
        :param x_pred: 高光谱图像：格式：(H, W, C)
        :return: 计算原始高光谱数据与重构高光谱数据的结构相似度
    """
    # multichannel ： bool，可选
    # 如果为True，则将数组的最后一个维度视为通道。对每个通道独立地进行相似性计算，然后进行平均
    # SSIM = compare_ssim(X=x_true, Y=x_pred, multichannel=True, data_range=1)
    n_bands = x_true.shape[0]
    p = [compare_ssim(x_true[k, :, :], x_pred[k, :, :], data_range=1) for k in range(n_bands)]
    # print(len(p))
    return np.mean(p)