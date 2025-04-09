import cv2
import os
import numpy as np
def homomorphic_filter(src,d0=70,c=0.1,rh=3,h=2.2,r1=0.8,l=0.7):
    gray = src.copy()

    if len(src.shape) > 2:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)
    rows, cols = gray.shape
    gray_fft = np.fft.fft2(gray)
    gray_fftshift = np.fft.fftshift(gray_fft)
    dst_fftshift = np.zeros_like(gray_fftshift)
    M, N = np.meshgrid(np.arange(-cols // 2, cols // 2), np.arange(-rows//2, rows//2))


    D = np.sqrt(M ** 2 + N ** 2)
    Z = (rh - r1) * (1 - np.exp(-c * (D ** 2 / d0 ** 2))) + r1#
    dst_fftshift = Z * gray_fftshift
    dst_fftshift = (h - l) * dst_fftshift + l
    dst_ifftshift = np.fft.ifftshift(dst_fftshift)
    dst_ifft = np.fft.ifft2(dst_ifftshift)
    dst = np.real(dst_ifft)
    dst = np.uint8(np.clip(dst, 0, 255))
    return dst


def adaptive_histogram_equalization(image, clip_limit=2.0, tile_size=(8, 8)):
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    equalized_image = clahe.apply(gray_image)

    return equalized_image
input_folder = r'D:\zzm\mine(3.15)\data\BUSI\val\images'
output_folder = r'D:\zzm\mine(3.15)\data\BUSI\val\enhanceimages'
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):

    image_path = os.path.join(input_folder, filename)
    image = cv2.imread(image_path)


    result_image = homomorphic_filter(adaptive_histogram_equalization(homomorphic_filter(image)))


    output_path = os.path.join(output_folder, filename)


    cv2.imwrite(output_path, result_image)

print("COMPLETED")