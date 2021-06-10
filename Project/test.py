import numpy as np
import cv2
import math
import hide
from pyzbar.pyzbar import decode
def getpsnr(img1,img2):
    orignal=np.float64(img1)
    carrier=np.float64(img2)
    mse = np.mean((orignal - carrier) ** 2)#求均值
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
def getNc(img1,img2):#越高越好
    if img1.shape!=img2.shape:
        return 0
    length,width=img1.shape
    numerator=0
    denominator=0
    for i in range(0,length):
        for j in range(0,width):
            numerator=img1[i][j]*img2[i][j]+numerator
            denominator=img1[i][j]*img1[i][j]+denominator
    nc=numerator/denominator
    return nc
def getCos(str1,str2):
    numerator=0
    denominator=0
    denominator1=0
    denominator2=0
    length=len(str1)
    for i in range(0,length):
        numerator=numerator+int(str1[i])*int(str2[i])
        denominator1=denominator1+int(str1[i])**2
        denominator2=denominator2+int(str2[i])**2
    denominator=denominator1**0.5*denominator2**0.5
    cos=numerator/denominator
    return cos
def GetQRCode(img):
    print("二维码内容为 ")
    barcodes = decode(img)
    for barcode in barcodes:
        url = barcode.data.decode("utf-8")
        print(url)
    return barcodes
if __name__ == '__main__':
    img1 = cv2.imread("D:/carrier.png", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("D:/carrywatermark.png", cv2.IMREAD_GRAYSCALE)
    print("psnr is:", getpsnr(img1, img2))
    print("Nc is:", getNc(img1, img2))
    str1 = hide.watermarkTostring(img1)
    str2 = hide.watermarkTostring(img2)
    print("Cos is:", getCos(str1, str2))