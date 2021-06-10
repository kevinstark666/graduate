import cv2
import numpy as np
import matplotlib.pyplot as plt#图片展示
import base64 #图片转string
import time


def show(str,img):
    plt.imshow(img,'gray')
    plt.title(str)
    plt.show()

def watermarkTostring(img):#水印图像转为标识字符串
    str=""
    height, width = img.shape
    print("The number of the img is:",height*width)
    print(img)
    for i in range(0,height):
        for x in range(0,width):
            if img[i][x]==255:#白色
                str=str+"0"
            else:#黑色
                str=str+"1"
    return str

def listToimg(list):
    img=np.array(list)
    return img
def dct_encode(img1,img,coefficient):  #返回处理好的dct
    starttime=time.time()
    height, width = img.shape
    print("Number of the carrirer can contain:",height*width/2)
    str=watermarkTostring(img1)
    if len(str)>height*width/2:#如果信息数量过多返回错误 只能存储在后二分之一
        print("Too many information to hide!")

    img1=img.astype('float')
    img_dct = cv2.dct(img1)
    h=0
    w=width-1                #行数向下递增 列数向右递增
    last_h=1                 #保存上一次跳转的列数
    print("Embeding...")
    print("隐藏",len(str),"位")
    black=0
    white=0
    for i in range(0,len(str)):#所有水印字符嵌入  范围前闭后开
        if h==height:
            h=last_h
            last_h=last_h+1
            w=width-1
        if str[i]=="0":#白色变小
            #print("白色颜色记录位置",h,w)
            if img_dct[h][w]<0:
                img_dct[h][w]=img_dct[h][w]*(1+coefficient)
            else:
                img_dct[h][w]=img_dct[h][w]*(1-coefficient)
            h=h+1
            w=w-1
            white=white+1
        else: #黑色变大
            if img_dct[h][w]<0:
                img_dct[h][w]=img_dct[h][w]*(1-coefficient)
            else:
                img_dct[h][w]=img_dct[h][w]*(1+coefficient)
            #print("黑色颜色记录位置", h, w)
            h=h+1
            w=w-1
            black=black+1
    print("stop caculuation location ", h, w)
    print("number of white is ",white)
    print("number of black is ",black)
    print("Embeding complete.")   #嵌入完成
    encode_img=cv2.idct(img_dct)
    cv2.imwrite('D:/carrywatermark.png',encode_img)
    endtime=time.time()
    costtime=endtime-starttime
    print("encode cost",costtime)
    show("carrier",img)
    show("encode",encode_img)
    return  encode_img

def dct_decode(watermark,img,orginal_img):  #现在图像减原图像 负数为白色 正数为黑色
    starttime=time.time()
    height,width=watermark.shape
    str_len=height*width
    print("需解析",str_len,"位")
    print("组成高为",height,"宽为",width,"的图像")
    carrier_height, carrier_width = img.shape
    carrier_h=0
    carrier_w=carrier_width-1
    last_carrierh=1
    height2,width2=orginal_img.shape
    '''if height1<height2:  #以最小的计算
        height=height1
    else:
        height=height2
    if width1<width2:
        width=width1
    else:
        width=width2'''
    black=0
    white=0
    img1 = img.astype('float')
    img1_dct = cv2.dct(img1)
    img2=orginal_img.astype('float')
    img2_dct=cv2.dct(img2)
    watermark_list=[[]for i in range(0,height)]    #生成空二维数组
    h=0
    w=width-1
    last_h=1
    list_height=0
    for i in range(0,str_len):#所有水印字符嵌入  范围前闭后开
        if carrier_h==carrier_height:#到底了
            carrier_h=last_carrierh
            carrier_w=carrier_width-1
            last_carrierh=last_carrierh+1

        flag=img1_dct[carrier_h][carrier_w]-img2_dct[carrier_h][carrier_w]
        carrier_w=carrier_w-1
        carrier_h=carrier_h+1
        if flag>0:#黑色
            black=black+1
            if len(watermark_list[list_height])==width:
                list_height=list_height+1
            watermark_list[list_height].append(0)
            h=h+1
            w=w-1
        else: #白色
            white=white+1
            if len(watermark_list[list_height]) == width:
                list_height = list_height + 1
            watermark_list[list_height].append(255)
            h=h+1
            w=w-1
    print("stop caculuation location ", carrier_h, carrier_w)
    print(len(watermark_list[0]))
    print("number of white is ", white)
    print("number of black is ", black)
    watermark=np.array(watermark_list)              #list 类型转为numpy.ndarray
    #print(watermark)
    print("Extract complete！")   #提取完成
    cv2.imwrite('D:/getwatermark.png', watermark)
    endtime=time.time()
    costtime=endtime-starttime
    print("decode cost",costtime)
    show("get_watermark",watermark)
    return watermark

def str_image(str,path):#字符串转图像
    fh = open(path, "wb")
    fh.write(base64.b64decode(str))
    fh.close()
def img_to_str(path):
    with open(path, "rb") as imageFile:
        str = base64.b64encode(imageFile.read())
        print("input",str)
        return str

if __name__ == '__main__':
    path = "D:/carrierwithpic.png"
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    water = "D:/watermark.png"
    watermark = cv2.imread(water, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread("D:/carrywatermark.png", cv2.IMREAD_GRAYSCALE)
    a = dct_encode(watermark, img, 0.3)
    dct_decode(watermark, a, img)
    dct_decode(watermark, img1, img)
