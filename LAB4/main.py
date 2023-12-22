import cv2    
import numpy as np
import random


# 直方图均衡化
def hist_equal(img, z_max=255):
	H, W = img.shape

	S = H * W  * 1.
	out = img.copy()
	sum_h = 0.
	for i in range(1, 255):
		ind = np.where(img == i)
		sum_h += len(img[ind])
		z_prime = z_max / S * sum_h
		out[ind] = z_prime
	out = out.astype(np.uint8)
	return out

# 二值化
def binary(img, thresh=128):
    out = img.copy()
    out[out>thresh] = 255
    out[out<=thresh] = 0
    return out
    
def threshold_By_OTSU(img):
    image=cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   ##要二值化图像，必须先将图像转为灰度图
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    print("threshold value %s" % ret)  #打印阈值，超过阈值显示为白色，低于该阈值显示为黑色
    cv2.imshow("threshold", binary) #显示二值化图像
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

# 椒盐噪声
def PepperandSalt(src,percetage=0.2):
    NoiseImg=src
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX=random.randint(0,src.shape[0]-1)
        randY=random.randint(0,src.shape[1]-1)
        if random.random()<=0.5:
            NoiseImg[randX,randY]=0
        else:
            NoiseImg[randX,randY]=255          
    return NoiseImg 

# 中值滤波 
def medianfliter(a, windowsize=5):
    output = a.copy()
    if windowsize == 3 :
        output1 = np.zeros(a.shape, np.uint8)
        for i in range(1, output.shape[0]-1):  # 求齐周围9个方格与模版进行冒泡排序
            for j in range(1, output.shape[1]-1):
                value1 = [output[i-1][j-1], output[i-1][j], output[i-1][j+1], output[i][j-1], output[i][j], output[i][j+1], output[i+1][j-1], output[i+1][j], +output[i+1][j+1]]
                value1.sort()  # 对这九个数进行排序                
                value = value1[4]    # 中值为排序后中间这个数的正中间
                output1[i-1][j-1] = value
    elif windowsize == 5:
        output1 = np.zeros(a.shape, np.uint8)
        for i in range(2, output.shape[0]-2):       # 求齐周围25个方格与模版进行卷积
            for j in range(2, output.shape[1]-2):
                value1 = [output[i-2][j-2],output[i-2][j-1],output[i-2][j],output[i-2][j+1],output[i-2][j+2],output[i-1][j-2],output[i-1][j-1],output[i-1][j],output[i-1][j+1],\
                            output[i-1][j+2],output[i][j-2],output[i][j-1],output[i][j],output[i][j+1],output[i][j+2],output[i+1][j-2],output[i+1][j-1],output[i+1][j],output[i+1][j+1],\
                            output[i+1][j+2],output[i+2][j-2],output[i+2][j-1],output[i+2][j],output[i+2][j+1],output[i+2][j+2]]
                value1.sort()   # 进行排序
                value = value1[12]    # 中值为排序后中间这个数的正中间
                output1[i-2][j-2] = value   # 将计算结果填入原本位置
    else :
        print('模版大小输入错误，请输入3或5，分别代表3*3或5*5模版！')
    return output1

# Canny边缘检测
def Canny_demo(image):
    out = image.copy()
    gray = cv2.GaussianBlur(out, (5, 5), 0)
    #gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    gradx = cv2.Sobel(gray, cv2.CV_16SC1, 1, 0)
    grady = cv2.Sobel(gray, cv2.CV_16SC1, 0, 1)
    edge_output = cv2.Canny(gradx, grady, 50, 120)
    return edge_output

if __name__ == '__main__':  
    img = cv2.imread("lena.jpg")
    img2 =cv2.imread("img_gray.jpg")
    b, g, r = cv2.split(img)
    histImgB = hist_equal(b)
    zeros = np.zeros(img.shape[:2],dtype="uint8")
    #cv2.imshow("DISPLAY BLUE COMPONENT",cv2.merge([histImgB,zeros,zeros]))
    cv2.imwrite("img_b.jpg", cv2.merge([histImgB,zeros,zeros]))
    histImgG = hist_equal(g)
    cv2.imwrite("img_g.jpg", cv2.merge([zeros,histImgG,zeros]))
    #cv2.imshow("DISPLAY GREEN COMPONENT",cv2.merge([zeros,histImgG,zeros]))
    histImgR = hist_equal(r)
    cv2.imwrite("img_r.jpg", cv2.merge([zeros,zeros,histImgR]))
    #cv2.imshow("DISPLAY RED COMPONENT",cv2.merge([zeros,zeros,histImgR]))
    img_hist_equal = cv2.merge([histImgB, histImgG, histImgR])
    cv2.imwrite("hist_equal.jpg", img_hist_equal)

#    binaryB = binary(b)
#    binaryG = binary(g)
#    binaryR = binary(r)
#    img_binary = cv2.merge([binaryB, binaryG, binaryR])
#
    noiseB = PepperandSalt(b)
    noiseG = PepperandSalt(g)
    noiseR = PepperandSalt(r)
    img_noise = cv2.merge([noiseB,noiseG,noiseR])
    cv2.imwrite("img_noise.jpg", img_noise)
#
    filterB = medianfliter(noiseB)
    filterG = medianfliter(noiseG)
    filterR = medianfliter(noiseR)
    img_filter = cv2.merge([filterB,filterG,filterR])
    cv2.imwrite("img_filter.jpg", img_filter)
#
    img_canny = Canny_demo(img2)
#    img_cannyB = Canny_demo(b)
#    img_cannyG = Canny_demo(g)
#    img_cannyR = Canny_demo(r)
    cv2.imwrite("img_verify-canny-50-100.jpg", img_canny)
    #threshold_By_OTSU("img_gray.jpg")
#    cv2.imwrite("img_cannyB.jpg", img_cannyB)
#    cv2.imwrite("img_cannyG.jpg", img_canny)
#    cv2.imwrite("img_cannyR.jpg", img_cannyR)
#
#    cv2.imshow("img",img)
#    cv2.imshow("img_hist_equal", img_hist_equal)
#    cv2.imshow("img_binary", img_binary)
#    cv2.imshow("img_noise", img_noise)
#    cv2.imshow("img_filter", img_filter)
#    cv2.imshow("img_canny", img_canny)
#
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
