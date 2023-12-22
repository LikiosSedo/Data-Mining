import matplotlib.pyplot as plt
import cv2

#灰度图像的二值化


#首先把一张彩色图片灰度化

#根据路径读取一张图片
img = cv2.imread('hist_equal.jpg')

# cv2.cvtColor(src, code[, dst[, dstCn]])
#函数作用：方法用于将图像从一种颜色空间转换为另一种颜色空间
#函数释义：
# src:它是要更改其色彩空间的图像。
# code:它是色彩空间转换代码。
# dst:它是与src图像大小和深度相同的输出图像。它是一个可选参数。
# dstCn:它是目标图像中的频道数。如果参数为0，则通道数自动从src和代码得出。它是一个可选参数。
#这里的作用是将彩色图片lenna.png转换为一张灰度图片img_gray
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imwrite("img_gray.jpg", img_gray)


#img_gray.shape：获取img_gray这张灰度图片的长*宽
hight,width = img_gray.shape

#循环打印img_gray中的每一个灰度值的坐标
for i in range(hight):
    for j in range(width):
        #img_gray的dtype是uint8,所有灰度值取值范围为0-255，
        #将大于128的变成255，小于128的变成0，就得到二值图的灰度值
        if img_gray[i,j] <= 128:
            img_gray[i,j] = 0
        else:
            img_gray[i,j] = 255
            





'''
这里有个需要特别注意的地方
cv2.imshow()没办法把二值图正常展示出来
需要用到plt
'''

plt.subplot(111)
plt.imshow(img_gray,cmap='gray')
plt.savefig("Binarization.png")
plt.show()

