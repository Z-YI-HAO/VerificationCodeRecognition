from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os


# 通过8邻域降噪的方法去除噪声
def removeNoise(img, threshold):
    # 统计邻域内颜色为非白色的像素个数
    def notWhiteNum(img, x_pos, y_pos):
        count = 0
        img_width, img_height = img.size
        for x in [x_pos - 1, x_pos, x_pos + 1]:
            if x < 0 or x > img_width - 1:
                continue
            for y in [y_pos - 1, y_pos, y_pos + 1]:
                if y < 0 or y > img_height - 1:
                    continue
                if x == x_pos and y == y_pos:
                    continue
                if img.getpixel((x, y)) < 200:
                    count += 1
        return count

    img_width, img_height = img.size

    flag = [[0 for i in range(img_height)] for j in range(img_width)]

    for x in range(img_width):
        for y in range(img_height):
            # 将边界区域设置为白色
            if x == 0 or x == img_width - 1 or y == 0 or y == img_height - 1:
                flag[x][y] = 1

            if img.getpixel((x, y)) == 255:
                flag[x][y] = 1
                continue

            if notWhiteNum(img, x, y) < threshold:
                flag[x][y] = 1

    for x in range(img_width):
        for y in range(img_height):
            if flag[x][y] == 1:
                img.putpixel((x, y), 255)
            else:
                img.putpixel((x, y), 0)

def revertColor(img):
    img_width,img_height=img.size
    for x in range(img_width):
        for y in range(img_height):
            if img.getpixel((x, y)) == 255:
                img.putpixel((x,y),0)
            else:
                img.putpixel((x,y),255)

def find_contours(img):
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL	, cv.CHAIN_APPROX_SIMPLE)
    
    # contours=list()
    
    print(len((contours)))

    img1 = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

    i=0
    for contour in contours:
        # print(contour)
        # rect=cv.minAreaRect(contour)
        # print(rect)

        i+=1
        x,y,w,h=cv.boundingRect(contour)
        print(x,y,w,h)
        # cv.rectangle(img1,(x,y),(x+w,y+h),(0,255,0),2)
        crop_img=img1[y:y+h,x:x+w]

        top=int((100-h)/2)
        bottom=int((100-h)/2)+(100-h)%2
        left=int((90-w)/2)
        right=int((90-w)/2)+(90-w)%2
        pad_img=cv.copyMakeBorder(crop_img,top,bottom,left,right,cv.BORDER_CONSTANT,value=(0,0,0))

        cv.imwrite("img"+str(i)+".png",pad_img)

        # box=np.int0(cv.boxPoints(rect)) #矩形四个角取整
        # print(box)
        # cv.drawContours(img1,[box],0,(0,0,255),2)

    # print(hierarchy)
    # cv.drawContours(img1,contours,-1,(0,0,255),2)
    
    cv.imshow("image",img1)
    cv.waitKey(0)
    cv.destroyAllWindows()


img = Image.open("0A8X.png")
print(img.size)
img = img.convert("L")
# img.show()
removeNoise(img,5)
# img.show()
revertColor(img)
# img.show()

# cv_img=np.asarray(img)
# find_contours(cv_img)

# cv.imshow("image",cv_img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# # 显示图片
# plt.imshow(im)
# plt.show()
# # im.show()