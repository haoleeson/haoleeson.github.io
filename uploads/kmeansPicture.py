# coding: utf-8
# coder: haoleeson
import pylab
from matplotlib import image as img
from matplotlib import pyplot as plt
import numpy as np
import random
import math

# 获取图片的rgb列表
def getImageRGB(file):
    image = img.imread(file)
    width, height, x = image.shape
    # 创建与照片像素规模同大小rgb列表，存rgb数据
    rgb = np.zeros((width, height, x))
    for i in range(width):
        for j in range(height):
            rgb[i][j] = image[i, j]
    return rgb


# 随机生成k个种子，返回k个随机像素点坐标
def initCentroids(imageRGB, k):
    center = []
    for i in range(k):
        x, y = random.randint(0, imageRGB.shape[0]), random.randint(0, imageRGB.shape[1])
        center += [[x, y]]
    return center


# 计算变量中每个像素点与k个中心点的欧式距离
def caclEucDistance(imageRGB, centers):
    region = []
    for i in range(imageRGB.shape[0]): #行
        x = []
        for j in range(imageRGB.shape[1]): #列
            temp = []
            for k in range(len(centers)): #计算k个像素点与k个中心点的欧式距离
                dist = np.sqrt(np.sum(np.square(imageRGB[i, j] - imageRGB[centers[k][0], centers[k][1]])))
                temp += [dist] #添加到temp临时数组中
            x.append(np.argmin(temp)) #距离最小的集群的下标，按距离分簇
        region.append(x)
    return region #返回与数组同大小的 像素与簇对应关系


# 迭代计算变量中每个像素点与k个中心点的欧式距离
def loopCaclEucDistance(imageRGB, CalCentercolor):
    region = []
    for i in range(imageRGB.shape[0]): #行
        x = []
        for j in range(imageRGB.shape[1]): #列
            temp = []
            for k in range(len(CalCentercolor)): #计算k个像素点与k个中心点的欧式距离
                dist = np.sqrt(np.sum(np.square(imageRGB[i, j] - CalCentercolor[k])))
                temp += [dist] #添加到temp临时数组中
            x.append(np.argmin(temp))  #距离最小的集群的下标，按距离分簇
        region.append(x)
    return region #返回与数组同大小的 像素与簇对应关系


# 计算集群的平均值
def calNewCenter(features, imageRGB, k):
    temp = [] #一维数组
    for i in features:
        for j in i:
            temp.append(j)
    centercolor = [0] * k
    # 累加 每个集群中所包含的 像素点的RGB值
    for i in range(len(features)): #Rows
        for j in range(len(features[i])): #Columns
            centercolor[features[i][j]] += imageRGB[i, j]
    
    for i in range(len(centercolor)):
        centercolor[i] /= temp.count(i) #求每个集群的RGB 均值
        # 将求得的均值[取整]
        for j in range(len(centercolor[i])):  #Columns
            centercolor[i][j] = int(centercolor[i][j])
    return centercolor


# 显示分割前后对比图程序
def showImage(imageRGB, centercolor, features, k, iteration):
    NewImage = np.empty((len(features), len(features[0]), 3))
    for i in range(len(features)):
        for j in range(len(features[i])):
            NewImage[i, j] = centercolor[features[i][j]]
    # 绘制图片
    fig = plt.figure(figsize=(10, 4), facecolor='white')
    fig.suptitle('k='+str(k)+', iteration='+str(iteration), fontsize=12, color='k')
    fig.gca().xaxis.set_major_locator(plt.NullLocator())
    fig.gca().yaxis.set_major_locator(plt.NullLocator())

    # 创建子图1: 绘制原图
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.axis('off') # 关闭坐标轴显示
    ax1.imshow(imageRGB / 255)
    ax1.set_title('Original image', fontsize=10, color='k')

    # 创建子图2: 绘制分割图
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis('off') # 关闭坐标轴显示
    ax2.imshow(NewImage / 255)
    ax2.set_title('Split graph', fontsize=10, color='k')

    # 显示绘制图像
    pylab.show()
    plt.show()


# -------------------main---------------------------
def main():
    #加载图片数据
    imageRGB = getImageRGB('KmeansTestPicture.jpg')
    print('Finish load image RGB data...')
    #设置集群数：k=3
    k = 3
    # 生成k个随机像素点坐标
    InitialCenter = initCentroids(imageRGB, k)
    # 计算样本中每个像素点与k个中心点的欧式距离，并重新分类
    features = caclEucDistance(imageRGB, InitialCenter)

    #设置k-means算法执行的最大迭代次数：iteration = 20
    iteration = 20
    for i in range(iteration, 0, -1):
        print('iteration = ', i)
        CalCentercolor = calNewCenter(features, imageRGB, k) # 得到每个簇的均值
        # 根据簇中的新均值，并重新分簇
        features = loopCaclEucDistance(imageRGB, CalCentercolor)
        print('\n'+'Center[k] = ', CalCentercolor, '\n')

    #显示分割前后对比图
    showImage(imageRGB, CalCentercolor, features, k, iteration)


if __name__ == '__main__':
    main()
