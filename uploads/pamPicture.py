# coding: utf-8
# coder: haoleeson
import pylab
from matplotlib import image as img
from matplotlib import pyplot as plt
import numpy as np
import random


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


# 随机选择一个非中心点Oi
def chooseOneNoCenterSample(imageRGB, centers):
    x, y = 0, 0
    isChooseACenterSampleFlag = True
    while(isChooseACenterSampleFlag):
        isExist = False
        x, y = random.randint(0, imageRGB.shape[0]), random.randint(0, imageRGB.shape[1])
        # 判断与k个中心 是否有重复样本,
        for k in range(len(centers)):
            if(x==centers[k][0] and y==centers[k][1]):
                isExist = True
                break;
        #若无重复，则退出while,并返回[x,y]
        if( isExist == False ):
            break;
        #若重复则继续while
    return [x, y]


# 计算变量中每个像素点与k个中心点的欧式距离，并分簇
def caclEucDistance(imageRGB, centers):
    region = []
    for i in range(imageRGB.shape[0]): #行
        x = []
        for j in range(imageRGB.shape[1]): #列
            temp = []
            for k in range(len(centers)): #计算k个像素点与k个中心点的欧式距离
                dist = np.sqrt(np.sum(np.square(imageRGB[i, j] - imageRGB[centers[k][0], centers[k][1]])))
                temp += [dist] #添加到temp临时数组中
            x.append(np.argmin(temp)) #添加[i,j]像素距离k个‘种子’最小的距离于region
        region.append(x)
    return region #返回与数组同大小的最小欧式距离数组


# 计算所有对象与其簇中中心点的距离值，将其全部累加得到损失值，记为cost
def calcCost(imageRGB, features, centers):
    cost = 0.0
    for i in range(imageRGB.shape[0]): #行
        for j in range(imageRGB.shape[1]): #列
            dist = np.sqrt(np.sum(np.square(imageRGB[i, j] - imageRGB[centers[features[i][j]][0], centers[features[i][j]][1]])))
            cost = cost + dist
    return cost


#获取中心对象下标所对应的RGB值
def getCenterColor(imageRGB, centers, k):
    centercolor = [0] * k
    for i in range(k):
        centercolor[i] = imageRGB[centers[i][0], centers[i][1]]
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


#PAM算法对图片数据聚类实现
def PAM(imageRGB, features, centers):
    # 计算初始cost0
    cost0 = calcCost(imageRGB, features, centers)
    # print('cost0=', cost0)
    # 随机选择一个非中心点Oi
    aNoCenterSample = chooseOneNoCenterSample(imageRGB, centers)
    # print('Center = ', centers)
    # print('aNoCenterSample = ', aNoCenterSample)

    # 替换该随机对象所属的第k个中心对象
    belongsK = features[aNoCenterSample[0]][aNoCenterSample[1]]
    TempCenters = centers
    TempCenters[belongsK][0] = aNoCenterSample[0]  # Rows下标
    TempCenters[belongsK][1] = aNoCenterSample[1]  # Columns下标
    # 重新分簇
    TempFeatures = caclEucDistance(imageRGB, TempCenters)
    # 计算代价cost
    Tempcost = calcCost(imageRGB, TempFeatures, TempCenters)
    # 比较替换后的 cost
    if (Tempcost < cost0):
        # 若比最初损失值cost0小，则确认替换
        centers = TempCenters
        features = TempFeatures
        cost0 = Tempcost
    return features, centers


def main():
    #加载图片数据
    imageRGB = getImageRGB('PamTestPicture.jpg')
    print('Finish load image RGB data...\n')
    # 设置集群数：k=3
    k = 3
    #设置k-means算法执行的最大迭代次数：iteration = 10
    iteration = 10

    # 生成k个随机像素点坐标，作为中心点
    centers = initCentroids(imageRGB, k)
    # 计算样本中每个像素点与k个中心点的欧式距离，并根据距离分到最近簇
    features = caclEucDistance(imageRGB, centers)
    print('PAM start...\n')
    for i in range(iteration, 0, -1):
        print('iteration = ', i)
        features, centers = PAM(imageRGB, features, centers) #PAM迭代
        print('\n' + 'Centers = ', centers, '\n')

    centercolor = getCenterColor(imageRGB, centers, k)
    #显示分割前后对比图
    print('Show the Comparison images...')
    showImage(imageRGB, centercolor, features, k, iteration)


if __name__ == '__main__':
    main()
