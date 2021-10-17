# coding: utf-8
# coder: eisenhao
# 基于密度的DBSCAN聚类算法实现，并对6个数据文件数据集进行聚类

import mat4py
import numpy as np
# import pandas as pd # Pandas库：强大、灵活的数据分析和探索工具
import matplotlib.pyplot as plt # Draw figures
import random

# 全局变量 定义点的 分类、是否访问 属性值
# 访问及分类信息: -3，未访问； -2，噪声点;
# 0 ~ N， 第0类点；...，第i类点;...第N类点
NoVisitedValue = -3 #未访问
VisitedValue = -4 #已访问
NoiseValue = -2    #噪声点
isAllVisitedValue = -8

# 加载数据文件文件数据
def loadDataFile(filename):
    print('加载', filename, '文件数据...')
    data = mat4py.loadmat(filename)
    return data


# 随机选择一个未访问过的点 p
def chooseOneNoVisitedP(data, classification):
    len_data = len(data)
    isExisted = False  # 是否存在未访问点标识, 初始化 False

    # 查询是否存在 未访问点
    for i in range(len_data):
        # 若存在（至少一个）未访问点，退出循环
        if (classification[i] == NoVisitedValue):
            isExisted = True
            break
    # 若不存在（至少一个）未访问点，所有变量均已访问，返回不存在标记
    if (isExisted == False):
        return isAllVisitedValue

    # 若未被访问的点 数量非0，生成随机点下标
    p = random.randint(0, len_data - 1)
    while(True):
        # 判断 p 是否访问过:
        #若没访问过，则退出while,并返回 p
        if ( classification[p] == NoVisitedValue ):
            break;
        #若已访问过，则增加 p 下标继续while
        p = (p + 1) % len_data

    return p #返回未访问的点 p 在 data 列表的下标


# 计算'data'列表中 下标'p1'与下标'p2'的欧式距离
def calcEucDistance(data, p1, p2):
    data1 = np.array([data[p1][0], data[p1][1]])
    data2 = np.array([data[p2][0], data[p2][1]])
    dist = np.sqrt(np.sum(np.square(data1 - data2)))
    return dist


# 找到 P 点的 Eps 邻域内的所有点, (包括 P 点自身)
def regionQuery(data, P, Eps):
    region = []
    for i in range(len(data)):
        # 若邻域内的所有点, (不包括 P 点自身)
        # if (i == P):
        #     continue #跳过
        dist = calcEucDistance(data, P, i)
        if (dist <= Eps):
            region += [i] # 将在 P 邻域内 的点(下标)添加到region列表
    return region


# 判断是否访问
# 已访问返回：True
# 未访问返回：False
def isVisited(classification, P):
    if (classification[P] == NoVisitedValue):
        return False
    return True


# 将 P 的邻域中 为中心对象的 P'的邻域内对象添加到 P 的邻域内
def addNeighborPts(NeighborPts, NeighborPts_i):
    for k in range(len(NeighborPts_i)):
        isExistedFlag = False # NeighborPts_i[k] 是否重复标志
        # 查询 NeighborPts 中有与 NeighborPts_i[k] 重复的点
        for i in range(len(NeighborPts)):
            # 若有重复
            if (NeighborPts_i[k] == NeighborPts[i]):
                isExistedFlag = True
                break
        # 若第k个邻居不重复， 则添加
        if (isExistedFlag == False):
            NeighborPts += [NeighborPts_i[k]]

    return NeighborPts


# 判断当前 P 不属于任何簇
# 不属于任何簇, 返回 True
# 属于某一簇，返回 False
# Cluster 为分簇信息（二维数组）， P 为数据下标
def isNoBelongsToAnyCluster(Cluster, Cluster_Pi, P):
    for i in range(len(Cluster)):
        for j in range(len(Cluster[i])):
            if (P == Cluster[i][j]):
                return False
    for k in range(len(Cluster_Pi)):
        if (P == Cluster_Pi[k]):
            return False

    return True


# 扩展当前核心对象 P 的所属簇
def expandCluster(data, classification, P, NeighborPts, Cluster, Eps, MinPts):
    Cluster_P = []
    Cluster_P += [P] #将核心对象添加到 P 的临时所属簇
    # 遍历点 P 的所有 NeighborPts， 这个NeighborPts可能会增加
    i = 0
    while(True):
        # 若邻域内的 P' 正好是 P，跳过
        if (NeighborPts[i] == P):
            i = i + 1
            # 若所有 NeighborPts 均已遍历，退出while
            if (i >= len(NeighborPts)):
                break
            else:
                continue # 跳过
        # 若 P' 未访问
        if (isVisited(classification, NeighborPts[i]) == False):
            classification[NeighborPts[i]] = VisitedValue # 标记 P' 已访问
            NeighborPts_i = regionQuery(data, NeighborPts[i], Eps) # 求 P' Eps邻域内的点
            if (len(NeighborPts_i) >= MinPts): # 若 P' 也为 核心对象
                NeighborPts = addNeighborPts(NeighborPts, NeighborPts_i) #增加P' Eps邻域内的点到 点 P 的邻域元素 NeighborPts中

            # 若 P' 不属于任何一个已有类(将当前邻域内元素添加到核心对象 P 的所属簇)
            if (isNoBelongsToAnyCluster(Cluster, Cluster_P, NeighborPts[i]) == True):
                Cluster_P += [NeighborPts[i]]

        i = i + 1
        # 若所有 NeighborPts 均已遍历，退出while
        if (i >= len(NeighborPts)):
            break

    Cluster += [Cluster_P]
    return classification, Cluster

# 基于密度的DBSCAN算法
def DBSCAN(data, Eps, MinPts):
    print('开始DBSCAN...')
    # 创建一个同规模数组，记录每个点的 分类、是否访问
    classification = np.ones(len(data)) * NoVisitedValue  # 存储是否访问及分类信息: -3，未访问； -2，噪声点；
    Cluster = [] #记录分簇信息，第i行存储所有归类于第i簇的元素下标

    while(True):
        # 选择一个 未访问的 点 P
        P = chooseOneNoVisitedP(data, classification)
        if (P == isAllVisitedValue): #已经全部遍历，退出While循环
            break;
        # # 若点 P 已被访问，跳过（注释，包含点P）
        # if (isVisited(classification, P) == True):
        #     continue
        #  若点 P 未被访问， 标记 P 为已访问
        classification[P] = VisitedValue
        NeighborPts = regionQuery(data, P, Eps)
        # print(P,'\'s NeighborPts is: ', NeighborPts)
        if (len(NeighborPts) < MinPts):  # 若 P非核心对象
            #标记 P 为 NOISE
            classification[P] = NoiseValue
        else:
            # print('扩展当前核心对象 P = ', P, '的所属簇')
            classification, Cluster = expandCluster(data, classification, P, NeighborPts, Cluster, Eps, MinPts)

    return Cluster

# 打印簇信息
def PrintCluster(Cluster):
    print('簇个数为：', len(Cluster))
    print('每簇元素个数分别为：')
    for i in range(len(Cluster)):
        print('No.', i, ', num = ', len(Cluster[i]))


# 计算给定二维数组的第一列，第二列元素值的范围
def getMinMaxXandY(data):
    Xmin, Xmax, Ymin, Ymax = data[0][0], data[0][0], data[0][1], data[0][1]
    for i in range(len(data)):
        if data[i][0] < Xmin:
            Xmin = data[i][0]
        if data[i][0] > Xmax:
            Xmax = data[i][0]
        if data[i][1] < Ymin:
            Ymin = data[i][1]
        if data[i][1] > Ymax:
            Ymax = data[i][1]
    return [Xmin, Xmax, Ymin, Ymax]


# 绘制原始数据图形
def drawPictureOfInitialData(data, title):
    len_data = len(data)
    # Plot data
    x = []
    y = []
    for i in range(len_data):
        # if data[i][2] == 0:
            x += [data[i][0]]
            y += [data[i][1]]
    plt.scatter(x, y, s=4)
    # 设置标题并加上轴标签
    plt.title(title, fontsize=24)
    # 设置刻度标记的大小
    plt.tick_params(axis='both', which='major', labelsize=14)
    # 设置每个坐标的取值范围
    plt.axis(getMinMaxXandY(data))
    plt.show()


# 显示集群图像（最多表示10种颜色的集群）
def showCluster(dataSet, k, centroids, clusterAssment):
    numSamples, dim = dataSet.shape

    # draw all samples
    for i in range(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

#           红       蓝   绿    黑
    mark = ['Dr', 'Db', 'Dg', 'Dk', 'Dy', 'Dc', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # draw the centroids
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)

    plt.show()

# 显示聚类前后对比图（最多表示8种颜色的集群）
def showClusterImage(data, Cluster, Eps, MinPts):
    num_Cluster = len(Cluster) # 分类个数

    # 绘制第 i 个类的 颜色列表
    # b--blue, c--cyan, g--green, k--black
    # m--magenta, r--red, w--white, y--yellow
    color = ['b', 'c', 'g', 'k', 'm', 'r', 'w', 'y']
    # 绘制第 i 个类的 形状列表
    mark = ['.', 'o', '^', '1', '8', 's', 'p', '*', 'h', '+', 'D']
    if num_Cluster > len(color):
        print('Sorry! Your len(Cluster)=', len(Cluster), ' is too large!( > len(color)=', len(color), ')')
        return 1

    # 绘制图片
    fig = plt.figure(figsize=(10, 4), facecolor='white')
    fig.suptitle('Eps='+str(Eps)+', MinPts='+str(MinPts)+'  Result Cluster='+str(len(Cluster)), fontsize=12, color='k')
    fig.gca().xaxis.set_major_locator(plt.NullLocator())
    fig.gca().yaxis.set_major_locator(plt.NullLocator())


    # 创建子图1: 绘制原图
    ax1 = fig.add_subplot(1, 2, 1)
    len_data = len(data)
    x = []
    y = []
    for i in range(len_data):
        x += [data[i][0]]
        y += [data[i][1]]
    ax1.scatter(x, y, s=4, c='k')
    # 设置标题并加上轴标签
    ax1.set_title('Original Graph', fontsize=10, color='k')
    # 设置坐标的取值范围
    ax1.axis(getMinMaxXandY(data))


    # 创建子图2: 绘制DBSCAN
    ax2 = fig.add_subplot(1, 2, 2)
    # 分别绘制不同 类别点
    for i in range(len(Cluster)):
        # 提取第 i 个类别的(x, y)坐标
        x = []
        y = []
        for j in range(len(Cluster[i])):
            x += [data[Cluster[i][j]][0]]
            y += [data[Cluster[i][j]][1]]
        # 对第 i 个类别内所有点 以不同颜色绘制
        ax2.scatter(x, y, s=4, c=color[i], marker=mark[i])
    ax2.set_title('DBSCAN Graph', fontsize=10, color='k')
    # 设置坐标的取值范围
    ax2.axis(getMinMaxXandY(data))

    plt.show() # 显示绘制图像

# main函数
if __name__ == '__main__':
    # # 加载long.mat数据
    # data = loadDataFile('long.mat')
    # data = data['long1'] #Eps=0.15， MinPts=8，最终分类数：2

    # # 加载moon.mat数据
    # data = loadDataFile('moon.mat')
    # data = data['a']  # Eps=0.11， MinPts=5，最终分类数：2

    # # 加载sizes5.mat数据
    # data = loadDataFile('sizes5.mat')
    # data = data['sizes5']  # Eps=1.32， MinPts=10，最终分类数：4

    # 加载smile.mat数据
    data = loadDataFile('smile.mat')
    data = data['smile']  # Eps=0.08， MinPts=10，最终分类数：3

    # # 加载spiral.mat数据
    # data = loadDataFile('spiral.mat')
    # data = data['spiral']  # Eps=1， MinPts=8，最终分类数：2

    # # 加载square1.mat数据
    # data = loadDataFile('square1.mat')
    # data = data['square1']  # Eps=1.2， MinPts=9，最终分类数：4

    # # 加载square4.mat数据
    # data = loadDataFile('square4.mat')
    # data = data['b'] # Eps=1， MinPts=20，最终分类数：4

    # 加载2d4c.mat数据
    # data = loadDataFile('2d4c.mat')
    # data = data['a']  #2d4c_a_ Eps=1.5， MinPts=20，最终分类数：4
    # data = data['moon']  #2d4c_moon_ Eps=0.11， MinPts=5，最终分类数：2
    # data = data['smile']  #2d4c_smile_ Eps=0.08， MinPts=10，最终分类数：3
    # data = data['b']  #2d4c_smile_ Eps=1， MinPts=20，最终分类数：4


    # DBSCAN两个重要参数
    Eps = 0.08 #邻域半径
    MinPts = 10 #邻域内元素个数（包括点P）
    Cluster = DBSCAN(data, Eps, MinPts) # 执行 DBSCAN 算法，返回聚类后的下标二维数组
    print('All Done')
    PrintCluster(Cluster) # 打印簇信息
    showClusterImage(data, Cluster, Eps, MinPts) # 绘制基于密度的DBSCAN算法后的效果图

