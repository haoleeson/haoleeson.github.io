# coding: utf-8
# coder: eisenhao
import random
import numpy as np
from matplotlib import image as img #读取图片
import matplotlib.pyplot as plt # 绘图
import pandas as pd # Pandas库：强大、灵活的数据分析和探索工具

np.set_printoptions(suppress=True) #使程序输出不是科学计数法的形式，而是浮点型的数据输出

# 计算欧式距离:
# 将'vector1'与'vector2'中所有对应属性值之差的平方求和，再求平方根
def caclEucDistance(vector1, vector2):
    distance = np.sqrt(np.sum(np.square(vector2 - vector1)))
    return distance


# 随机生成 k 个 “种子”
def initCentroids(dataSet, k):
    numSamples, dim = dataSet.shape
    centroids = np.zeros((k, dim)) #创建大小为：'k x 元素属性列值' 的全0矩阵，用于存放k个‘种子’（集群）
    # 随机生成 k 个 “种子”
    for i in range(k):
        index = int(random.uniform(0, numSamples))
        centroids[i, :] = dataSet[index, :] #在数据集中取 k 个值
    return centroids


# k-means 算法实现
def kmeans(dataSet, k):
    numSamples = dataSet.shape[0]
    clusterAssment = np.mat(np.zeros((numSamples, 2)))
    # clusterAssment '元素属性值行 x 2列矩阵'
    # 第一列：存储此示例所属的('种子')集群
    # 第二列：存储当前(第i个)元素与其所属的('种子')集群的欧式距离
    clusterChanged = True
    # clusterChanged：在迭代时'种子'是否改变标志

    ## 随机生成 k 个 “种子”
    centroids = initCentroids(dataSet, k)

	# 一直迭代直到没有1个'种子'的所属('种子')集群发生改变
    while clusterChanged:
   
        # 迭代前为'种子'是否改变标志赋初值(False:未改变，如果执行完循环体后仍未False,视作迭代完成，退出迭代)
        clusterChanged = False 
                
        ## 依次求解每个元素对k个'种子'的最小距离
        for i in range(numSamples):
            minDist = 100000.0
            minIndex = 0
            
            #比较当前(第i个)元素 对应 k个'种子' 的欧式距离，求出最小（存于minDist），并记录对应种子编号（存于minIndex）
            for j in range(k):
                distance = caclEucDistance(centroids[j, :], dataSet[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j

            ## 如果当前(第i个)元素的与k个‘种子’(集群)的最小欧式距离所对应的【种子编号】遇上一次记录不一致 ==> (第i个)元素所属集群发生改变
            if clusterAssment[i, 0] != minIndex:
            	# 该变标志置'True' , 更新当前(第i个)元素的所属集群编号
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist ** 2

        ## 重新计算各个集群的中心点，并更新
        for j in range(k):
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]
            centroids[j, :] = np.mean(pointsInCluster, axis=0)

    # 此处表示退出迭代 ==> 聚类成功
    print('Congratulations, cluster complete!')
    return centroids, clusterAssment


# 最多表示10种颜色的'种子'集群
def showCluster(dataSet, k, centroids, clusterAssment):
    numSamples, dim = dataSet.shape
    if dim != 2:
        print('Sorry! I can not draw because the dimension of your data is not 2!')
        return 1

    mark = ['or', 'ob', 'og', 'ok', 'oy', 'oc', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print('Sorry! Your k is too large!(>10)')
        return 1

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


# 在data中，第0~2类，每获取100个数据并简单拼接
def geteveryClass3Data100Samples(data):
    data_class0 = data[data[21]==0].head(100) #取第0类数据前100个
    data_class1 = data[data[21]==1].head(100) #取第1类数据前100个
    data_class2 = data[data[21]==2].head(100) #取第2类数据前100个
    data_3x100 = pd.concat([data_class0, data_class1, data_class2]) #简单拼接到一起
    data_3x100 = data_3x100.sort_index() #将这300个样本，按原文件的索引从小到大排序(打乱第0~2类的数据)
    data_3x100 = data_3x100.reset_index(drop=True) #重新建立索引
    return data_3x100


# 获取图片的rgb数据
def inputImage(file):
    image = img.imread(file, 'r')
    width, height, x = image.shape
    # 创建与照片像素规模同大小rgb列表，存rgb数据
    rgb = np.zeros((width, height, x))
    for i in range(width):
        for j in range(height):
            rgb[i][j] = image[i, j]
    return rgb


# ----------------------main----------------------------------
# 用K-means算法对UCI的waveform数据集中筛选出的300个样本进行分割

# 数据加载
print('S1.数据加载...\n')
data = pd.read_csv('waveform.data', sep=',', engine='python', header=None, skiprows=0, names=None) # waveform.data
# 在3类中每类取100个数据使用上一步实现的K-means算法的进行聚类
data_3x100 = geteveryClass3Data100Samples(data)

print('S2.运行k-means计算...')
data_3x100 = np.mat(data_3x100)
k = 10
centroids, clusterAssment = kmeans(data_3x100, k)

print('\ncentroids:\n', centroids)
print('\nclusterAssment:\n', clusterAssment)


