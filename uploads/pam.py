# coding: utf-8
# coder: haoleeson
import numpy as np
import pandas as pd
import copy

# 获取噪声文件的部分数据前number个
def getPartWaveNoiseData(filename, number):
    # print('含噪声数据加载...\n')
    df = pd.read_csv(filename, sep=',', engine='python', header=None, skiprows=0, names=None)  # waveform-+noise.data
    data_numSamples, data_dim = df.shape
    #取含噪声waveform数据集前number个数据样本进行聚类
    s = np.array(df)
    data0 = s[:number]
    data = np.array([data0])
    data = data.reshape(-1, data_dim)
    return data


# 计算欧式距离:
# 将'vector1'与'vector2'中所有对应属性值之差的平方求和，再求平方根
def caclEucDistance(vector1, vector2):
    distance = np.sqrt(np.sum(np.square(vector2 - vector1), axis=1))
    return distance


# PAM算法实现
# iteration:最大迭代次数，k:'种子'集群个数，data:待聚类数据
def PAM(iteration, k, data):
    data_numSamples, data_dim = data.shape
    data_new = copy.deepcopy(data)  # 前40列存放数据，不可变。最后1列即第41列存放标签，标签列随着每次迭代而更新。
    data_now = copy.deepcopy(data)  # data_now用于存放中间过程的数据

    center_point = np.random.choice(data_numSamples, k, replace=False)
    center = data_new[center_point, :(data_dim-2)]  # 随机形成的k个中心，维度为（3，40）

    distance = [[] for i in range(k)]
    distance_now = [[] for i in range(k)]  # distance_now用于存放中间过程的距离
    lost = np.ones([data_numSamples, k]) * float('inf')  # 初始lost为维度为（numSamples，3）的无穷大

    for j in range(k):  # 首先完成第一次划分，即第一次根据距离划分所有点到k个类别中
        distance[j] = np.sqrt(np.sum(np.square(data_new[:, :(data_dim-2)] - np.array(center[j])), axis=1))
    data_new[:, data_dim-1] = np.argmin(np.array(distance), axis=0)  # data_new 的最后一列，即标签列随之改变，变为距离某中心点最近的标签，例如与第0个中心点最近，则为0

    for i in range(iteration):  # 假设迭代n次

        for m in range(k):  # 每一次都要分别替换k=k个中心点，所以循环k次。这层循环结束即算出利用所有点分别替代k个中心点后产生的i个lost值

            for l in range(data_numSamples):  # 替换某个中心点时都要利用全部点进行替换，所以循环numSamples次。这层循环结束即算出利用所有点分别替换1个中心点后产生的numSamples个lost值

                center_now = copy.deepcopy(center)  # center_now用于存放中间过程的中心点
                center_now[m] = data_now[l, :(data_dim-2)]  # 用第l个点替换第m个中心点
                for j in range(k):  # 计算暂时替换1个中心点后的距离值
                    distance_now[j] = np.sqrt(np.sum(np.square(data_now[:, :(data_dim-2)] - np.array(center_now[j])), axis=1))
                data_now[:, (data_dim-1)] = np.argmin(np.array(distance),
                                            axis=0)  # data_now的标签列更新，注意data_now时中间过程，所以这里不能选择更新data_new的标签列

                lost[l, m] = (caclEucDistance(data_now[:, :(data_dim-2)], center_now[data_now[:, (data_dim-1)].astype(int)]) \
                              - caclEucDistance(data_now[:, :(data_dim-2)], center[data_new[:, (data_dim-1)].astype(
                            int)])).sum()  # 这里很好理解lost的维度为什么为numSamples*3了。lost[l,m]的值代表用第l个点替换第m个中心点的损失值

        if np.min(lost) < 0:  # lost意味替换代价，选择代价最小的来完成替换
            index = np.where(np.min(lost) == lost)  # 即找到min(lost)对应的替换组合
            index_l = index[0][0]  # index_l指将要替代某个中心点的候选点
            index_m = index[1][0]  # index_m指将要被替代的某个中心点，即用index_l来替代index_m

        center[index_m] = data_now[index_l, :data_dim-2]  # 更新聚类中心

        for j in range(k):
            distance[j] = np.sqrt(np.sum(np.square(data_now[:, :(data_dim-2)] - np.array(center[j])), axis=1))
        data_new[:, (data_dim-1)] = np.argmin(np.array(distance), axis=0)  # 更新参考矩阵,至此data_new的标签列得以更新，即完成了一次迭代

    return center ,data_new  # 最后返回center：集群中心对象，data_new：其最后一列即为最终聚好的标签


if __name__ == '__main__':
    # 获取噪声文件的部分数据前number个
    data = getPartWaveNoiseData('waveform-+noise.data', 500)
    data_numSamples, data_dim = data.shape
    
    iteration = 10 # iteration:最大迭代次数
    k = 3 # k:'种子'集群个数
    center, data_new = PAM(iteration, k, data)
    print('\ncenter\n', center)
    print(data_new[:, (data_dim-1)])
    # print(np.mean(data[:, (data_dim-1)] == data_new[:, (data_dim-1)]))  # 验证划分准确度
