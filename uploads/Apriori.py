# -*- coding: utf-8 -*-

from numpy import *
import pandas as pd # Pandas库：强大、灵活的数据分析和探索工具
import matplotlib.pyplot as plt # Draw figures
import re #find int from str
from ID3 import *

#加载数据集
def loadDataSet():
    return [[1, 3, 4], [2, 3], [1, 2, 3, 5], [2, 5]]


#加载BigMart 数据
def loadBigSmartDataSet():
    # 数据加载 BigMart
    print("\n加载 BigMart 数据...\n")
    # 定义销售数据11个属性列名
    Big_variables = ['Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type', 'Item_MRP',
                     'Outlet_Identifier', 'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type',
                     'Outlet_Type', 'Item_Outlet_Sales']  # Train_UWu5bXk.csv
    Big_data = pd.read_csv('Train_UWu5bXk.csv', sep=',', engine='python', header=None, skiprows=1,
                           names=Big_variables)  # Train_UWu5bXk.csv
    return Big_data


#数据加载 Black Friday
def loadBlackFridayDataSet():
    print("\n加载 Black Friday 数据...\n")
    Black_variables = ['User_ID', 'Product_ID', 'Gender', 'Age', 'Occupation', 'City_Category',
                   'Stay_In_Current_City_Years', 'Marital_Status', 'Product_Category_1',
                   'Product_Category_2', 'Product_Category_3', 'Purchase']
    Black_data = pd.read_csv('BlackFridayTrain.csv', sep=',', engine='python', header=None, skiprows=1,
                       names=Black_variables)  # BlackFridayTrain.csv
    return Black_data


# C1 是大小为1的所有候选项集的集合
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item]) #store all the item unrepeatly
    # C1.sort()
    #return map(frozenset, C1)#frozen set, user can't change it.
    return list(map(frozenset, C1))


#扫描数据库，返回频繁出现的候选项目集Lk（出现频率大于给定阈值minSupport）
def scanD(D, Ck, minSupport):
    ssCnt={}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                #if not ssCnt.has_key(can):
                if not can in ssCnt:
                    ssCnt[can]=1
                else: ssCnt[can]+=1
    numItems=float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems #compute support
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData


#total apriori组合，向上合并L
def aprioriGen(Lk, k):
    #creates Ck 参数：频繁项集列表 Lk 与项集元素个数 k
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk): #两两组合遍历
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1 == L2: #若两个集合的前k-2个项相同时, 则将两个集合合并
                retList.append(Lk[i] | Lk[j]) #set union
    return retList


#生成候选规则集合：计算规则的可信度以及找到满足最小可信度要求的规则
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    #针对项集中只有两个元素时，计算可信度
    prunedH = []#返回一个满足最小可信度要求的规则列表
    for conseq in H:#后件，遍历 H中的所有项集并计算它们的可信度值
        conf = supportData[freqSet]/supportData[freqSet-conseq] #可信度计算，结合支持度数据
        if conf >= minConf:
            print (freqSet-conseq, '-->', conseq, 'conf:', conf)
            #如果某条规则满足最小可信度值, 那么将这些规则输出到屏幕显示
            brl.append((freqSet-conseq, conseq, conf))#添加到规则里，brl 是前面通过检查的 bigRuleList
            prunedH.append(conseq)#同样需要放入列表到后面检查
    return prunedH


#合并
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    #参数:一个是频繁项集, 另一个是可以出现在规则右部的元素列表 H
    m = len(H[0])
    if (len(freqSet) > (m + 1)): #频繁项集元素数目大于单个集合的元素数
        Hmp1 = aprioriGen(H, m+1)#存在不同顺序、元素相同的集合，合并具有相同部分的集合
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)#计算可信度
        if (len(Hmp1) > 1):    #满足最小可信度要求的规则列表多于1, 则递归
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)


#生成关联规则
def generateRules(L, supportData, minConf=0.7):
    #频繁项集列表、包含那些频繁项集支持数据的字典、最小可信度阈值
    bigRuleList = [] #存储所有的关联规则
    for i in range(1, len(L)):  #只获取有两个或者更多集合的项目，从1, 即第二个元素开始，L[0]是单个元素的
        # 两个及以上的才可能有关联一说，单个元素的项集不存在关联问题
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            #该函数遍历L中的每一个频繁项集并对每个频繁项集创建只包含单个元素集合的列表H1
            if (i > 1):
            #如果频繁项集元素数目超过2, 那么会考虑对它做进一步的合并
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:#第一层时，后件数为1
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)# 调用函数2
    return bigRuleList


#apriori算法核心函数
def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet)
    D = list(map(set, dataSet)) #python3
    L1, supportData = scanD(D, C1, minSupport)#单项最小支持度判断 0.5，生成L1
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):#创建包含更大项集的更大列表, 直到下一个大的项集为空
        Ck = aprioriGen(L[k-2], k)#Ck
        Lk, supK = scanD(D, Ck, minSupport)#get Lk
        supportData.update(supK)
        L.append(Lk)
        k += 1 #继续向上合并 生成项集个数更多的
    return L, supportData


# 从Big_data中选取商品的部分属性用于Apriori测试
# 返回 Big_TestData  (List类型)
def getBigTestDataFrom(Big_data):
    Big_TestData = []
    for i in range(len(Big_data)):
    # for i in range(3):
        temp = []
        # 属性1： Item_Fat_Content, 产品是否低脂肪
        Attributes1 = Big_data['Item_Fat_Content'][i]
        temp += [Attributes1]

        # 属性2： Item_Type, 产品所属的类别
        Attributes2 = Big_data['Item_Type'][i]
        temp += [Attributes2]

        # 属性3： Outlet_Establishment_Year, 商店成立的那一年
        Attributes3 = Big_data['Outlet_Establishment_Year'][i]
        temp += [Attributes3]

        # 属性4： Outlet_Size, 商店的面积覆盖面积
        Attributes4 = Big_data['Outlet_Size'][i]
        temp += [Attributes4]

        # 属性5： Outlet_Location_Type, 商店所在的城市类型
        Attributes5 = Big_data['Outlet_Location_Type'][i]
        temp += [Attributes5]

        # 属性6： Outlet_Type, 出口是一家杂货店还是某种超市
        Attributes6 = Big_data['Outlet_Type'][i]
        temp += [Attributes6]

        Big_TestData.append(temp)

    return Big_TestData


# 从 Black_data 中选取数据的部分属性值用于Apriori测试
# 返回 Big_TestData  (List类型)
def getBlackFridayTestDataFrom(Black_data):
    len_Black_data = 27570 #len(Black_data)
    BlackFriday_TestData = []
    This_User_ID = Black_data['User_ID'][0]
    i = 0
    while (i < len_Black_data):
        temp = []
        # 属性1： Gender, 性别
        Attributes1 = Black_data['Gender'][i]
        temp += [Attributes1]
        # 属性2： Age, 年龄段
        Attributes2 = Black_data['Age'][i]
        temp += [Attributes2]
        #属性3：City_Category, 城市类别
        Attributes3 = Black_data['City_Category'][i]
        temp += [Attributes3]
        # 添加购买商品，Product_ID
        Attributes4 = Black_data['Product_ID'][i]
        temp += [Attributes4]
        #同一用户多个商品：
        while (i+1 < len_Black_data):
            #试探检测下一i是否为同一用户
            if Black_data['User_ID'][i+1] == This_User_ID:
                i = i + 1
                temp += [Black_data['Product_ID'][i]]
            else:
                # print(i, ':  ID=', This_User_ID)
                This_User_ID = Black_data['User_ID'][i + 1]
                break
        BlackFriday_TestData.append(temp)
        i = i + 1
    return BlackFriday_TestData


# 格式化控制输出的频繁关联项集合的显示
def showLAndSupport(L, supportData):
    print('格式化控制输出的频繁关联项集合的显示:')
    for i in range(len(L)-1):
        print('* 包含', i+1, '项的频繁关联项集合：')
        for key in L[i]:
            print('    ', key, '=', supportData[key])


# main函数
if __name__ == "__main__":
    # Q1：用Python实现apriori算法，挖掘频繁项集
    # # 加载数据集
    # dataSet = loadDataSet()
    # L, supportData = apriori(dataSet, 0.5)
    # print('L = ', L)
    # print('supportData = ', supportData)


    # # Q2_1： 用Apriori算法对Bigmart销售数据进行分析
    # # 数据加载 BigMart
    # Big_data = loadBigSmartDataSet()
    # # 预处理： 填补缺失值
    # # 使用平均数填补 Item_Weight
    # Item_Weight_mean = Big_data['Item_Weight'].mean()  # 计算'Item_Weight'平均值
    # Big_data['Item_Weight'] = Big_data['Item_Weight'].fillna(Item_Weight_mean)  # 用'Item_Weight'平均值填充缺失值
    # # 使用出现次数最多的值填补 Outlet_Size
    # Outlet_Size_mode = Big_data['Outlet_Size'].mode()  # 获取'utlet_Size'众数
    # Big_data['Outlet_Size'] = Big_data['Outlet_Size'].fillna(Outlet_Size_mode[0])  # 用'Outlet_Size'出现最多的值填充缺失值
    # # 预处理：数据转换
    # # Item_Fat_Content
    # Item_Fat_Content_mapping = {'reg': 'Regular', 'Low Fat': 'lowFat', 'LF': 'lowFat', 'low fat': 'lowFat'}
    # Big_data['Item_Fat_Content'] = Big_data['Item_Fat_Content'].map(Item_Fat_Content_mapping)
    # Big_data['Item_Fat_Content'] = Big_data['Item_Fat_Content'].fillna('Regular') #缺失值填充
    # #选取商品的部分属性用于Apriori测试
    # Big_TestData = getBigTestDataFrom(Big_data)
    # #Q2_1：用Apriori算法对Bigmart销售数据，挖掘频繁项集
    # L, supportData = apriori(Big_TestData, 0.14)
    # showLAndSupport(L, supportData)


    # #Q2_2：用Apriori算法对 黑色星期五 销售数据进行分析
    # #数据加载 Black Friday
    # Black_data = loadBlackFridayDataSet()
    # # 选取商品的部分属性用于Apriori测试
    # BlackFriday_TestData = getBlackFridayTestDataFrom(Black_data)
    # # Q2_2：用Apriori算法对 Black Friday 销售数据，挖掘频繁项集
    # L, supportData = apriori(BlackFriday_TestData, 0.1)
    # showLAndSupport(L, supportData)


    # Q3：使用决策树方法找出商品销售中哪些属性是对促销有关键作用
    # 数据加载 BigMart
    Big_data = loadBigSmartDataSet()
    # 预处理： 填补缺失值
    # 使用平均数填补 Item_Weight
    Item_Weight_mean = Big_data['Item_Weight'].mean()  # 计算'Item_Weight'平均值
    Big_data['Item_Weight'] = Big_data['Item_Weight'].fillna(Item_Weight_mean)  # 用'Item_Weight'平均值填充缺失值
    # 使用出现次数最多的值填补 Outlet_Size
    Outlet_Size_mode = Big_data['Outlet_Size'].mode()  # 获取'utlet_Size'众数
    Big_data['Outlet_Size'] = Big_data['Outlet_Size'].fillna(Outlet_Size_mode[0])  # 用'Outlet_Size'出现最多的值填充缺失值
    # 预处理：数据转换
    # Item_Fat_Content
    Item_Fat_Content_mapping = {'reg': 'Regular', 'Low Fat': 'lowFat', 'LF': 'lowFat', 'low fat': 'lowFat'}
    Big_data['Item_Fat_Content'] = Big_data['Item_Fat_Content'].map(Item_Fat_Content_mapping)
    Big_data['Item_Fat_Content'] = Big_data['Item_Fat_Content'].fillna('Regular') #缺失值填充
    #选取商品的部分属性用于Apriori测试
    Big_TestData = getBigTestDataFrom(Big_data)
    #Q3：使用决策树方法找出商品销售中哪些属性是对促销有关键作用
    dataLabels = ['Item_Fat_Content', 'Item_Type', 'Outlet_Establishment_Year',
                  'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']  # 数据的属性名称
    myTree = createTree(Big_TestData, dataLabels)  # 构造决策树
    print('决策树构造函数测试', myTree)
    createPlot(myTree)  # 显示决策树