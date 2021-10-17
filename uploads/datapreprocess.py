#import numpy as np # Numpy库:提供数组支持，以及相应的高效的处理函数
import pandas as pd # Pandas库：强大、灵活的数据分析和探索工具
import matplotlib.pyplot as plt # Draw figures
import re #find int from str
import string

# 数据加载
#print("数据加载:\n")
# 定义销售数据12个列名
variables=['Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type', 'Item_MRP', 'Outlet_Identifier', 'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Item_Outlet_Sales'] # Train_UWu5bXk.csv
data = pd.read_csv('Train_UWu5bXk.csv', sep=',', engine='python', header=None, skiprows=1, names=variables) # Train_UWu5bXk.csv

'''
# 数据加载
print("数据加载:\n")
# 定义销售数据11个属性列名
variables=['Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type', 'Item_MRP', 'Outlet_Identifier', 'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'] # Test_u94Q5KV.csv
data = pd.read_csv('Test_u94Q5KV.csv', sep=',', engine='python', header=None, skiprows=1, names=variables) # Test_u94Q5KV.csv
'''

'''
# 数据粗略查看
print("\n查看各字段的信息")
data.info()     #查看各字段的信息

#最小值、最大值、四分位数和数值型变量的均值，以及因子向量和逻辑型向量的频数统计
print('\n数据粗略查看:')
list_view = data.describe()    #使用describe函数输出计算结果
list_view.loc['jicha'] = list_view.loc['max'] - list_view.loc['min']    #求极差
list_view.loc['bianyixishu'] = list_view.loc['std']/list_view.loc['mean']    #变异系数
list_view.loc['sifenweijianju'] = list_view.loc['75%'] - list_view.loc['25%']    #四分位间距
print(list_view)

# 查询丢失信息
print('\n输出每个列丢失值也即值为NaN的数据和，并从多到少排序：')
total = data.isnull().sum().sort_values(ascending=False)
percent =(data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data)
'''

# 数据清洗
# 使用平均数填补 Item_Weight
#print('NaN number Before:', data['Item_Weight'].isnull().sum()) #NaN number
Item_Weight_mean = data['Item_Weight'].mean() #计算'Item_Weight'平均值
#print('Item_Weight_mean = ', Item_Weight_mean)
data['Item_Weight'] = data['Item_Weight'].fillna(Item_Weight_mean) #用'Item_Weight'平均值填充缺失值
#print('NaN number After:', data['Item_Weight'].isnull().sum()) #NaN number

# 使用出现次数最多的值填补 Outlet_Size
#print('NaN number Before:', data['Outlet_Size'].isnull().sum()) #NaN number
utlet_Size_mode = data['Outlet_Size'].mode() #获取'Outlet_Size'众数
#print('Mode = ', utlet_Size_mode[0])
data['Outlet_Size'] = data['Outlet_Size'].fillna(utlet_Size_mode[0]) #用'Outlet_Size'出现最多的值填充缺失值
#print('NaN number After:', data['Outlet_Size'].isnull().sum()) #NaN number

# Draw picture and 查找偏离值
'''
#Plot Item_Weight
x = list(range(0, len(data['Item_Weight'])))
y_Item_Weight = data['Item_Weight']
plt.scatter(x, y_Item_Weight, s=2)
#设置标题并加上轴标签
plt.title("Item_Weight", fontsize=24)
#设置刻度标记的大小
plt.tick_params(axis='both', which='major', labelsize=14)
#设置每个坐标的取值范围
plt.axis([0, len(data['Item_Weight']), min(data['Item_Weight']), max(data['Item_Weight'])])
plt.show()

#Plot Item_Visibility
x = list(range(0, len(data['Item_Visibility'])))
y_Item_Visibility = data['Item_Visibility']
plt.scatter(x, y_Item_Visibility, s=2)
#设置标题并加上轴标签
plt.title("Item_Visibility", fontsize=24)
#设置刻度标记的大小
plt.tick_params(axis='both', which='major', labelsize=14)
#设置每个坐标的取值范围
plt.axis([0, len(data['Item_Visibility']), min(data['Item_Visibility']), max(data['Item_Visibility'])])
plt.show()

#Plot Item_MRP
x = list(range(0, len(data['Item_MRP'])))
y_Item_MRP = data['Item_MRP']
plt.scatter(x, y_Item_MRP, s=2)
#设置标题并加上轴标签
plt.title("Item_MRP", fontsize=24)
    #设置刻度标记的大小
plt.tick_params(axis='both', which='both', labelsize=14)
#设置每个坐标的取值范围
plt.axis([0, len(data['Item_MRP']), min(data['Item_MRP']), max(data['Item_MRP'])])
plt.psd(y_Item_MRP, 10, 10) #Draw Grid
plt.show()

#Plot Outlet_Establishment_Year
x = list(range(0, len(data['Outlet_Establishment_Year'])))
y_Outlet_Establishment_Year = data['Outlet_Establishment_Year']
plt.scatter(x, y_Outlet_Establishment_Year, s=2)
#设置标题并加上轴标签
plt.title("Outlet_Establishment_Year", fontsize=24)
    #设置刻度标记的大小
plt.tick_params(axis='both', which='both', labelsize=14)
#设置每个坐标的取值范围
plt.axis([0, len(data['Outlet_Establishment_Year']), min(data['Outlet_Establishment_Year']), max(data['Outlet_Establishment_Year'])])
plt.show()
'''

# Data Transform

# Item_Fat_Content
#print(data['Item_Fat_Content'].value_counts())
Item_Fat_Content_mapping = {'Regular':int(1), 'reg':int(1), 'Low Fat':int(2), 'LF':int(2), 'low fat':int(2)}
data['Item_Fat_Content'] = data['Item_Fat_Content'].map(Item_Fat_Content_mapping)
data['Item_Fat_Content'] = data['Item_Fat_Content'].fillna(0)

# Item_Type
#print(data['Item_Type'].value_counts())
Item_Type_mapping = {'Seafood':int(1),
                     'Breakfast':int(2),
                     'Starchy Foods':int(3),
                     'Others':int(4),
                     'Hard Drinks':int(5),
                     'Breads':int(6),
                     'Meat':int(7),
                     'Soft Drinks':int(8),
                     'Health and Hygiene':int(9),
                     'Baking Goods':int(10),
                     'Canned':int(11),
                     'Dairy':int(12),
                     'Frozen Foods':int(13),
                     'Household':int(14),
                     'Snack Foods':int(15),
                     'Fruits and Vegetables':int(16)}
data['Item_Type'] = data['Item_Type'].map(Item_Type_mapping)
data['Item_Type'] = data['Item_Type'].fillna(0)

#Outlet_Identifier
#print(data['Outlet_Identifier'].value_counts())
#自定义函数清理数据'OUT'
def convert_currency(var):
    new_value = var.replace('OUT', '')
    #new_value = var[3:6]
    return int(new_value)

data['Outlet_Identifier'] = data['Outlet_Identifier'].apply(convert_currency)

#'Outlet_Size'
#打印'Outlet_Size'数据情况
#print(data['Outlet_Size'].value_counts())
Outlet_Size_mapping = {'Small':int(1), 'Medium':int(2), 'High':int(3)}
data['Outlet_Size'] = data['Outlet_Size'].map(Outlet_Size_mapping)
data['Outlet_Size'] = data['Outlet_Size'].fillna(0)

# 商店所在的城市类型'Outlet_Location_Type'
#print(data['Outlet_Location_Type'].value_counts())
Outlet_Location_Type_mapping = {'Tier 1':int(1), 'Tier 2':int(2), 'Tier 3':int(3)}
data['Outlet_Location_Type'] = data['Outlet_Location_Type'].map(Outlet_Location_Type_mapping)
data['Outlet_Location_Type'] = data['Outlet_Location_Type'].fillna(0)

# 出口是一家杂货店还是某种超市 'Outlet_Type'
#print(data['Outlet_Type'].value_counts())
# Outlet_Type
Outlet_Type_mapping = {'Supermarket Type1':int(1), 'Supermarket Type2':int(2), 'Supermarket Type3':int(3), 'Grocery Store':int(4)}
data['Outlet_Type'] = data['Outlet_Type'].map(Outlet_Type_mapping)
data['Outlet_Type'] = data['Outlet_Type'].fillna(0)

# 数据归一化
## 产品重量'Item_Weight'
Item_Weight_min = data['Item_Weight'].min()
Item_Weight_max = data['Item_Weight'].max()
data['Item_Weight'] = (data['Item_Weight'] - Item_Weight_min) / (Item_Weight_max - Item_Weight_min)

## 分配给特定产品的商店中所有产品的总显示区域的百分比'Item_Visibility'
Item_Visibility_min = data['Item_Visibility'].min()
Item_Visibility_max = data['Item_Visibility'].max()
data['Item_Visibility'] = (data['Item_Visibility'] - Item_Visibility_min) / (Item_Visibility_max - Item_Visibility_min)

## 产品的最大零售价 'Item_MRP'
Item_MRP_min = data['Item_MRP'].min()
Item_MRP_max = data['Item_MRP'].max()
data['Item_MRP'] = (data['Item_MRP'] - Item_MRP_min) / (Item_MRP_max - Item_MRP_min)

## 商店成立的年份 'Outlet_Establishment_Year'
Outlet_Establishment_Year_min = data['Outlet_Establishment_Year'].min()
Outlet_Establishment_Year_max = data['Outlet_Establishment_Year'].max()
data['Outlet_Establishment_Year'] = (data['Outlet_Establishment_Year'] - Outlet_Establishment_Year_min) / (Outlet_Establishment_Year_max - Outlet_Establishment_Year_min)

## 商店的面积覆盖面积 'Outlet_Size'
Outlet_Size_min = data['Outlet_Size'].min()
Outlet_Size_max = data['Outlet_Size'].max()
data['Outlet_Size'] = (data['Outlet_Size'] - Outlet_Size_min) / (Outlet_Size_max - Outlet_Size_min)







# 输出处理后的data到'Output_Train_UWu5bXk.csv'目录
data.to_csv('Output_Train_UWu5bXk.csv') # Train_UWu5bXk.csv
#data.to_csv('Output_Test_u94Q5KV.csv') # Test_u94Q5KV.csv