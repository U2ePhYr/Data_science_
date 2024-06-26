# ts analysis
import pandas as pd
import statsmodels.api as sm


"""处理缺失值""" 
# 假设df是一个DataFrame，其中有一些缺失值
df = pd.DataFrame({"value": [1, None, 2, 3, None, 4]})

# 使用线性插值填充缺失值
df = df.interpolate(method='linear')

"""季节性调整"""
# 假设df是一个DataFrame，其中包含时间序列数据
df = pd.DataFrame({"value": [1.2,2.6,3,4,5,6.5,7,8,9.9,10,11,12]})
 
# 进行季节性分解
res = sm.tsa.seasonal_decompose(df.value, model='additive', period=3)
 
# 季节性调整
df_deseasonalized = df.value - res.seasonal
# 普通字符串
"""去噪"""
# 假设df是一个DataFrame，其中包含时间序列数据
df = pd.DataFrame({"value": [1,8,9,4,5,61,7,18,90,10,121,112]})
df_copy = df.copy()
 
# 使用移动平均方法去噪
df_smooth = df.value.rolling(window=3).mean().round(2)
df_new = pd.concat([df_copy,df_smooth],axis=1)

'''检查并确保平稳性'''
# 假设df是一个DataFrame，其中包含时间序列数据
df = pd.DataFrame({"value": [1,2,3,4,5,6,7,8,9,10,11,12]})
 
# 进行Augmented Dickey-Fuller test
result = adfuller(df.value)
 
# 输出测试结果
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])





