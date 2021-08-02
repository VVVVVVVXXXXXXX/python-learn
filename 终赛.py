#数据的导入与清洗

import pandas as pd
import numpy as np
from numpy import inf
import matplotlib.pyplot as plt
from pandas import DataFrame,Series
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier


#变形
#数据导入
path_return=r'C:\Users\White\Desktop\fn\涨跌幅 (1).csv'
data_return= pd.read_csv(path_return,index_col=0)
lrdfh = data_return
lrdf = np.delete(lrdfh.values,(0,),axis = 0)
lrdfm = pd.DataFrame(lrdf)

        


#因子导入
path_shizhi= r'C:\Users\White\Desktop\fn\流通市值 (1).csv'
shizhi1 = pd.read_csv(path_shizhi,index_col = 0)


path_roa = r'C:\Users\White\Desktop\fn\净资产收益率（加权） (1).csv'
roa1 = pd.read_csv(path_roa,index_col = 0)

path_nar = r'C:\Users\White\Desktop\fn\每股净资产同比增长率 (1).csv'
nar1 = pd.read_csv(path_nar,index_col = 0)
#$%每股收益
path_rps = r'C:\Users\White\Desktop\fn\每股收益（摊薄） (1).csv'
rps1 = pd.read_csv(path_rps,index_col= 0 )
#$%市盈率
path_pe = r'C:\Users\White\Desktop\fn\市盈率 (1).csv'
pe1 = pd.read_csv(path_pe,index_col= 0 )

#$%营业收入同比增长率
path_rrr = r'C:\Users\White\Desktop\fn\营业收入同比增长率 (1).csv'
rrr1 = pd.read_csv(path_rrr, index_col=0) 

shizhi= shizhi1.fillna(0)
roa = roa1.fillna(0)
nar = nar1.fillna(0)
rps = rps1.fillna(0)
pe = pe1.fillna(0)
rrr = rrr1.fillna(0)

lrdf2 = lrdfm.fillna(0)
#inf 正负无穷设为nan
zero = np.zeros(1217)


        


#空值设为0



for i in range(1217):
    ro = np.array(roa.iloc[:,i])
    na = np.array(nar.iloc[:,i])
    rp = np.array(rps.iloc[:,i])
    pe2 =np.array(pe.iloc[:,i])
    rr = np.array(rrr.iloc[:,i]) 
    sz = np.array(shizhi.iloc[:,i])
    lr = np.array(lrdf2.iloc[:,i])
    
    X = np.dstack(([ro],[pe2],[rr],[sz]))
    X2 = X.reshape(1217,4)
    X2[X2 == np.inf] = 0
    Xh = np.delete(X2,(1216,),axis= 0)
    X_train,X_test,y_train,y_test = \
    train_test_split(Xh,lr,train_size = 1000 )
    
    model_linear = LinearRegression()
    model_linear.fit(X_train,y_train)
    m = model_linear.score(X_test, y_test )
    XP = np.delete(X2,(0,1215),axis = 0)
    Pre = model_linear.predict(XP)


#多重共线性检验
X3 = pd.DataFrame(X2)
from statsmodels.stats.outliers_influence import variance_inflation_factor

#IF临界10
vif = [variance_inflation_factor(X3.values,X3.columns.get_loc(i))for i in X3.columns]
print(vif)

tol = [1./variance_inflation_factor(X3.values, X3.columns.get_loc(i)) for i in X3.columns]
print(tol)






