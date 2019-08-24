
# coding: utf-8

# ## 一、数据预处理
# ### 常用的预处理方法有哪些？
#   python预处理
#   数据合并（堆叠合并、主键合并、重叠合并）
#   数据清理 （重复值处理、缺失值处理、异常值处理)
#   数据标准化(离差标准化、标准差标准化）
#   数据转换（等宽法、等频法）
#  
# ### 对附件1中的数据进行了哪些预处理/
# 1.对数据进行去重，对每一行的订单号进行去重，没有重复行，对每一列进行特征去重，实际金额和应付金额相似度非常高可以选择去除其中一列
# 2.检测缺失值，无缺失值
# 3.检测异常值，有检测到C售货机的一个异常日期，对它进行了删除操作
# 4.按照每一台售货机进行了分类，分别存到了test_A、test_B、test_C、test_D、test_E五个表格中，为后面统计性分析做准备
# 

# In[199]:


#读取原始数据
import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
all1=pd.read_csv(open(r'C:\Users\admin\Desktop\autoshj\1.csv'), index_col=None)
all1


# ### 1.数据合并

# In[337]:


##将附件1和附件2的数据合并，将有分类的商品信息存到S，以便后面提取数据
all1=pd.read_csv(open(r'C:\Users\admin\Desktop\autoshj\1.csv'))
all2=pd.read_csv(open(r'C:\Users\admin\Desktop\autoshj\2.csv'))
s=pd.merge(all1,all2,left_on='商品',right_on ='商品',how = 'inner')
s
#s.to_csv(r'C:\Users\admin\Desktop\autoshj\S.csv')


# ### 2.数据清理

# In[338]:


a=pd.read_csv(open(r'C:\Users\admin\Desktop\autoshj\1.csv'))
#记录去重
##定义去重函数
def delRep(list1):
    list2=[]
    for i in list1:
        if i not in list2:
            list2.append(i)
    return list2 
## 去重
orders=list(a['订单号']) ##将dishes_name从数据框中提取出来
print('去重前订单总数为：',len(orders)) 
order = delRep(orders) ##使用自定义的去重函数去重
print('去重后订单总数为：',len(order))


# #### 由上面的结果可以知道数据不存在相同记录

# In[211]:


#特征去重 两个特征相似度高 可选择性的进行调用
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = 'SimHei'## 设置中文显示
plt.rcParams['axes.unicode_minus'] = False
dfa=a[['应付金额','实际金额']].corr(method='kendall')
print('两者的kendall相似度为：\n',dfa)
a[['应付金额','实际金额']].plot.scatter(x='应付金额',y='实际金额')


# In[ ]:


##识别异常值并进行处理 有一条记录日期异常进行删除操作
#original=pd.to_datetime(l['支付时间'], format='%Y/%m/%d %H:%M')#转换为日期,出现问题，把日期异常的列删掉
a[a['支付时间']=='2017/2/29  3:44:00 PM'].index
a.drop([70679],inplace=True)


# In[213]:


##缺失值 无缺失值
a.isnull().sum()


# ## 2.计算每台售货机2017年5月份的销售额和订单量

# In[58]:


##将清洗完之后的数据进行划分
A=a[a[u'地点']=='A']
#A.to_csv(r'C:\Users\admin\Desktop\autoshj\test_A.csv')
B=a[a[u'地点']=='B']
#B.to_csv(r'C:\Users\admin\Desktop\autoshj\test_B.csv')
C=a[a[u'地点']=='C']
#C.to_csv(r'C:\Users\admin\Desktop\autoshj\test_C.csv')
D=a[a[u'地点']=='D']
#D.to_csv(r'C:\Users\admin\Desktop\autoshj\test_D.csv')
E=a[a[u'地点']=='E']
#E.to_csv(r'C:\Users\admin\Desktop\autoshj\test_E.csv')


# In[216]:


##统计各台售货机的订单数量
a[u'地点'].value_counts()


# In[ ]:


import datetime
A['支付时间']= pd.to_datetime(A['支付时间'], format='%Y/%m/%d %H:%M')#转换为日期
A['YM']=A['支付时间'].apply(lambda x:datetime.datetime.strftime(x,'%Y-%m'))#日期转换为字符串
B['支付时间']= pd.to_datetime(B['支付时间'], format='%Y/%m/%d %H:%M')#转换为日期
B['YM']=B['支付时间'].apply(lambda x:datetime.datetime.strftime(x,'%Y-%m'))#日期转换为字符串
C['支付时间']= pd.to_datetime(C['支付时间'], format='%Y/%m/%d %H:%M')#转换为日期
C['YM']=C['支付时间'].apply(lambda x:datetime.datetime.strftime(x,'%Y-%m'))#日期转换为字符串
D['支付时间']= pd.to_datetime(D['支付时间'], format='%Y/%m/%d %H:%M')#转换为日期
D['YM']=D['支付时间'].apply(lambda x:datetime.datetime.strftime(x,'%Y-%m'))#日期转换为字符串
E['支付时间']= pd.to_datetime(E['支付时间'], format='%Y/%m/%d %H:%M')#转换为日期
E['YM']=E['支付时间'].apply(lambda x:datetime.datetime.strftime(x,'%Y-%m'))#日期转换为字符串
print(A[A['YM']=='2017-05']['实际金额'].sum())
print(B[B['YM']=='2017-05']['实际金额'].sum())
print(C[C['YM']=='2017-05']['实际金额'].sum())
print(D[D['YM']=='2017-05']['实际金额'].sum())
print(E[E['YM']=='2017-05']['实际金额'].sum())
print(A[A['YM']=='2017-05']['订单号'].count())
print(B[B['YM']=='2017-05']['订单号'].count())
print(C[C['YM']=='2017-05']['订单号'].count())
print(D[D['YM']=='2017-05']['订单号'].count())
print(E[E['YM']=='2017-05']['订单号'].count())


# In[ ]:


print(A[A['YM']=='2017-05']['实际金额'].value_counts())


# In[238]:


ser1=pd.Series([3385.1,3681.2,3729.4,2392.1,5699.0])
ser2=pd.Series([756,869,789,564,1292])
data={'销售额(元)':[3385.1,3681.2,3729.4,2392.1,5699.0,ser1.sum()],'订单量':[756,869,789,564,1292,ser2.sum()]}
MAY=pd.DataFrame(data,index=['A','B','C','D','E','Total'],columns=['销售额(元)','订单量'])
MAY.to_csv(r'C:\Users\admin\Desktop\autoshj\MAY.csv')


# In[43]:


import pandas as pd
import datetime
A=pd.read_csv(r'C:\Users\admin\Desktop\autoshj\test_A.csv')
A['支付时间']= pd.to_datetime(A['支付时间'], format='%Y/%m/%d %H:%M')#转换为日期

every=A['支付时间'].dt.month.value_counts()#统计每个月的总的订单量
every
A['订单号'].groupby(by=A['支付时间'].dt.month).count()#统计每个月的订单数量


# In[136]:


#计算每台售货机的交易总额和总订单量
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import calendar
import datetime
A= pd.read_csv(r'C:\Users\admin\Desktop\autoshj\test_A.csv')
B= pd.read_csv(r'C:\Users\admin\Desktop\autoshj\test_B.csv')
C= pd.read_csv(r'C:\Users\admin\Desktop\autoshj\test_C.csv')
D= pd.read_csv(r'C:\Users\admin\Desktop\autoshj\test_D.csv')
E= pd.read_csv(r'C:\Users\admin\Desktop\autoshj\test_E.csv')

ser1=pd.Series([A['实际金额'].sum(),B['实际金额'].sum(),C['实际金额'].sum(),D['实际金额'].sum(),E['实际金额'].sum()])
ser2=pd.Series([A['订单号'].count(),B['订单号'].count(),C['订单号'].count(),D['订单号'].count(),E['订单号'].count()])

data={'交易额':[A['实际金额'].sum(),B['实际金额'].sum(),C['实际金额'].sum(),D['实际金额'].sum(),E['实际金额'].sum(),ser1.sum()],
      '订单量':[A['订单号'].count(),B['订单号'].count(),C['订单号'].count(),D['订单号'].count(),E['订单号'].count(),ser2.sum()]}
Total=pd.DataFrame(data,index=['A','B','C','D','E','Total'],columns=['交易额','订单量'])
Total.to_csv(r'C:\Users\admin\Desktop\autoshj\Total.csv')
Total


# In[112]:


#每台售货机每月的每单平均交易额和日均订单量
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import calendar
import datetime
A= pd.read_csv(r'C:\Users\admin\Desktop\autoshj\test_A.csv')
B= pd.read_csv(r'C:\Users\admin\Desktop\autoshj\test_B.csv')
C= pd.read_csv(r'C:\Users\admin\Desktop\autoshj\test_C.csv')
D= pd.read_csv(r'C:\Users\admin\Desktop\autoshj\test_D.csv')
E= pd.read_csv(r'C:\Users\admin\Desktop\autoshj\test_E.csv')


# In[330]:


#每台售货机每月的每单平均交易额和日均订单量
#计算每月日均订单量
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import calendar
import datetime
A= pd.read_csv(r'C:\Users\admin\Desktop\autoshj\test_A.csv')
B= pd.read_csv(r'C:\Users\admin\Desktop\autoshj\test_B.csv')
C= pd.read_csv(r'C:\Users\admin\Desktop\autoshj\test_C.csv')
D= pd.read_csv(r'C:\Users\admin\Desktop\autoshj\test_D.csv')
E= pd.read_csv(r'C:\Users\admin\Desktop\autoshj\test_E.csv')
A['支付时间']= pd.to_datetime(A['支付时间'], format='%Y/%m/%d %H:%M')#转换为日期
B['支付时间']= pd.to_datetime(B['支付时间'], format='%Y/%m/%d %H:%M')#转换为日期
C['支付时间']= pd.to_datetime(C['支付时间'], format='%Y/%m/%d %H:%M')#转换为日期
D['支付时间']= pd.to_datetime(D['支付时间'], format='%Y/%m/%d %H:%M')#转换为日期
E['支付时间']= pd.to_datetime(E['支付时间'], format='%Y/%m/%d %H:%M')#转换为日期
###A
#计算每月每单平均交易额
def mdpjjye(month):
    s=sum(A['实际金额'][A['支付时间'].dt.month==month])/A['订单号'][A['支付时间'].dt.month==month].count()
    return round(s,2)
#计算每月日均订单量
def myrjddl(month):
    import calendar
    d=A['订单号'][A['支付时间'].dt.month==month].count()/calendar.monthrange(2017,month)[1]
    return round(d)
data={'每月日均订单量':[myrjddl(1),myrjddl(2),myrjddl(3),myrjddl(4),myrjddl(5),myrjddl(6),myrjddl(7),myrjddl(8),myrjddl(9),myrjddl(10),myrjddl(11),myrjddl(12)],
      '每月每单平均交易额':[mdpjjye(1),mdpjjye(2),mdpjjye(3),mdpjjye(4),mdpjjye(5),mdpjjye(6),mdpjjye(7),mdpjjye(8),mdpjjye(9),mdpjjye(10),mdpjjye(11),mdpjjye(12)]}
order_A=pd.DataFrame(data,index=['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月'],columns=['每月每单平均交易额','每月日均订单量'])

print(order_A)

###B
#计算每月每单平均交易额
def mdpjjye(month):
    s=sum(B['实际金额'][B['支付时间'].dt.month==month])/B['订单号'][B['支付时间'].dt.month==month].count()
    return round(s,2)
#计算每月日均订单量
def myrjddl(month):
    import calendar
    d=B['订单号'][B['支付时间'].dt.month==month].count()/calendar.monthrange(2017,month)[1]
    return round(d)
data={'每月日均订单量':[myrjddl(1),myrjddl(2),myrjddl(3),myrjddl(4),myrjddl(5),myrjddl(6),myrjddl(7),myrjddl(8),myrjddl(9),myrjddl(10),myrjddl(11),myrjddl(12)],
      '每月每单平均交易额':[mdpjjye(1),mdpjjye(2),mdpjjye(3),mdpjjye(4),mdpjjye(5),mdpjjye(6),mdpjjye(7),mdpjjye(8),mdpjjye(9),mdpjjye(10),mdpjjye(11),mdpjjye(12)]}
order_B=pd.DataFrame(data,index=['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月'],columns=['每月每单平均交易额','每月日均订单量'])
print(order_B)

###C
#计算每月每单平均交易额
def mdpjjye(month):
    s=sum(C['实际金额'][C['支付时间'].dt.month==month])/C['订单号'][C['支付时间'].dt.month==month].count()
    return round(s,2)
#计算每月日均订单量
def myrjddl(month):
    import calendar
    d=C['订单号'][C['支付时间'].dt.month==month].count()/calendar.monthrange(2017,month)[1]
    return round(d)
data={'每月日均订单量':[myrjddl(1),myrjddl(2),myrjddl(3),myrjddl(4),myrjddl(5),myrjddl(6),myrjddl(7),myrjddl(8),myrjddl(9),myrjddl(10),myrjddl(11),myrjddl(12)],
      '每月每单平均交易额':[mdpjjye(1),mdpjjye(2),mdpjjye(3),mdpjjye(4),mdpjjye(5),mdpjjye(6),mdpjjye(7),mdpjjye(8),mdpjjye(9),mdpjjye(10),mdpjjye(11),mdpjjye(12)]}
order_C=pd.DataFrame(data,index=['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月'],columns=['每月每单平均交易额','每月日均订单量'])

print(order_C)

###D
#计算每月每单平均交易额
def mdpjjye(month):
    s=sum(D['实际金额'][D['支付时间'].dt.month==month])/D['订单号'][D['支付时间'].dt.month==month].count()
    return round(s,2)
#计算每月日均订单量
def myrjddl(month):
    import calendar
    d=D['订单号'][D['支付时间'].dt.month==month].count()/calendar.monthrange(2017,month)[1]
    return round(d)
data={'每月日均订单量':[myrjddl(1),myrjddl(2),myrjddl(3),myrjddl(4),myrjddl(5),myrjddl(6),myrjddl(7),myrjddl(8),myrjddl(9),myrjddl(10),myrjddl(11),myrjddl(12)],
      '每月每单平均交易额':[mdpjjye(1),mdpjjye(2),mdpjjye(3),mdpjjye(4),mdpjjye(5),mdpjjye(6),mdpjjye(7),mdpjjye(8),mdpjjye(9),mdpjjye(10),mdpjjye(11),mdpjjye(12)]}
order_D=pd.DataFrame(data,index=['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月'],columns=['每月每单平均交易额','每月日均订单量'])

print(order_D)

###E
#计算每月每单平均交易额
def mdpjjye(month):
    s=sum(E['实际金额'][E['支付时间'].dt.month==month])/E['订单号'][E['支付时间'].dt.month==month].count()
    return round(s,2)
#计算每月日均订单量
def myrjddl(month):
    import calendar
    d=E['订单号'][E['支付时间'].dt.month==month].count()/calendar.monthrange(2017,month)[1]
    return round(d)
data={'每月日均订单量':[myrjddl(1),myrjddl(2),myrjddl(3),myrjddl(4),myrjddl(5),myrjddl(6),myrjddl(7),myrjddl(8),myrjddl(9),myrjddl(10),myrjddl(11),myrjddl(12)],
      '每月每单平均交易额':[mdpjjye(1),mdpjjye(2),mdpjjye(3),mdpjjye(4),mdpjjye(5),mdpjjye(6),mdpjjye(7),mdpjjye(8),mdpjjye(9),mdpjjye(10),mdpjjye(11),mdpjjye(12)]}
order_E=pd.DataFrame(data,index=['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月'],columns=['每月每单平均交易额','每月日均订单量'])

print(order_E)

order_A.to_csv(r'C:\Users\admin\Desktop\autoshj\order_A.csv')
order_B.to_csv(r'C:\Users\admin\Desktop\autoshj\order_B.csv')
order_C.to_csv(r'C:\Users\admin\Desktop\autoshj\order_C.csv')
order_D.to_csv(r'C:\Users\admin\Desktop\autoshj\order_D.csv')
order_E.to_csv(r'C:\Users\admin\Desktop\autoshj\order_E.csv')


# In[328]:


import matplotlib.pylab as plt
order_A.plot()
plt.rcParams['font.sans-serif'] = 'SimHei'## 设置字体为SimHei显示中文
plt.rcParams['axes.unicode_minus'] = False ## 设置正常显示符号

plt.show()


# In[36]:


#任务二 可视化  2017年6月销量前五的商品柱状图
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import datetime
plt.style.use('bmh')
plt.rcParams['font.sans-serif'] = 'SimHei'## 设置字体为SimHei显示中文
plt.rcParams['axes.unicode_minus'] = False ## 设置正常显示符号

all1= pd.read_csv(open(r'C:\Users\admin\Desktop\autoshj\1.csv'))
all1['支付时间']= pd.to_datetime(all1['支付时间'], format='%Y/%m/%d %H:%M')#转换为日期
all1['YM']=all1['支付时间'].apply(lambda x:datetime.datetime.strftime(x,'%Y-%m'))#日期转换为字符串
p=all1[all1['YM']=='2017-06']['商品'].value_counts().head(5)
plt.bar(x=p.index,height=p)
plt.title('2017年6月销售前5商品销量柱状图')
plt.xlabel('商品')## 添加横轴名称
plt.ylabel('月销量')## 添加y轴名称
plt.savefig('2017.6销售前五柱状图.png')
plt.show()


# In[305]:


A= pd.read_csv(r'C:\Users\admin\Desktop\autoshj\test_A.csv')
B= pd.read_csv(r'C:\Users\admin\Desktop\autoshj\test_B.csv')
C= pd.read_csv(r'C:\Users\admin\Desktop\autoshj\test_C.csv')
D= pd.read_csv(r'C:\Users\admin\Desktop\autoshj\test_D.csv')
E= pd.read_csv(r'C:\Users\admin\Desktop\autoshj\test_E.csv')
A['支付时间']= pd.to_datetime(A['支付时间'], format='%Y/%m/%d %H:%M')#转换为日期
B['支付时间']= pd.to_datetime(B['支付时间'], format='%Y/%m/%d %H:%M')#转换为日期
C['支付时间']= pd.to_datetime(C['支付时间'], format='%Y/%m/%d %H:%M')#转换为日期
D['支付时间']= pd.to_datetime(D['支付时间'], format='%Y/%m/%d %H:%M')#转换为日期
E['支付时间']= pd.to_datetime(E['支付时间'], format='%Y/%m/%d %H:%M')#转换为日期
#s=sum(A['实际金额']).groupby(by=A['支付时间'].dt.month)
#计算每月每单平均交易额
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import datetime
plt.style.use('ggplot')

plt.rcParams['font.sans-serif'] = 'SimHei'## 设置字体为SimHei显示中文
plt.rcParams['axes.unicode_minus'] = False ## 设置正常显示符号
def myjye(month):
    s=sum(A['实际金额'][A['支付时间'].dt.month==month])
    return round(s,3)

def hbzzl(month):
    b=(myjye(month)-myjye(month-1))/myjye(month-1)
    return(b)
    
data={'每月交易额':[myjye(1),myjye(2),myjye(3),myjye(4),myjye(5),myjye(6),myjye(7),myjye(8),myjye(9),myjye(10),myjye(11),myjye(12)], '每月环比增长率':[0,hbzzl(2),hbzzl(3),hbzzl(4),hbzzl(5),hbzzl(6),hbzzl(7),hbzzl(8),hbzzl(9),hbzzl(10),hbzzl(11),hbzzl(12)]}
sa=pd.DataFrame(data,index=['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月'],columns=['每月交易额','每月环比增长率'])    
plt.rcParams['figure.figsize'] = (10.0, 5.0)
# 第一张图（subplot（m，n，p） m 代表行，n 代表列，p 代表的这个图形画在第几行、第几列）

plt.plot(sa['每月交易额'])
plt.title('A售货机每月销量折线图')
plt.xlabel('月份')## 添加横轴名称
plt.ylabel('月销量')## 添加y轴名称
plt.savefig('A售货机每月销量折线图.png')
plt.show()

plt.bar(sa.index,sa['每月环比增长率'],edgecolor='white')
plt.title('A售货机环比增长率柱状图')
plt.xlabel('月份')## 添加横轴名称
plt.ylabel('环比增长率')## 添加y轴名称
plt.savefig('A售货机环比增长率柱状图.png')
plt.show()


# In[201]:


def myjye(month):
    s=sum(B['实际金额'][B['支付时间'].dt.month==month])
    return round(s,3)

def hbzzl(month):
    b=(myjye(month)-myjye(month-1))/myjye(month-1)
    return(b)
    
data={'每月交易额':[myjye(1),myjye(2),myjye(3),myjye(4),myjye(5),myjye(6),myjye(7),myjye(8),myjye(9),myjye(10),myjye(11),myjye(12)], '每月环比增长率':[0,hbzzl(2),hbzzl(3),hbzzl(4),hbzzl(5),hbzzl(6),hbzzl(7),hbzzl(8),hbzzl(9),hbzzl(10),hbzzl(11),hbzzl(12)]}
sa=pd.DataFrame(data,index=['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月'],columns=['每月交易额','每月环比增长率'])    
plt.rcParams['figure.figsize'] = (10.0, 5.0)
# 第一张图（subplot（m，n，p） m 代表行，n 代表列，p 代表的这个图形画在第几行、第几列）

plt.plot(sa['每月交易额'])
plt.title('B售货机每月销量折线图')
plt.xlabel('月份')## 添加横轴名称
plt.ylabel('月销量')## 添加y轴名称
plt.savefig('B售货机每月销量折线图.png')
plt.show()

plt.bar(sa.index,sa['每月环比增长率'],edgecolor='white')
plt.title('B售货机环比增长率柱状图')
plt.xlabel('月份')## 添加横轴名称
plt.ylabel('环比增长率')## 添加y轴名称
plt.savefig('B售货机环比增长率柱状图.png')
plt.show()


# In[202]:


def myjye(month):
    s=sum(C['实际金额'][C['支付时间'].dt.month==month])
    return round(s,3)

def hbzzl(month):
    b=(myjye(month)-myjye(month-1))/myjye(month-1)
    return(b)
    
data={'每月交易额':[myjye(1),myjye(2),myjye(3),myjye(4),myjye(5),myjye(6),myjye(7),myjye(8),myjye(9),myjye(10),myjye(11),myjye(12)], '每月环比增长率':[0,hbzzl(2),hbzzl(3),hbzzl(4),hbzzl(5),hbzzl(6),hbzzl(7),hbzzl(8),hbzzl(9),hbzzl(10),hbzzl(11),hbzzl(12)]}
sa=pd.DataFrame(data,index=['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月'],columns=['每月交易额','每月环比增长率'])    
plt.rcParams['figure.figsize'] = (10.0, 5.0)
# 第一张图（subplot（m，n，p） m 代表行，n 代表列，p 代表的这个图形画在第几行、第几列）

plt.plot(sa['每月交易额'])
plt.title('C售货机每月销量折线图')
plt.xlabel('月份')## 添加横轴名称
plt.ylabel('月销量')## 添加y轴名称
plt.savefig('C售货机每月销量折线图.png')
plt.show()

plt.bar(sa.index,sa['每月环比增长率'],edgecolor='white')
plt.title('C售货机环比增长率柱状图')
plt.xlabel('月份')## 添加横轴名称
plt.ylabel('环比增长率')## 添加y轴名称
plt.savefig('C售货机环比增长率柱状图.png')
plt.show()


# In[203]:


def myjye(month):
    s=sum(D['实际金额'][D['支付时间'].dt.month==month])
    return round(s,3)

def hbzzl(month):
    b=(myjye(month)-myjye(month-1))/myjye(month-1)
    return(b)
    
data={'每月交易额':[myjye(1),myjye(2),myjye(3),myjye(4),myjye(5),myjye(6),myjye(7),myjye(8),myjye(9),myjye(10),myjye(11),myjye(12)], '每月环比增长率':[0,hbzzl(2),hbzzl(3),hbzzl(4),hbzzl(5),hbzzl(6),hbzzl(7),hbzzl(8),hbzzl(9),hbzzl(10),hbzzl(11),hbzzl(12)]}
sa=pd.DataFrame(data,index=['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月'],columns=['每月交易额','每月环比增长率'])    
plt.rcParams['figure.figsize'] = (10.0, 5.0)
# 第一张图（subplot（m，n，p） m 代表行，n 代表列，p 代表的这个图形画在第几行、第几列）

plt.plot(sa['每月交易额'])
plt.title('D售货机每月销量折线图')
plt.xlabel('月份')## 添加横轴名称
plt.ylabel('月销量')## 添加y轴名称
plt.savefig('D售货机每月销量折线图.png')
plt.show()

plt.bar(sa.index,sa['每月环比增长率'],edgecolor='white')
plt.title('D售货机环比增长率柱状图')
plt.xlabel('月份')## 添加横轴名称
plt.ylabel('环比增长率')## 添加y轴名称
plt.savefig('D售货机环比增长率柱状图.png')
plt.show()


# In[204]:


def myjye(month):
    s=sum(E['实际金额'][E['支付时间'].dt.month==month])
    return round(s,3)

def hbzzl(month):
    b=(myjye(month)-myjye(month-1))/myjye(month-1)
    return(b)
    
data={'每月交易额':[myjye(1),myjye(2),myjye(3),myjye(4),myjye(5),myjye(6),myjye(7),myjye(8),myjye(9),myjye(10),myjye(11),myjye(12)], '每月环比增长率':[0,hbzzl(2),hbzzl(3),hbzzl(4),hbzzl(5),hbzzl(6),hbzzl(7),hbzzl(8),hbzzl(9),hbzzl(10),hbzzl(11),hbzzl(12)]}
sa=pd.DataFrame(data,index=['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月'],columns=['每月交易额','每月环比增长率'])    
plt.rcParams['figure.figsize'] = (10.0, 5.0)
# 第一张图（subplot（m，n，p） m 代表行，n 代表列，p 代表的这个图形画在第几行、第几列）

plt.plot(sa['每月交易额'])
plt.title('E售货机每月销量折线图')
plt.xlabel('月份')## 添加横轴名称
plt.ylabel('月销量')## 添加y轴名称
plt.savefig('E售货机每月销量折线图.png')
plt.show()

plt.bar(sa.index,sa['每月环比增长率'],edgecolor='white')
plt.title('E售货机环比增长率柱状图')
plt.xlabel('月份')## 添加横轴名称
plt.ylabel('环比增长率')## 添加y轴名称
plt.savefig('E售货机环比增长率柱状图.png')
plt.show()


# In[330]:


#计算每台售货机的毛利润
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
all2= pd.read_csv(open(r'C:\Users\admin\Desktop\autoshj\2.csv'))
print(all2.head(5))
all2.set_index(["商品"], inplace=True)
print(all2.index)
#将每台售货机的每一种商品的销量形成表格
sales_A=A['订单号'].groupby(by=A['商品']).count()
sales_A.to_csv(r'C:\Users\admin\Desktop\autoshj\sales_A.csv')
sales_B=B['订单号'].groupby(by=B['商品']).count()
sales_B.to_csv(r'C:\Users\admin\Desktop\autoshj\sales_B.csv')
sales_C=C['订单号'].groupby(by=C['商品']).count()
sales_C.to_csv(r'C:\Users\admin\Desktop\autoshj\sales_C.csv')
sales_D=D['订单号'].groupby(by=D['商品']).count()
sales_D.to_csv(r'C:\Users\admin\Desktop\autoshj\sales_D.csv')
sales_E=E['订单号'].groupby(by=E['商品']).count()
sales_E.to_csv(r'C:\Users\admin\Desktop\autoshj\sales_E.csv')

all2.set_index(["商品"], inplace=True)
print(all2.index)
sales_A= pd.read_csv(r'C:\Users\admin\Desktop\autoshj\sales_A.csv')
print(sales_A.head(5))
print(sales_A.index)


sale_A=pd.merge(sales_A,all2,left_on='商品',right_on ='商品',how = 'inner')
print(sale_A.head(5))

g1=sale_A['订单号'][sale_A['大类']=='饮料'].sum()
g2=sale_A['订单号'][sale_A['大类']=='非饮料'].sum()
gross_profit_A=g1*0.25+g2*0.2
print(g1,g2)
gross_profit_A
sale_A['二级类'].value_counts()
print(g1*0.25)
print(g2*0.2)


# In[329]:


sales_B= pd.read_csv(r'C:\Users\admin\Desktop\autoshj\sales_B.csv')
sale_B=pd.merge(sales_B,all2,left_on='商品',right_on ='商品',how = 'inner')
print(sale_B.head(5))

g1=sale_B['订单号'][sale_B['大类']=='饮料'].sum()
g2=sale_B['订单号'][sale_B['大类']=='非饮料'].sum()
gross_profit_B=g1*0.25+g2*0.2
print(g1,g2)
gross_profit_B
sale_B['二级类'].value_counts()
print(g1*0.25)
print(g2*0.2)


# In[331]:


sales_C= pd.read_csv(r'C:\Users\admin\Desktop\autoshj\sales_C.csv')
sale_C=pd.merge(sales_C,all2,left_on='商品',right_on ='商品',how = 'inner')
print(sale_C.head(5))

g1=sale_C['订单号'][sale_C['大类']=='饮料'].sum()
g2=sale_C['订单号'][sale_C['大类']=='非饮料'].sum()
gross_profit_C=g1*0.25+g2*0.2
print(g1,g2)
gross_profit_C
print(g1*0.25)
print(g2*0.2)


# In[332]:


sales_D= pd.read_csv(r'C:\Users\admin\Desktop\autoshj\sales_D.csv')
sale_D=pd.merge(sales_D,all2,left_on='商品',right_on ='商品',how = 'inner')
print(sale_D.head(5))

g1=sale_D['订单号'][sale_B['大类']=='饮料'].sum()
g2=sale_D['订单号'][sale_B['大类']=='非饮料'].sum()
gross_profit_D=g1*0.25+g2*0.2
print(g1,g2)
gross_profit_D
print(g1*0.25)
print(g2*0.2)


# In[363]:


#sales_E= pd.read_csv(r'C:\Users\admin\Desktop\autoshj\sales_E.csv')
#sale_E=pd.merge(sales_E,all2,left_on='商品',right_on ='商品',how = 'inner')
#print(sale_E.head(5))

g1=sale_E['订单号'][sale_E['大类']=='饮料'].sum()
g2=sale_E['订单号'][sale_E['大类']=='非饮料'].sum()
gross_profit_E=g1*0.25+g2*0.2
print(g1,g2)
print(gross_profit_E)
print(g1*0.25)
print(g2*0.2)


# In[365]:


#根据上面计算的数据绘制饼状图
data=[gross_profit_A,gross_profit_B,gross_profit_C,gross_profit_D,gross_profit_E]  
fig = plt.figure(figsize=(5, 5), dpi= 80, facecolor='w', edgecolor='k')
explode=[0.01,0.01,0.01,0.01,0.05]
label1= ['A','B','C','D','E']
plt.pie(data,labels=label1,autopct='%4.1f%%',explode=explode,shadow=True,radius=0.9)
plt.title('2017年各售货机毛利率饼状图',fontsize=15)
plt.savefig('2017年各售货机毛利率饼状图.png')
plt.rcParams['font.sans-serif'] = 'SimHei'## 设置中文显示
plt.rcParams['axes.unicode_minus'] = False


# In[369]:


order_A.set_index(['Unnamed: 0'],inplace=True)
order_A.index


# In[272]:


all1=pd.read_csv(open(r'C:\Users\admin\Desktop\autoshj\1.csv'))
all2=pd.read_csv(open(r'C:\Users\admin\Desktop\autoshj\2.csv'))
s=pd.merge(all1,all2,left_on='商品',right_on ='商品',how = 'inner')
s


# In[263]:


all2=pd.read_csv(open(r'C:\Users\admin\Desktop\autoshj\2.csv'))
A1=pd.merge(A,all2,left_on='商品',right_on ='商品',how = 'inner')
A1['支付时间']=pd.to_datetime(A1['支付时间'], format='%Y/%m/%d %H:%M')#转换为日期
A1['二级类'][A1['支付时间'].dt.month==1]
import calendar
def df1(month):
    df1=A1[A1['二级类']=='肉干/豆制品/蛋'][A1['支付时间'].dt.month==month]['实际金额'].sum()/calendar.monthrange(2017,month)[1]
    return round(df1,2) 
df1(5)
#df1['实际金额'].sum()/calendar.monthrange(2017,3)[1]


# In[265]:


#绘制C6.7.8三个月的订单量的热力图，横轴为天，纵轴以小时为单位，可以得出那些结论
C['支付时间']=pd.to_datetime(C['支付时间'],format='%Y/%m/%d %H:%M')
C[C['支付时间'].dt.month==6][]


# In[370]:


A= pd.read_csv(r'C:\Users\admin\Desktop\autoshj\test_A.csv')
B= pd.read_csv(r'C:\Users\admin\Desktop\autoshj\test_B.csv')
C= pd.read_csv(r'C:\Users\admin\Desktop\autoshj\test_C.csv')
D= pd.read_csv(r'C:\Users\admin\Desktop\autoshj\test_D.csv')
E= pd.read_csv(r'C:\Users\admin\Desktop\autoshj\test_E.csv')
A['支付时间']= pd.to_datetime(A['支付时间'], format='%Y/%m/%d %H:%M')#转换为日期
B['支付时间']= pd.to_datetime(B['支付时间'], format='%Y/%m/%d %H:%M')#转换为日期
C['支付时间']= pd.to_datetime(C['支付时间'], format='%Y/%m/%d %H:%M')#转换为日期
D['支付时间']= pd.to_datetime(D['支付时间'], format='%Y/%m/%d %H:%M')#转换为日期
E['支付时间']= pd.to_datetime(E['支付时间'], format='%Y/%m/%d %H:%M')#转换为日期


# In[300]:


#绘制A12月份时期销售热力图
data= 0       
lrt12=pd.DataFrame(data,index=[23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0],                  columns=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31])  
for day in range(1,32):
    lrt12.loc[range(0,24),day]=A[A['支付时间'].dt.month==12][A['支付时间'].dt.day==day].groupby(by=A['支付时间'].dt.hour)['订单号'].count()

lrt12=lrt12.fillna(0)
lrt12
import seaborn as sns
plt.rcParams['font.sans-serif'] = 'SimHei'## 设置中文显示
plt.rcParams['axes.unicode_minus'] = False
plt.title('2017年a12月时序销售热力图',fontsize=20)## 添加图表标题
plt.rcParams['figure.figsize'] = (8.0, 6.5)
sns.heatmap(lrt12,vmax=20,vmin=0,center=16)
plt.savefig('2017年a12月时序销售热力图.png')
plt.show()


# In[379]:


#绘制D2月份时期销售热力图
data= 0       
lrt12=pd.DataFrame(data,index=[23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0],                  columns=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28])  
for day in range(1,29):
    lrt12.loc[range(0,24),day]=D[D['支付时间'].dt.month==2][D['支付时间'].dt.day==day].groupby(by=D['支付时间'].dt.hour)['订单号'].count()

lrt12=lrt12.fillna(0)
lrt12
import seaborn as sns
plt.rcParams['font.sans-serif'] = 'SimHei'## 设置中文显示
plt.rcParams['axes.unicode_minus'] = False
plt.title('2017年D2月时序销售热力图',fontsize=20)## 添加图表标题
plt.rcParams['figure.figsize'] = (8.0, 6.5)
sns.heatmap(lrt12,vmax=10,vmin=0,center=8)
plt.savefig('2017年D2月时序销售热力图.png')
plt.show()


# In[309]:


#绘制B4月份时期销售热力图
data= 0       
lrt12=pd.DataFrame(data,index=[23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0],                  columns=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])  
for day in range(1,31):
    lrt12.loc[range(0,24),day]=B[B['支付时间'].dt.month==12][B['支付时间'].dt.day==day].groupby(by=B['支付时间'].dt.hour)['订单号'].count()

lrt12=lrt12.fillna(0)
lrt12
import seaborn as sns
plt.rcParams['font.sans-serif'] = 'SimHei'## 设置中文显示
plt.rcParams['axes.unicode_minus'] = False
plt.title('2017年B4月时序销售热力图',fontsize=20)## 添加图表标题
plt.rcParams['figure.figsize'] = (8.0, 6.5)
sns.heatmap(lrt12,vmax=20,vmin=0,center=16)
plt.savefig('2017年B4月时序销售热力图.png')
plt.show()


# In[157]:


#绘制C6月份时期销售热力图
data= 0       
lrt6=pd.DataFrame(data,index=[23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0],                  columns=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])  
for day in range(1,31):
    lrt6.loc[range(0,24),day]=C[C['支付时间'].dt.month==6][C['支付时间'].dt.day==day].groupby(by=C['支付时间'].dt.hour)['订单号'].count()

lrt6=lrt6.fillna(0)
lrt6


# In[201]:


import seaborn as sns
plt.rcParams['font.sans-serif'] = 'SimHei'## 设置中文显示
plt.rcParams['axes.unicode_minus'] = False
plt.title('2017年6月时序销售热力图',fontsize=20)## 添加图表标题
plt.rcParams['figure.figsize'] = (8.0, 6.5)
sns.heatmap(lrt6,vmax=20,vmin=0,center=16)
plt.savefig('2017年6月时序销售热力图.png')
plt.show()


# In[213]:


#绘制C7月份时期销售热力图
data= 0       
lrt7=pd.DataFrame(data,index=[23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0],                  columns=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31])  
for day in range(1,32):
    lrt7.loc[range(0,24),day]=C[C['支付时间'].dt.month==7][C['支付时间'].dt.day==day].groupby(by=C['支付时间'].dt.hour)['订单号'].count()
lrt7=lrt7.fillna(0)
lrt7


# In[369]:


import seaborn as sns
plt.rcParams['font.sans-serif'] = 'SimHei'## 设置中文显示
plt.rcParams['axes.unicode_minus'] = False
plt.title('2017年7月时序销售热力图',fontsize=20)## 添加图表标题
plt.rcParams['figure.figsize'] = (8.0, 6.5)
sns.heatmap(lrt7,center=20)
plt.savefig('2017年7月时序销售热力图.png')
plt.show()


# In[371]:


#绘制C8月份时期销售热力图
data= 0       
lrt8=pd.DataFrame(data,index=[23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0],                  columns=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31])  
for day in range(1,32):
    lrt8.loc[range(0,24),day]=C[C['支付时间'].dt.month==8][C['支付时间'].dt.day==day].groupby(by=C['支付时间'].dt.hour)['订单号'].count()
lrt8=lrt8.fillna(0)
lrt8


# In[228]:


import seaborn as sns
plt.rcParams['font.sans-serif'] = 'SimHei'## 设置中文显示
plt.rcParams['axes.unicode_minus'] = False
plt.title('2017年8月时序销售热力图',fontsize=20)## 添加图表标题
plt.rcParams['figure.figsize'] = (8.0, 6.5)
sns.heatmap(lrt8,center=10)
plt.savefig('2017年8月时序销售热力图.png')
plt.show()


# In[289]:


print(sale_A['订单号'][sale_A['大类']=='饮料'].describe())

a=sale_A[sale_A['大类']=='饮料'].sort_index(by=['订单号'],ascending=False)
a['标签']=0
a.index=np.arange(1,113)
print(np.where(a['订单号']==67))
a.ix[0:28,'标签']='热销'
print(np.where(a['订单号']==19))
a.ix[29:56,'标签']='正常'

a.ix[57:,'标签']='滞销'
a
#a.to_csv(r'C:\Users\admin\Desktop\autoshj\task3-1A.csv')


# In[32]:


print(sale_B['订单号'][sale_B['大类']=='饮料'].describe())

b=sale_B[sale_B['大类']=='饮料'].sort_index(by=['订单号'],ascending=False)
b['标签']=0
b.index=np.arange(1,116)
print(np.where(b['订单号']==94))
b.ix[0:29,'标签']='热销'
print(np.where(b['订单号']==17))
b.ix[30:58,'标签']='正常'

b.ix[59:,'标签']='滞销'
b
b.to_csv(r'C:\Users\admin\Desktop\autoshj\task3-1B.csv')


# In[382]:


print(sale_C['订单号'][sale_C['大类']=='饮料'].describe())

c=sale_C[sale_C['大类']=='饮料'].sort_index(by=['订单号'],ascending=False)
c['标签']=0
c.index=np.arange(1,115)
print(np.where(c['订单号']==79))
c.ix[0:29,'标签']='热销'
print(np.where(c['订单号']==24))
c.ix[30:57,'标签']='正常'

c.ix[58:,'标签']='滞销'
c
c.to_csv(r'C:\Users\admin\Desktop\autoshj\task3-1C.csv')


# In[38]:


print(sale_D['订单号'][sale_D['大类']=='饮料'].describe())

d=sale_D[sale_D['大类']=='饮料'].sort_index(by=['订单号'],ascending=False)
d['标签']=0
d.index=np.arange(1,106)
print(np.where(d['订单号']==64))
d.ix[0:26,'标签']='热销'
print(np.where(d['订单号']==15))
d.ix[27:52,'标签']='正常'

d.ix[53:,'标签']='滞销'
d
d.to_csv(r'C:\Users\admin\Desktop\autoshj\task3-1D.csv')


# In[41]:


print(sale_E['订单号'][sale_E['大类']=='饮料'].describe())

e=sale_E[sale_E['大类']=='饮料'].sort_index(by=['订单号'],ascending=False)
e['标签']=0
e.index=np.arange(1,114)
print(np.where(e['订单号']==169))
e.ix[0:28,'标签']='热销'
print(np.where(e['订单号']==47))
e.ix[29:58,'标签']='正常'
e.ix[59:,'标签']='滞销'
e
e.to_csv(r'C:\Users\admin\Desktop\autoshj\task3-1E.csv')


# In[386]:


#售货机画像
from wordcloud import WordCloud
import PIL.Image as image
import numpy as np
import jieba
def trans_CN(text):
    word_list=jieba.cut(text)
    result=' '.join(word_list)
    return result;

with open(r'C:\Users\admin\Desktop\minister.txt') as fp:
    text=fp.read()
    text=trans_CN(text)
    #print(text)
    stopwords_file=open(r'D:\stopwords.txt',encoding='utf-8')
    stopwords=[words.strip() for words in stopwords_file.readlines()]
    mask=np.array(image.open(r'C:\Users\admin\Desktop\6.JPG'))
    WordCloud=WordCloud(mask=mask,stopwords=stopwords,font_path="C:/Windows/Fonts/simfang.ttf").generate(text)
    image_produce=WordCloud.to_image()
    image_produce.show()
    WordCloud.to_file('A.JPG')


# In[394]:


#B画像
from wordcloud import WordCloud
import PIL.Image as image
import numpy as np
import jieba
def trans_CN(text):
    word_list=jieba.cut(text)
    result=' '.join(word_list)
    return result;

with open(r'C:\Users\admin\Desktop\B.txt') as fp:
    text=fp.read()
    text=trans_CN(text)
    #print(text)
    stopwords_file=open(r'D:\stopwords.txt',encoding='utf-8')
    stopwords=[words.strip() for words in stopwords_file.readlines()]
    mask=np.array(image.open(r'C:\Users\admin\Desktop\6.JPG'))
    WordCloud=WordCloud(mask=mask,stopwords=stopwords,font_path="C:/Windows/Fonts/simfang.ttf").generate(text)
    image_produce=WordCloud.to_image()
    image_produce.show()
    WordCloud.to_file('B.JPG')


# In[393]:


##C
from wordcloud import WordCloud
import PIL.Image as image
import numpy as np
import jieba
def trans_CN(text):
    word_list=jieba.cut(text)
    result=' '.join(word_list)
    return result;

with open(r'C:\Users\admin\Desktop\C.txt') as fp:
    text=fp.read()
    text=trans_CN(text)
    #print(text)
    stopwords_file=open(r'D:\stopwords.txt',encoding='utf-8')
    stopwords=[words.strip() for words in stopwords_file.readlines()]
    mask=np.array(image.open(r'C:\Users\admin\Desktop\6.JPG'))
    WordCloud=WordCloud(mask=mask,stopwords=stopwords,font_path="C:/Windows/Fonts/simfang.ttf").generate(text)
    image_produce=WordCloud.to_image()
    image_produce.show()
    WordCloud.to_file('C.JPG')


# In[392]:


#D
from wordcloud import WordCloud
import PIL.Image as image
import numpy as np
import jieba
def trans_CN(text):
    word_list=jieba.cut(text)
    result=' '.join(word_list)
    return result;

with open(r'C:\Users\admin\Desktop\D.txt') as fp:
    text=fp.read()
    text=trans_CN(text)
    #print(text)
    stopwords_file=open(r'D:\stopwords.txt',encoding='utf-8')
    stopwords=[words.strip() for words in stopwords_file.readlines()]
    mask=np.array(image.open(r'C:\Users\admin\Desktop\6.JPG'))
    WordCloud=WordCloud(mask=mask,stopwords=stopwords,font_path="C:/Windows/Fonts/simfang.ttf").generate(text)
    image_produce=WordCloud.to_image()
    image_produce.show()
    WordCloud.to_file('D.JPG')


# In[391]:


##E
from wordcloud import WordCloud
import PIL.Image as image
import numpy as np
import jieba
def trans_CN(text):
    word_list=jieba.cut(text)
    result=' '.join(word_list)
    return result;

with open(r'C:\Users\admin\Desktop\E.txt') as fp:
    text=fp.read()
    text=trans_CN(text)
    #print(text)
    stopwords_file=open(r'D:\stopwords.txt',encoding='utf-8')
    stopwords=[words.strip() for words in stopwords_file.readlines()]
    mask=np.array(image.open(r'C:\Users\admin\Desktop\6.JPG'))
    WordCloud=WordCloud(mask=mask,stopwords=stopwords,font_path="C:/Windows/Fonts/simfang.ttf").generate(text)
    image_produce=WordCloud.to_image()
    image_produce.show()
    WordCloud.to_file('E.JPG')


# In[ ]:


#进行标签拓展，根据画像给出总结描述，给出营销意见


# ### 预测常用的模型有哪些？分别如何使用？
# 
# 常用的预测模型有时间序列模型，灰色预测模型，支持向量机模型
# 时间序列模型
# 时间序列模型是依靠已有的时间序列数据预测未来的变化。首先要根据现有时间序列，判断该时间序列是否有明显的趋势和明显性，有的话要消除它的不平稳性和季节性等因素，再选择用合适的预测方法，如移动平均法等，使用平均误差，平均绝对误差、均方误差等来评价该种方法，找到误差最小的预测方法进行预测。
# 
# 灰色预测模型
# 是根据原始数据序列生成的数据序列进行建模，通过GM(1,1)的函数对原始数据进行累加或累减，找到新序列的内在规律，再进行建模，最后对预测结果作残差检验和拟合度检验，适用于短中期指数型变化数据的预测。
# 
# 
# 支持向量机模型
# 
#   SVR在做拟合时采用了支持向量机的思想来对数据进行回归分析，对数据进行预处理之后，划分训练集和测试集，训练SVM模型分类器对测试集进行预测，得出预测精度，最后得出预测结果。
# 

# ## 业务预测

# In[43]:


##预测每台售货机饮料类与非饮料类在未来一个月的交易额
#构建每台售货机每个月饮料类与非饮料类的交易额表格
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import calendar
import datetime
A= pd.read_csv(r'C:\Users\admin\Desktop\autoshj\test_A.csv')
B= pd.read_csv(r'C:\Users\admin\Desktop\autoshj\test_B.csv')
C= pd.read_csv(r'C:\Users\admin\Desktop\autoshj\test_C.csv')
D= pd.read_csv(r'C:\Users\admin\Desktop\autoshj\test_D.csv')
E= pd.read_csv(r'C:\Users\admin\Desktop\autoshj\test_E.csv')
A['支付时间']= pd.to_datetime(A['支付时间'], format='%Y/%m/%d %H:%M')#转换为日期
B['支付时间']= pd.to_datetime(B['支付时间'], format='%Y/%m/%d %H:%M')#转换为日期
C['支付时间']= pd.to_datetime(C['支付时间'], format='%Y/%m/%d %H:%M')#转换为日期
D['支付时间']= pd.to_datetime(D['支付时间'], format='%Y/%m/%d %H:%M')#转换为日期
E['支付时间']= pd.to_datetime(E['支付时间'], format='%Y/%m/%d %H:%M')#转换为日期


# In[82]:


all2=pd.read_csv(open(r'C:\Users\admin\Desktop\autoshj\2.csv'))
A1=pd.merge(A,all2,left_on='商品',right_on ='商品',how = 'inner')
B1=pd.merge(B,all2,left_on='商品',right_on ='商品',how = 'inner')
C1=pd.merge(C,all2,left_on='商品',right_on ='商品',how = 'inner')
D1=pd.merge(D,all2,left_on='商品',right_on ='商品',how = 'inner')
E1=pd.merge(E,all2,left_on='商品',right_on ='商品',how = 'inner')

A1['支付时间']=pd.to_datetime(A1['支付时间'], format='%Y/%m/%d %H:%M')#转换为日期
B1['支付时间']=pd.to_datetime(B1['支付时间'], format='%Y/%m/%d %H:%M')#转换为日期
C1['支付时间']=pd.to_datetime(C1['支付时间'], format='%Y/%m/%d %H:%M')#转换为日期
D1['支付时间']=pd.to_datetime(D1['支付时间'], format='%Y/%m/%d %H:%M')#转换为日期
E1['支付时间']=pd.to_datetime(E1['支付时间'], format='%Y/%m/%d %H:%M')#转换为日期




def drink(month):
    df1=A1[A1['大类']=='饮料'][A1['支付时间'].dt.month==month]['实际金额'].sum()
    return round(df1,2) 
def nodrink(month):
    df1=A1[A1['大类']=='非饮料'][A1['支付时间'].dt.month==month]['实际金额'].sum()
    return round(df1,2) 
data={'饮料类':[drink(1),drink(2),drink(3),drink(4),drink(5),drink(6),drink(7),drink(8),drink(9),drink(10),drink(11),drink(12)],'非饮料类':     [nodrink(1),nodrink(2),nodrink(3),nodrink(4),nodrink(5),nodrink(6),nodrink(7),nodrink(8),nodrink(9),nodrink(10),nodrink(11),nodrink(12)]}
pre_A=pd.DataFrame(data,index=['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月'],columns=['饮料类','非饮料类'])
pre_A


# In[83]:



def drink(month):
    df1=B1[B1['大类']=='饮料'][B1['支付时间'].dt.month==month]['实际金额'].sum()
    return round(df1,2) 
def nodrink(month):
    df1=B1[B1['大类']=='非饮料'][B1['支付时间'].dt.month==month]['实际金额'].sum()
    return round(df1,2) 
data={'饮料类':[drink(1),drink(2),drink(3),drink(4),drink(5),drink(6),drink(7),drink(8),drink(9),drink(10),drink(11),drink(12)],'非饮料类':     [nodrink(1),nodrink(2),nodrink(3),nodrink(4),nodrink(5),nodrink(6),nodrink(7),nodrink(8),nodrink(9),nodrink(10),nodrink(11),nodrink(12)]}
pre_B=pd.DataFrame(data,index=['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月'],columns=['饮料类','非饮料类'])
pre_B


# In[84]:


def drink(month):
    df1=C1[C1['大类']=='饮料'][C1['支付时间'].dt.month==month]['实际金额'].sum()
    return round(df1,2) 
def nodrink(month):
    df1=C1[C1['大类']=='非饮料'][C1['支付时间'].dt.month==month]['实际金额'].sum()
    return round(df1,2) 
data={'饮料类':[drink(1),drink(2),drink(3),drink(4),drink(5),drink(6),drink(7),drink(8),drink(9),drink(10),drink(11),drink(12)],'非饮料类':     [nodrink(1),nodrink(2),nodrink(3),nodrink(4),nodrink(5),nodrink(6),nodrink(7),nodrink(8),nodrink(9),nodrink(10),nodrink(11),nodrink(12)]}
pre_C=pd.DataFrame(data,index=['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月'],columns=['饮料类','非饮料类'])
pre_C


# In[85]:


def drink(month):
    df1=D1[D1['大类']=='饮料'][D1['支付时间'].dt.month==month]['实际金额'].sum()
    return round(df1,2) 
def nodrink(month):
    df1=D1[D1['大类']=='非饮料'][D1['支付时间'].dt.month==month]['实际金额'].sum()
    return round(df1,2) 
data={'饮料类':[drink(1),drink(2),drink(3),drink(4),drink(5),drink(6),drink(7),drink(8),drink(9),drink(10),drink(11),drink(12)],'非饮料类':     [nodrink(1),nodrink(2),nodrink(3),nodrink(4),nodrink(5),nodrink(6),nodrink(7),nodrink(8),nodrink(9),nodrink(10),nodrink(11),nodrink(12)]}
pre_D=pd.DataFrame(data,index=['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月'],columns=['饮料类','非饮料类'])
pre_D


# In[86]:


def drink(month):
    df1=E1[E1['大类']=='饮料'][E1['支付时间'].dt.month==month]['实际金额'].sum()
    return round(df1,2) 
def nodrink(month):
    df1=E1[E1['大类']=='非饮料'][E1['支付时间'].dt.month==month]['实际金额'].sum()
    return round(df1,2) 
data={'饮料类':[drink(1),drink(2),drink(3),drink(4),drink(5),drink(6),drink(7),drink(8),drink(9),drink(10),drink(11),drink(12)],'非饮料类':     [nodrink(1),nodrink(2),nodrink(3),nodrink(4),nodrink(5),nodrink(6),nodrink(7),nodrink(8),nodrink(9),nodrink(10),nodrink(11),nodrink(12)]}
pre_E=pd.DataFrame(data,index=['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月'],columns=['饮料类','非饮料类'])
pre_E


# In[87]:


##对每台售货机的数据进行预测
#提取数据、标签和列名

data2_data=pre_A.iloc[:,0:2].values
data2_target=pre_A.index
data2_name=pre_A.columns[::]
print('原始数据集数据的形状为：',data2_data.shape)


# In[88]:


#划分测试集和数据集
from sklearn.model_selection import train_test_split
data2_data_train,data2_data_test,data2_target_train,data2_target_test= train_test_split(data2_data,data2_target,test_size=0.2,random_state =62)
print('训练集数据的形状为：',data2_data_train.shape)
print('训练集标签的形状为：',data2_target_train.shape)
print('测试集数据的形状为：',data2_data_test.shape)
print('测试集标签的形状为：',data2_target_test.shape)
print('原始数据集标签的形状为：',data2_target.shape)


# In[89]:


#标准差标准化
from sklearn.preprocessing import StandardScaler
stdScale = StandardScaler().fit(data2_data_train)
data2_trainScaler=stdScale.transform(data2_data_train)
data2_testScaler=stdScale.transform(data2_data_test)
print('标注差标准化后训练集数据的方差为：',np.var(data2_trainScaler))
print('标注差标准化后训练集数据的均值为：',np.mean(data2_trainScaler))
print('标注差标准化后测试集数据的方差为：',np.var(data2_testScaler))
print('标注差标准化后测试集数据的均值为：',np.mean(data2_testScaler))


# In[90]:


#PCA降维
from sklearn.decomposition import PCA
pca_model = PCA(n_components=1).fit(data2_trainScaler)
data2_trainPca= pca_model.transform(data2_trainScaler)
data2_testPca=pca_model.transform(data2_testScaler)
print('pca降维前data2训练集数据的形状为：',data2_trainScaler.shape)
print('pca降维后data2训练集数据的形状为：',data2_trainPca.shape)
print('pca降维前data2测试集数据的形状为：',data2_testScaler.shape)
print('pca降维后data2测试集数据的形状为：',data2_testPca.shape)


# In[91]:


#替换索引和列名
data2=pre_A
data2.index=[1,2,3,4,5,6,7,8,9,10,11,12]
data2.columns=['x1','x2']
data3=data2.to_excel('./dataA.xlsx')


# In[92]:


data3=pd.read_excel('./dataA.xlsx')
data3


# In[95]:


#自定义灰色预测函数
import numpy as np
import pandas as pd
def GM11(x0): 
  x1 = x0.cumsum() #1-AGO序列 累加生成1
  z1 = (x1[:len(x1)-1] + x1[1:])/2.0 #紧邻均值（MEAN）生成序列2
  z1 = z1.reshape((len(z1),1))
  B = np.append(-z1, np.ones_like(z1), axis = 1)
  Yn = x0[1:].reshape((len(x0)-1, 1))
  [[a],[b]] = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Yn) #计算参数
  f = lambda k: (x0[0]-b/a)*np.exp(-a*(k-1))-(x0[0]-b/a)*np.exp(-a*(k-2)) #累减还原值
  delta = np.abs(x0 - np.array([f(i) for i in range(1,len(x0)+1)]))
  C = delta.std()/x0.std() #残差的方差=s2/s1
  P = 1.0*(np.abs(delta - delta.mean()) < 0.6745*x0.std()).sum()/len(x0)#后验差比值计算
  return f, a, b, x0[0], C, P #返回灰色预测函数、a、b、首项、方差比、小残差概率


# In[102]:


#预测2018年1月的各项值
data3=pd.read_excel('./dataA.xlsx')
data3.index=range(1,13)
data3.loc[2018]=None

l=['x1','x2']
for i in l:
    f=GM11(data3.loc[range(1,13),i].as_matrix())[0]
    data3.loc[2018,i]=f(len(data3))
    data3[i]=data3[i].round(2)
outputfile='./dataA_GM11.xls'
#y.extend([np.nan,np.nan])
data3.to_excel(outputfile)
print('A售货机预测结果为：\n',data3)


# In[103]:


#提取数据、标签和列名

data2_data=pre_B.iloc[:,0:2].values
data2_target=pre_B.index
data2_name=pre_B.columns[::]
print('原始数据集数据的形状为：',data2_data.shape)
#划分测试集和数据集
from sklearn.model_selection import train_test_split
data2_data_train,data2_data_test,data2_target_train,data2_target_test= train_test_split(data2_data,data2_target,test_size=0.2,random_state =62)
print('训练集数据的形状为：',data2_data_train.shape)
print('训练集标签的形状为：',data2_target_train.shape)
print('测试集数据的形状为：',data2_data_test.shape)
print('测试集标签的形状为：',data2_target_test.shape)
print('原始数据集标签的形状为：',data2_target.shape)
#标准差标准化
from sklearn.preprocessing import StandardScaler
stdScale = StandardScaler().fit(data2_data_train)
data2_trainScaler=stdScale.transform(data2_data_train)
data2_testScaler=stdScale.transform(data2_data_test)
print('标注差标准化后训练集数据的方差为：',np.var(data2_trainScaler))
print('标注差标准化后训练集数据的均值为：',np.mean(data2_trainScaler))
print('标注差标准化后测试集数据的方差为：',np.var(data2_testScaler))
print('标注差标准化后测试集数据的均值为：',np.mean(data2_testScaler))
#PCA降维
from sklearn.decomposition import PCA
pca_model = PCA(n_components=1).fit(data2_trainScaler)
data2_trainPca= pca_model.transform(data2_trainScaler)
data2_testPca=pca_model.transform(data2_testScaler)
print('pca降维前data2训练集数据的形状为：',data2_trainScaler.shape)
print('pca降维后data2训练集数据的形状为：',data2_trainPca.shape)
print('pca降维前data2测试集数据的形状为：',data2_testScaler.shape)
print('pca降维后data2测试集数据的形状为：',data2_testPca.shape)
#替换索引和列名
data2=pre_B
data2.index=[1,2,3,4,5,6,7,8,9,10,11,12]
data2.columns=['x1','x2']
data3=data2.to_excel('./dataB.xlsx')
#自定义灰色预测函数
import numpy as np
import pandas as pd
def GM11(x0): 
  x1 = x0.cumsum() #1-AGO序列 累加生成1
  z1 = (x1[:len(x1)-1] + x1[1:])/2.0 #紧邻均值（MEAN）生成序列2
  z1 = z1.reshape((len(z1),1))
  B = np.append(-z1, np.ones_like(z1), axis = 1)
  Yn = x0[1:].reshape((len(x0)-1, 1))
  [[a],[b]] = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Yn) #计算参数
  f = lambda k: (x0[0]-b/a)*np.exp(-a*(k-1))-(x0[0]-b/a)*np.exp(-a*(k-2)) #累减还原值
  delta = np.abs(x0 - np.array([f(i) for i in range(1,len(x0)+1)]))
  C = delta.std()/x0.std() #残差的方差=s2/s1
  P = 1.0*(np.abs(delta - delta.mean()) < 0.6745*x0.std()).sum()/len(x0)#后验差比值计算
  return f, a, b, x0[0], C, P #返回灰色预测函数、a、b、首项、方差比、小残差概率
#预测2018年1月的各项值
data3=pd.read_excel('./dataB.xlsx')
data3.index=range(1,13)
data3.loc[2018]=None

l=['x1','x2']
for i in l:
    f=GM11(data3.loc[range(1,13),i].as_matrix())[0]
    data3.loc[2018,i]=f(len(data3))
    data3[i]=data3[i].round(2)
outputfile='./dataB_GM11.xls'
#y.extend([np.nan,np.nan])
data3.to_excel(outputfile)
print('B售货机预测结果为：\n',data3)


# In[104]:


#提取数据、标签和列名

data2_data=pre_C.iloc[:,0:2].values
data2_target=pre_C.index
data2_name=pre_C.columns[::]
print('原始数据集数据的形状为：',data2_data.shape)
#划分测试集和数据集
from sklearn.model_selection import train_test_split
data2_data_train,data2_data_test,data2_target_train,data2_target_test= train_test_split(data2_data,data2_target,test_size=0.2,random_state =62)
print('训练集数据的形状为：',data2_data_train.shape)
print('训练集标签的形状为：',data2_target_train.shape)
print('测试集数据的形状为：',data2_data_test.shape)
print('测试集标签的形状为：',data2_target_test.shape)
print('原始数据集标签的形状为：',data2_target.shape)
#标准差标准化
from sklearn.preprocessing import StandardScaler
stdScale = StandardScaler().fit(data2_data_train)
data2_trainScaler=stdScale.transform(data2_data_train)
data2_testScaler=stdScale.transform(data2_data_test)
print('标注差标准化后训练集数据的方差为：',np.var(data2_trainScaler))
print('标注差标准化后训练集数据的均值为：',np.mean(data2_trainScaler))
print('标注差标准化后测试集数据的方差为：',np.var(data2_testScaler))
print('标注差标准化后测试集数据的均值为：',np.mean(data2_testScaler))
#PCA降维
from sklearn.decomposition import PCA
pca_model = PCA(n_components=1).fit(data2_trainScaler)
data2_trainPca= pca_model.transform(data2_trainScaler)
data2_testPca=pca_model.transform(data2_testScaler)
print('pca降维前data2训练集数据的形状为：',data2_trainScaler.shape)
print('pca降维后data2训练集数据的形状为：',data2_trainPca.shape)
print('pca降维前data2测试集数据的形状为：',data2_testScaler.shape)
print('pca降维后data2测试集数据的形状为：',data2_testPca.shape)
#替换索引和列名
data2=pre_C
data2.index=[1,2,3,4,5,6,7,8,9,10,11,12]
data2.columns=['x1','x2']
data3=data2.to_excel('./dataC.xlsx')
#自定义灰色预测函数
import numpy as np
import pandas as pd
def GM11(x0): 
  x1 = x0.cumsum() #1-AGO序列 累加生成1
  z1 = (x1[:len(x1)-1] + x1[1:])/2.0 #紧邻均值（MEAN）生成序列2
  z1 = z1.reshape((len(z1),1))
  B = np.append(-z1, np.ones_like(z1), axis = 1)
  Yn = x0[1:].reshape((len(x0)-1, 1))
  [[a],[b]] = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Yn) #计算参数
  f = lambda k: (x0[0]-b/a)*np.exp(-a*(k-1))-(x0[0]-b/a)*np.exp(-a*(k-2)) #累减还原值
  delta = np.abs(x0 - np.array([f(i) for i in range(1,len(x0)+1)]))
  C = delta.std()/x0.std() #残差的方差=s2/s1
  P = 1.0*(np.abs(delta - delta.mean()) < 0.6745*x0.std()).sum()/len(x0)#后验差比值计算
  return f, a, b, x0[0], C, P #返回灰色预测函数、a、b、首项、方差比、小残差概率
#预测2018年1月的各项值
data3=pd.read_excel('./dataC.xlsx')
data3.index=range(1,13)
data3.loc[2018]=None

l=['x1','x2']
for i in l:
    f=GM11(data3.loc[range(1,13),i].as_matrix())[0]
    data3.loc[2018,i]=f(len(data3))
    data3[i]=data3[i].round(2)
outputfile='./dataC_GM11.xls'
#y.extend([np.nan,np.nan])
data3.to_excel(outputfile)
print('C售货机预测结果为：\n',data3)


# In[105]:


#提取数据、标签和列名

data2_data=pre_D.iloc[:,0:2].values
data2_target=pre_D.index
data2_name=pre_D.columns[::]
print('原始数据集数据的形状为：',data2_data.shape)
#划分测试集和数据集
from sklearn.model_selection import train_test_split
data2_data_train,data2_data_test,data2_target_train,data2_target_test= train_test_split(data2_data,data2_target,test_size=0.2,random_state =62)
print('训练集数据的形状为：',data2_data_train.shape)
print('训练集标签的形状为：',data2_target_train.shape)
print('测试集数据的形状为：',data2_data_test.shape)
print('测试集标签的形状为：',data2_target_test.shape)
print('原始数据集标签的形状为：',data2_target.shape)
#标准差标准化
from sklearn.preprocessing import StandardScaler
stdScale = StandardScaler().fit(data2_data_train)
data2_trainScaler=stdScale.transform(data2_data_train)
data2_testScaler=stdScale.transform(data2_data_test)
print('标注差标准化后训练集数据的方差为：',np.var(data2_trainScaler))
print('标注差标准化后训练集数据的均值为：',np.mean(data2_trainScaler))
print('标注差标准化后测试集数据的方差为：',np.var(data2_testScaler))
print('标注差标准化后测试集数据的均值为：',np.mean(data2_testScaler))
#PCA降维
from sklearn.decomposition import PCA
pca_model = PCA(n_components=1).fit(data2_trainScaler)
data2_trainPca= pca_model.transform(data2_trainScaler)
data2_testPca=pca_model.transform(data2_testScaler)
print('pca降维前data2训练集数据的形状为：',data2_trainScaler.shape)
print('pca降维后data2训练集数据的形状为：',data2_trainPca.shape)
print('pca降维前data2测试集数据的形状为：',data2_testScaler.shape)
print('pca降维后data2测试集数据的形状为：',data2_testPca.shape)
#替换索引和列名
data2=pre_D
data2.index=[1,2,3,4,5,6,7,8,9,10,11,12]
data2.columns=['x1','x2']
data3=data2.to_excel('./dataD.xlsx')
#自定义灰色预测函数
import numpy as np
import pandas as pd
def GM11(x0): 
  x1 = x0.cumsum() #1-AGO序列 累加生成1
  z1 = (x1[:len(x1)-1] + x1[1:])/2.0 #紧邻均值（MEAN）生成序列2
  z1 = z1.reshape((len(z1),1))
  B = np.append(-z1, np.ones_like(z1), axis = 1)
  Yn = x0[1:].reshape((len(x0)-1, 1))
  [[a],[b]] = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Yn) #计算参数
  f = lambda k: (x0[0]-b/a)*np.exp(-a*(k-1))-(x0[0]-b/a)*np.exp(-a*(k-2)) #累减还原值
  delta = np.abs(x0 - np.array([f(i) for i in range(1,len(x0)+1)]))
  C = delta.std()/x0.std() #残差的方差=s2/s1
  P = 1.0*(np.abs(delta - delta.mean()) < 0.6745*x0.std()).sum()/len(x0)#后验差比值计算
  return f, a, b, x0[0], C, P #返回灰色预测函数、a、b、首项、方差比、小残差概率
#预测2018年1月的各项值
data3=pd.read_excel('./dataD.xlsx')
data3.index=range(1,13)
data3.loc[2018]=None

l=['x1','x2']
for i in l:
    f=GM11(data3.loc[range(1,13),i].as_matrix())[0]
    data3.loc[2018,i]=f(len(data3))
    data3[i]=data3[i].round(2)
outputfile='./dataD_GM11.xls'
#y.extend([np.nan,np.nan])
data3.to_excel(outputfile)
print('D售货机预测结果为：\n',data3)


# In[107]:


#提取数据、标签和列名

data2_data=pre_E.iloc[:,0:2].values
data2_target=pre_E.index
data2_name=pre_E.columns[::]
print('原始数据集数据的形状为：',data2_data.shape)
#划分测试集和数据集
from sklearn.model_selection import train_test_split
data2_data_train,data2_data_test,data2_target_train,data2_target_test= train_test_split(data2_data,data2_target,test_size=0.2,random_state =62)
print('训练集数据的形状为：',data2_data_train.shape)
print('训练集标签的形状为：',data2_target_train.shape)
print('测试集数据的形状为：',data2_data_test.shape)
print('测试集标签的形状为：',data2_target_test.shape)
print('原始数据集标签的形状为：',data2_target.shape)
#标准差标准化
from sklearn.preprocessing import StandardScaler
stdScale = StandardScaler().fit(data2_data_train)
data2_trainScaler=stdScale.transform(data2_data_train)
data2_testScaler=stdScale.transform(data2_data_test)
print('标注差标准化后训练集数据的方差为：',np.var(data2_trainScaler))
print('标注差标准化后训练集数据的均值为：',np.mean(data2_trainScaler))
print('标注差标准化后测试集数据的方差为：',np.var(data2_testScaler))
print('标注差标准化后测试集数据的均值为：',np.mean(data2_testScaler))
#PCA降维
from sklearn.decomposition import PCA
pca_model = PCA(n_components=1).fit(data2_trainScaler)
data2_trainPca= pca_model.transform(data2_trainScaler)
data2_testPca=pca_model.transform(data2_testScaler)
print('pca降维前data2训练集数据的形状为：',data2_trainScaler.shape)
print('pca降维后data2训练集数据的形状为：',data2_trainPca.shape)
print('pca降维前data2测试集数据的形状为：',data2_testScaler.shape)
print('pca降维后data2测试集数据的形状为：',data2_testPca.shape)
#替换索引和列名
data2=pre_E
data2.index=[1,2,3,4,5,6,7,8,9,10,11,12]
data2.columns=['x1','x2']
data3=data2.to_excel('./dataE.xlsx')
#自定义灰色预测函数
import numpy as np
import pandas as pd
def GM11(x0): 
  x1 = x0.cumsum() #1-AGO序列 累加生成1
  z1 = (x1[:len(x1)-1] + x1[1:])/2.0 #紧邻均值（MEAN）生成序列2
  z1 = z1.reshape((len(z1),1))
  B = np.append(-z1, np.ones_like(z1), axis = 1)
  Yn = x0[1:].reshape((len(x0)-1, 1))
  [[a],[b]] = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Yn) #计算参数
  f = lambda k: (x0[0]-b/a)*np.exp(-a*(k-1))-(x0[0]-b/a)*np.exp(-a*(k-2)) #累减还原值
  delta = np.abs(x0 - np.array([f(i) for i in range(1,len(x0)+1)]))
  C = delta.std()/x0.std() #残差的方差=s2/s1
  P = 1.0*(np.abs(delta - delta.mean()) < 0.6745*x0.std()).sum()/len(x0)#后验差比值计算
  return f, a, b, x0[0], C, P #返回灰色预测函数、a、b、首项、方差比、小残差概率
#预测2018年1月的各项值
data3=pd.read_excel('./dataE.xlsx')
data3.index=range(1,13)
data3.loc[2018]=None

l=['x1','x2']
for i in l:
    f=GM11(data3.loc[range(1,13),i].as_matrix())[0]
    data3.loc[2018,i]=f(len(data3))
    data3[i]=data3[i].round(2)
outputfile='./dataE_GM11.xls'
#y.extend([np.nan,np.nan])
data3.to_excel(outputfile)
print('E售货机预测结果为：\n',data3)

