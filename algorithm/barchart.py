import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.font_manager import findfont,FontProperties
# 查询matplotlibrc所在目录
print(matplotlib.matplotlib_fname())
# 查询当前的font.family, matplotlib 只接受.ttf结尾的字体
print(matplotlib.get_configdir())
# 查看matplotlib的字体存放目录
print(findfont(FontProperties(family=FontProperties().get_family())))
# 将下载的黑体字体放置到字体库中
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
X = np.arange(2008, 2017, 1)
print(X.shape)
Y = np.array([0.5, 0.6, 1.0, 1.3, 1.4, 1.5, 2.0, 2.5, 3.2])
labels = ['2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016']
print(Y.shape)
fig = plt.figure()
plt.bar(X, Y, 0.6, color='green', tick_label=labels)
plt.xlabel("年")
plt.ylabel("十亿美元")
plt.title("中国机器视觉市场份额")
plt.show()
