# DailyPractice
## svm  
寻找分离超平面&相应的分类决策函数

## naive bayes 
利用朴素贝叶斯分类时，给定输入x,通过学习到的模型计算后验概率P(Y=Ck|X=x)  
后验概率越大，物体属于这一类的概率越大，选择最大的类x作为输出。
贝叶斯推理（inference）是通过计算数据的后验分布，不断迭代更新先验分布，在document classification和spam filtering中效果很好。
适合较小的数据集

## pca 
将高维度的矩阵降维处理，丢弃不重要的奇异值，减少处理量。
1.将原始数据转换为矩阵
2.对每一个属性字段进行均值化处理
3.求协方差 C= V*L*V.T (V是矩阵，其每一个列向量都是一个特征值；L是对角矩阵，按特征值的大小降序排序)
4.主元（principal axes）1是最大奇异值对应的奇异向量，主成分（principal components)1是数据在主元上的投影。

## decision tree
决策树剪枝的核心思想：
任何一个复杂的决策树都能生成有限个数的子树{T0,T1,...,Tn},为了得到树的所有子序列，我们需要自下而上的每次只进行一次剪枝操作。剪枝操作后的子序列应当是当前参数的最优子树。用n+1棵子树预测独立的验证集，选择误差小的子序列。优先减去g(t)「即剪枝后整体损失函数减少的程度」最小的子树。
剪枝前的以t为根结点的子树Tt的损失函数：Ca(Tt)=C(Tt)+a|Tt|  
剪枝后（子结点吞并了它的子树,原先的子结点已经变成了当前的叶结点）损失函数：Ca(t)=C(t)+a

Reference:[scikit-learn](http://scikit-learn.org/stable/)、[scipy](https://docs.scipy.org/)、[matplotlib](https://matplotlib.org)、李航《统计学习方法》
