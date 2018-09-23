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

Reference:[scikit-learn](http://scikit-learn.org/stable/)、李航《统计学习方法》
