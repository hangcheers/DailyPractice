# DailyPractice
## svm    
### 分类  
1.线性可分支持向量机  
  输入：线性可分训练集。目标：几何间隔最大的分离超平面。迭代的过程就相当于超平面向误分类点移动的过程，以减少距离，直到超平面超过误分点直至该误分类点被完全分类  
2.线性支持向量机  
3.非线性支持向量机  
通过对偶优化问题中的最优解，求解得到原问题的最优解。
寻找分离超平面&相应的分类决策函数

## naive bayes 

### 极大似然估计  
极大似然估计属于一个参数估计问题，相当于把实际问题的求解简化为参数估计的一种方式。只有当参数选定时，才会有一个描述给定现象的模型实例。MLE就是一组使得观测值概率最大的参数估计的问题。在PCA中，对先验概率进行极大似然估计。

### 原理
朴素贝叶斯法对条件概率分布做了条件独立性的假设。
利用朴素贝叶斯分类时，给定输入x,通过学习到的模型计算后验概率<a href="https://www.codecogs.com/eqnedit.php?latex=$$P(Y=C_k|X=x)&space;$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$P(Y=C_k|X=x)&space;$$" title="$$P(Y=C_k|X=x) $$" /></a>
后验概率越大，物体属于这一类的概率越大，选择最大的类x作为输出。
贝叶斯推理（inference）是通过计算数据的后验分布，不断迭代更新先验分布，在document classification和spam filtering中效果很好。
适合较小的数据集

## pca 
将高维度的矩阵降维处理，丢弃不重要的奇异值，减少处理量。  

### 步骤
1.将原始数据转换为矩阵
2.对每一个属性字段进行均值化处理
3.求协方差<a href="https://www.codecogs.com/eqnedit.php?latex=$$&space;C=&space;V*L*V^T&space;$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$&space;C=&space;V*L*V^T&space;$$" title="$$ C= V*L*V^T $$" /></a> (V是矩阵，其每一个列向量都是一个特征值；L是对角矩阵，按特征值的大小降序排序)
4.主元（principal axes）1是最大奇异值对应的奇异向量，主成分（principal components)1是数据在主元上的投影。

## decision tree
决策树表示的是对象属性和对象值之间的一种映射关系。每个内部节点表示的是一个特征属性的测试，每个分支表示的是一个测试输出，每个叶节点存放一个类别。使用决策树的过程就是从根节点开始，测试分类项相应的特征属性，按照特征属性的值选择输出分支，直到到达叶节点，把叶子节点存放的类别作为决策的结果。
### 三种生成算法：  
1.ID3 --- 信息增益最大的原则  
2.C4.5 --- 信息增益比最大的原则  
3.CART  
  回归树：平方误差最小的原则  
  分类树：基尼指数最小的原则    
寻找最优决策树是一个NP-C问题，无法利用计算机在多项式时间内找到全局最优解，只能通过启发式算法进行近似求解，在每一个节点上做局部最优的选择。
 
### 决策树剪枝的核心思想：  

任何一个复杂的决策树都能生成有限个数的子树{T0,T1,...,Tn},为了得到树的所有子序列，我们需要自下而上的每次只进行一次剪枝操作。剪枝操作后的子序列应当是当前参数的最优子树，通过最优子树来求解参数a, 用n+1棵子树预测独立的验证集，选择误差小的子序列。优先减去g(t)「即剪枝后整体损失函数减少的程度也成为误差增益」最小的子树。g(t)是由剪枝前后模型的拟合能力变化决定的。
设决策树的任意结点为t,结点t下有若干子结点「即以t为根结点的子树」

剪枝前的损失函数：Ca(Tt)=C(Tt)+a|Tt|  
剪枝后（子结点吞并了它的子树,原先的子结点已经变成了当前的叶结点）损失函数：<a href="https://www.codecogs.com/eqnedit.php?latex=$$C_{\alpha&space;}(t)=C(t)&plus;\alpha&space;$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$C_{\alpha&space;}(t)=C(t)&plus;\alpha&space;$$" title="$$C_{\alpha }(t)=C(t)+\alpha $$" /></a>

## adaboost
### 集成学习：
通过训练若干个体学习器，通过一定的结合策略形成一个强学习器。boost系列属于个体学习器之间存在强依赖关系【串行】。  
假设现在是二分类问题，学习器的精确率大于1/2时才有意义。（因为随机判断的精确率都已经是1/2了）  
<a href="https://www.codecogs.com/eqnedit.php?latex=$$G(x)=sign[f(x)]=sign[\alpha_{1}G1(x)&plus;\alpha_{2}G2(x)&plus;...&plus;\alpha_{n}Gn(x)]&space;$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$G(x)=sign[f(x)]=sign[\alpha_{1}G1(x)&plus;\alpha_{2}G2(x)&plus;...&plus;\alpha_{n}Gn(x)]&space;$$" title="$$G(x)=sign[f(x)]=sign[\alpha_{1}G1(x)+\alpha_{2}G2(x)+...+\alpha_{n}Gn(x)] $$" /></a>其中ai表示个体学习器的重要性；Gi(x)表示个体学习器  
boost意为提升，一个逐步优化集成学习器的过程。adaboost通过指数型损失函数exp(x)调整权重，分类正确的降低权重，分类错误的增加权重，「希望把之前分类错误的数据在下一个个体学习器中分类正确」降低个体学习器的分类误差，得到最终的集成学习器。  

## 提升树  
### GBDT  
GBDT的弱学习器限定了只能使用CART回归树模型，使用前向分布算法。前一轮迭代得到的学习器是<a href="https://www.codecogs.com/eqnedit.php?latex=$$f_{t-1}(x)$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$f_{t-1}(x)$$" title="$$f_{t-1}(x)$$" /></a>，损失函数是<a href="https://www.codecogs.com/eqnedit.php?latex=$$L(y,f_{t-1}(x))$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$L(y,f_{t-1}(x))$$" title="$$L(y,f_{t-1}(x))$$" /></a>。  第t轮的第i个样本的损失函数的负梯度表示为<a href="https://www.codecogs.com/eqnedit.php?latex=$&space;T_{ti}=-[{\cfrac{\partial&space;L(y_i,f(x_i))}{\partial&space;f(x_i)}}]&space;_{f(x)=f_{t-1}(x)}$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$&space;T_{ti}=-[{\cfrac{\partial&space;L(y_i,f(x_i))}{\partial&space;f(x_i)}}]&space;_{f(x)=f_{t-1}(x)}$" title="$ T_{ti}=-[{\cfrac{\partial L(y_i,f(x_i))}{\partial f(x_i)}}] _{f(x)=f_{t-1}(x)}$" /></a>  用损失函数的负梯度来拟合本轮损失的近似值

用<a href="https://www.codecogs.com/eqnedit.php?latex=$$(x_i,r_{ti})(i=1,2,...,m)$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$(x_i,r_{ti})(i=1,2,...,m)$$" title="$$(x_i,r_{ti})(i=1,2,...,m)$$" /></a>
残差拟合得到CART回归树，得到第t棵回归树，其对应的叶节点的区域为<a href="https://www.codecogs.com/eqnedit.php?latex=$$R_{tj}(j=1,2,...,m)$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$R_{tj}(j=1,2,...,m)$$" title="$$R_{tj}(j=1,2,...,m)$$" /></a>针对每一个叶节点的样本，使得损失函数最小，即拟合叶子节点的输出值

### 步骤  
1.初始化弱学习器 2.迭代次数 t=1,2,...T时，针对样本i=1,2....,m 计算负梯度值，利用残差拟合CART回归树得到第t棵回归树 3.对叶子区域，计算拟合值，更新强学习器，得到强学习器f(x)表达树

Reference:[scikit-learn](http://scikit-learn.org/stable/)、[scipy](https://docs.scipy.org/)、[matplotlib](https://matplotlib.org)、李航《统计学习方法》
