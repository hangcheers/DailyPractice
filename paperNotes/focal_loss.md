## focal loss for dense object detection

目标检测中 根据有标签的数据划分 positive / negative training examples(其中
通过 bounding box 的IOU来确定正负样本，IOU超过一定的阈值则为正样本)
一般负样本会多于正样本.但是为了训练出来的模型不偏向于negative，需要保证样本数目的balance.
在本文中的imbalance指的是# foreground-background即前景和背景的数据的imbalance。
### hard negative mining 
传统的方法是通过找到hard negative training examples进行不断的迭代。 
但是如何确定useful negative training examples?
当classifier不能很好的工作的时候，it throws a bunch of false positive
(E.x people detected where there aren’t actually people)
所以hard negative就是把falsely detected path重新作为negative examples，
并add the negative to the training set. 然后retrain the classifier, 其目的是
expect these “new” examples to help steer the classifier away from its current mistakes.  

*R-CNN中是这么使用 hard negative mining method:*  
> we adopt the standard hard negative mining method. Hard negative mining converges quickly and in practice mAP stops increasing after only a single pass over all images.


### cross entropy
熵entropy -> KL散度K-L divergence ->交叉熵 cross entropy   

熵反映了一个事件所包含的信息量，p(x)越小，熵越大，
即越不可能发生的事情包含的信息量越大。此外，熵还可以从不确定性度量理解，分布越集中，不确定性越小，熵越小。

KL散度反映了两个分布之间的不同D_KL(A| |B)
KL散度由A自己的熵与B在A上的期望所决定.
假设有两个分布p,q  在给定样本集上面的交叉熵为H(p)+D_KL(p| |q)  

交叉熵是用于理解两个事件之间的相互关系
当p已知时，H(p)可以理解为常数，
因此KL散度和交叉熵在特定条件下是等价的。  

机器学习的过程就是希望训练数据在模型上的分布P(model)与
真实的数据分布P(real)近似，使用KL散度可以最小化两个分布之间的不同。但是没有真实数据的分布时，退而求其次
希望模型学到的分布和训练数据的分布近似
即P(training)和P(model)近似，最小化模型数据分布与训练数据分布的散度。
因为训练数据分布情况是已知的，所以最小化KL散度等价于最小化交叉熵，但是为了防止过拟合的现象，还需要在测试集上测试

### 本文的做法
本文是通过「loss」改变为「focal loss」而不是「architecture」创新
来提高one-stage来保证speed的同时有比较高的accuracy
Focal loss: 为了differentiate between easy and hard examples
like false positive ( people detected where they are not really people)
FL(Pt)=-(1-Pt)^gama(log(Pt)) 系数相当于modulating factors
其中gama的取值[0,5]
0<Pt<=1时 log(Pt) 始终为负，1-Pt始终为非负 则FL(Pt)始终为正
当gama=0时，FL(Pt)就是传统的交叉熵值，gama增大时，modulating factors也会随之增大。
设计的FL的目的在于即使易分类的样本(easy examples)很多时，但是modulating factors控制了易分类的样本对整体的loss的贡献程度，
即降低了权重，但是如果出现了误分类情况，即Pt的值很小，modulating factors趋近于1，相比原来情况，loss并没有很大的改变。
总之通过降低easy examples的重要性，使得模型在训练时可以更focus on hard examples. 从而提高了accuracy

### 训练过程
本文以RetinaNet为backbone（主干网络），两个subnets，其中一个为分类网络对feature map的输出进行分类，另一个为回归网络对位置进行修正。  

思考了一下minibatch-SGD的相关问题：  
优化：利用关于优化解的信息，不断的逼近最优解。  

梯度下降法有效的前提条件是：
梯度的方向指向最优化的方向，因此沿着梯度的方向，
我们可以逐步靠近最优解。凸函数等比较简单的问题，
可以满足这一前提，因此梯度下降法可以work 。
但是当优化的目标又大量的局部极值时，绝大多数
解的空间位置的梯度方向不再指向最优解。  

在DL中，梯度下降是一种最小化风险函数、损失函数
的常用方法。
在GD(Gradient descent)问题中，每次迭代更新模型参数时，
都要用到全部的训练数据。
在SGD(Stochastic Gradient Descent)问题中，每次迭代更新
模型参数时，只用到一个训练数据来更新参数。因此，
minibatch-SGD的收敛速度会比minibatch-GD的要快。
因为是随机化的用minibatch中的一个训练数据，
虽然不是每次迭代得到的损失都向着全局最优的方向，
但是大的整体的方向是向着全局最优解的，
最终的结果往往是在全局最优解的附近。



