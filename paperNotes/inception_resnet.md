## 前言
Google Research推出的Inception module历经了四次“迭代”，GoogleNet也被成为Inception-v1，加入了batch Normalization的inception 
module成为inception-v2，加入了factorization的称为inception-v3,在和Microsoft Research推出的Resnet 相结合得到了inception-v4。前面三种在之前有提到，今天的笔记主要研究*Inception-v4, Inception-ResNet and
the Impact of Residual Connections on Learning*
## Resnet
### introduction
一般来说，模型的深度加深，学习能力增强，但是不能简单的增加网络的深度，否则会出现随着网络深度增加，training error和test error变高的现象。这也说明网络结构变复杂时，optimization变得更加困难。
Kaiming He提出了深度残差学习（deep residual learning framework）,通过网络结构的创新来有效的解决了上述的梯度弥散现象(degradation)。
### 网络结构
![1](https://img-blog.csdn.net/20170220201128938?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd3NwYmE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)  

从以下两点来分析网络结构的创新  
- ***shortcut connection***   
残差的网络结构是前向神经网络+shortcut。从上图可以看出在已有的网络结构中增加了一个branch，起到的是恒等映射（identity mapping）的作用，这样保证了一个深度模型的training error最起码保证shallow counterpart模型的training error是一致的，错误率不会高于浅层。此外，这种做法既不会增加额外的参数也不会增加额外的计算量。
- ***residual representations***  
两种函数的表达效果是相同的，但是优化的难度是不同的。对残差进行拟合显然要更加容易。
残差函数  F(x):=H(x)-x。引入残差后的映射对输出的变化更加敏感，如H(5)=5.1,F(5)=0.1, 输出从5.1变化到5.2，增加的幅度为2%，但是残差是从0.1变化到0.2，增加幅度为100%。*残差的思想就是去除相同的主体部分，突出微小的变化*。  

### 计算公式  

<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{y}=F(\mathbf{x},{W_i})&plus;\mathbf{x}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{y}=F(\mathbf{x},{W_i})&plus;\mathbf{x}" title="\mathbf{y}=F(\mathbf{x},{W_i})+\mathbf{x}" /></a>
其中，<a href="https://www.codecogs.com/eqnedit.php?latex=F(\mathbf{x},{W_i})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F(\mathbf{x},{W_i})" title="F(\mathbf{x},{W_i})" /></a>是需要学习的残差映射，在计算的时候需要保证<a href="https://www.codecogs.com/eqnedit.php?latex=F(\mathbf{x},{W_i})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F(\mathbf{x},{W_i})" title="F(\mathbf{x},{W_i})" /></a>和<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{x}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{x}" title="\mathbf{x}" /></a>的维数一致，考虑到在两个feature map 中进行逐元素相加（element-wise addition)。当维数不一致的时候，通过引入线性映射来匹配维度<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{y}=F(\mathbf{x},{W_i})&plus;W_s\mathbf{x}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{y}=F(\mathbf{x},{W_i})&plus;W_s\mathbf{x}" title="\mathbf{y}=F(\mathbf{x},{W_i})+W_s\mathbf{x}" /></a>  

最终取得的效果是：
> these residual networks are easier to optimize, and can gain accuracy from considerably increased depth.

## Inception-Resnet
因为Residual connections在训练深度网络时，十分有优势，再加上inception网络的深度也比较深，所以作者尝试将两种方法结合起来。更确切的说是，将Inception中的filter concatenation中的一部分替换为residual connections的结构。结合了residual connection的inception模块如下所示：  

![2](https://pic1.zhimg.com/v2-492470e1687e1e547156ee90364b4f60_r.jpg)   

但是引入residual connection后，网络太深，稳定性不好，因此再次做了修改。  
![3](https://pic2.zhimg.com/80/v2-d65572749d1553f9ccf50bca659a6ffd_hd.jpg)  
图中inception框可以用任意其他subnetwork替代，但是这次修改是在输出后引入了缩放系数(scale),再相加和激活。
文章提出了两个版本：Inception-ResNet v1和Inception-ResNet v2，相比原来，加快训练的收敛速度。在图像识别，视频检测等领域都作为了base-network。
> In the experimental section we demonstrate that it is not very difficult to train competitive very deep networks without utilizing residual connections. However the use of residual connections seems to ***improve the training speed greatly***, which is alone a great argument for their use.  
