Inception系列有四篇重要的paper，分别是：[Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)、
[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shif](https://arxiv.org/abs/1502.03167)、[Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)、[Inception-v4](https://arxiv.org/abs/1602.07261)
在此，依次阅读并做笔记。
## GoogleNet  
[Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842) 首次提出了「**Inception**」模块作为网络构架，
该网络构架也是后续作为classification和detection的baseline。
> we introduce a new level of organization in the form of the "Inception module" and also in a more direct sense of increased
network depth.  

适当的「加深网络depth」、谨慎的「调节超参数」可以在recognition和object detection取得更好的效果。  
但是网络的size过大，会直接影响运行的性能。就像一个人过胖，会直接影响身体健康。
网络的size主要包括两方面：depth - the number of levels - of the network 和 width - the number of units at each level。  

当网络的size过大的时候，参数#paramters过多，在训练的过程中更容易出现过拟合的现象，特别是在labeled examples很有限的情况下。
同时网络的size越大，消耗的计算资源就越多。很重要的提示如下：  
> Since in practice the computational budget is always finite, an efficient distribution of computing resources is preferred
to an indiscriminate increase of size。  

在这里我们就要结合「数据结构」&「网络结构」来考虑，如何更好的利用计算资源。 

将Fully Connected变为稀疏连接（sparse connection）的时候，可以在增加网络深度和宽度的同时减少参数个数，
但是大部分的硬件是针对密集矩阵计算优化的，稀疏矩阵虽然数据
量变少，但计算所消耗的时间很难减少。GoogleNet希望做的就是既保证网络结构的稀疏性、又利用密集矩阵的高计算性能。
提出新的「ideas & algorithm & network architectures」是一件很有创新性的事情，现在来简单介绍一下GoogleNet的network architecture。  

GoogleNet的核心是Inception module，而Inception相当于一个Convolutional building block，也是一个局部稀疏最优解的网络构架，然后我们在
空间上做堆叠。下面我们结合论文的插图来仔细分析一下Inception module。
> Our network will be built from convolutional building blocks.
All we need is to find the **optimal local construction** and to repeat it spatially.    
![1](https://qph.fs.quoracdn.net/main-qimg-6833c13a23c33584be86a30bae210cf9)  

输入有四个分支，使用多个尺度（1x1或3x3或5x5）的卷积和池化进行特征提取「相当于将稀疏矩阵分解为密集矩阵」，每一尺度提取的特征是均匀分布的，
但是经过「filter concatenation」这步操作后，输出的特征不再是均匀分布的，相关性强的特征会被加强，而相关性弱的特征会被弱化。
「filter concatenation」，这一步其实相当于在特征维度/沿着深度方向进行拼接，从而输出 a single output vector forming the input of 
next stage.得到全新的网络结构。结合code来加深一下理解
> 
    concatenated_tensor = tf.concat(3,[branch1, branch2, branch3, branch 4])  

Inception module 借鉴了NIN(Network In Network)，引入1x1的卷积操作，进行了网络结构的改进。更具体的来说，在3x3、5x5卷积操作前
使用1x1的卷积，起到降维的作用，显著降低了计算量。


