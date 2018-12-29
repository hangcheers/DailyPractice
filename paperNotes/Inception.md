Inception系列有四篇重要的paper，分别是：[Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)、
[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shif](https://arxiv.org/abs/1502.03167)、[Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)、[Inception-v4](https://arxiv.org/abs/1602.07261)
在此，依次阅读并做笔记。
## GoogleNet
### Introduction & Motivation
[Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842) 首次提出了「**Inception**」模块作为网络构架，
该网络构架也是后续作为classification和detection的base network的重要组成部分。
> we introduce a new level of organization in the form of the "Inception module" and also in a more direct sense of increased
network depth.  

网络的size主要从两方面进行考虑：depth - the number of levels - of the network 和 width - the number of units at each level。「加深网络depth」、「调节超参数」可以在recognition和object detection取得更好的效果。但是网络的size过大，会直接影响运行的性能。就像一个人过胖，会直接影响身体健康。当网络的size过大的时候，参数#paramters过多，消耗的计算资源就越多，此外，特别是在labeled examples很有限的情况下，更容易出现overfitting。一般是采用「dropout」或者「regularization」，并且「调整超参数」和「设置学习率」去防止训练过程中过拟合现象的出现。
> For larger datasets such as Imagenet, deeper architectures are used to get better results and dropout is used to prevent
overfitting …… Since in practice the computational budget is always finite, an efficient distribution of computing resources is preferred to an indiscriminate increase of size。  

### Inception module
上面的方法挺好的，但也挺蛮烦的，所以作者试图结合「数据结构、网络结构」来考虑，如何设计一个创新性的architecture，来更好的利用计算资源以及稍微放心、大胆的设置参数一些?

首先，开个小分支，介绍一下「稀疏结构」的理论基础：**Hebbian原理**。 作者从neuroscience的角度得到了启发，提出了网络结构的创新：。
该原理指出：各个神经元是组合效应，通过神经突触进行信息的传递，大脑皮层接收信息。此外，
神经反射活动的持续与重复会导致神经元连接稳定性的持久提升，当两个神经元细胞A和B距离很近，并且A参与了对B重复、持续的兴奋，那么某些代谢变化会导致A将作为能使B兴奋的细胞。
> neurons that fire together, wire together.
将Fully Connected变为稀疏连接（sparse connection）的时候，可以在增加网络深度和宽度的同时减少参数个数，
但是大部分的硬件是针对密集矩阵计算优化的，稀疏矩阵虽然数据量变少，但计算所消耗的时间很难减少。
GoogleNet希望做的就是既保证网络结构的稀疏性、又利用密集矩阵的高计算性能。

GoogleNet的核心是Inception module，而Inception相当于一个Convolutional building block，也是一个局部稀疏最优解的网络构架，然后我们在
空间上做堆叠。下面我们结合论文的插图来仔细分析一下Inception module。图a是原始的Inception module，图b是借鉴了NIN（Network In Network）
引入1x1的卷积操作，改进后的Inception module。
> Our network will be built from convolutional building blocks.
All we need is to find the **optimal local construction** and to repeat it spatially.   

![1](https://cdn-images-1.medium.com/max/2000/1*aq4tcBl9t5Z36kTDeZSOHA.png)  

输入有四个分支，使用多个尺度（1x1或3x3或5x5）的卷积和池化进行特征提取「相当于将稀疏矩阵分解为密集矩阵」，每一尺度提取的特征是均匀分布的，
但是经过「filter concatenation」这步操作后，输出的特征不再是均匀分布的，**相关性强的特征会被加强，而相关性弱的特征会被弱化**。*这个相关性高的节点应该被连接在一起的结论，即是从神经网络的角度对Hebbian原理有效性的证明*
「filter concatenation」，这一步其实相当于沿着深度方向（或者说在depth这个维度）进行拼接，
> stack up the first volume to the second volume to make the dimensions match up …… Output a single output vector forming the input of 
next stage。  

结合[Udacity视频](https://becominghuman.ai/understanding-and-coding-inception-module-in-keras-eb56e9056b4b)和code来加深一下对「filter concatenation」的理解
> 
    concatenated_tensor = tf.concat(3,[branch1, branch2, branch3, branch 4])  


**E.g:**   
{General}：输入 28x28x192 volume ，并列经过 1x1卷积操作、3x3卷积操作、5x5卷积操作、max-pool，分别得到28x28x64、28x28x128、
28x28x32、28x28x32 volume, 将并列的volume沿着深度方向进行拼接，输出 28x28x256 volume。 
> Feifei-Li的cs231n的课件里是描述CNN的：every layer of a ConvNet transforms one volume of activations to another through a differentiable function.We use three main types of layers to build ConvNet architectures:Convolutional Layer, Pooling Layer,
and Fully-Connected Layer.  Conv layer will compute **the output of neurons** that are connected to local regions in the input,
each computing a **dot product between their weights** and a small region they are connected to the input volume.
Pool layer will perform a downsampling operation along **the spatial dimensions**(*width,height*)
FC layer will compute the class score,resulting in volume of size「1x1x#class」。

{Specific}：5x5的卷积操作得到了28x28x32的block。
filter size =5x5x192，5 pixels width and height, 192 pixels depth（filter的深度需要和*前一feature map的深度*保持一致。）


设input volume width = W,  the width of receptive field = F_w, zero padding on the border = P, stride = S
那么output volume width = (W-F+2P)/S+1。同理也可以得到output volume height。此外, input volume depth = D1
此外，被filter覆盖的图像区域称为receptive field，具体操作是：slide each filter across the width and height of the input volume and compute dot products between the entries of the filter and the input at any position，即filter中的值和原始图像中receptive field中的像素值进行点积运算，产生activation map或feature map。**图像一般都是局部相关的**，
第n+1层的每个神经元和第n层的receptive field中的神经元连接，而不需要和第n层的所有神经元连接，ConvNet具有**local connectivity(局部连接)** 的性质。当filter的receptive field越大，filter能够处理的原始输入内容的范围就越大。随着经过更多的卷积层，得到的激活映射也就具有更为复杂的特征。  

![4](http://cs231n.github.io/assets/cnn/depthcol.jpeg)

设 number of filters = K, 也是output volume depth的值。当filter的数目越多，spatial dimensions就会保留的越好。
CNN具有local connection和parameter sharing的特点。
每个filter的权重的个数 = F_w x F_h x D1, 总的权重个数= F_w x F_h x D1 x K

我们再分析一下**compution cost**
> cs231n 指出： the largest bottleneck to be aware of when constructing the ConvNet is the memory bottle neck.
we need to keep track of the intermediate volume size, the paramter size and the memory.
[Reference:cs231n](http://cs231n.github.io/convolutional-networks/#conv)  

现在我们来分析一下，上面的图b相比图a的优势在哪里🧐。  
1x1的卷积是作为瓶颈层的作用，用很小的计算量可以增加一层特征变换和非线性变换。*此外，一般涉及到改变通道数，都会使用1x1卷积操作，
例如残差连接和Dense连接*  
> the bottleneck is usually the smallest part of something
我们来计算一下图a中5x5的卷积操作得到了28x28x32的block的时候，所需要的multiples的次数。以及图b中先使用1x1的卷积操作先得到28x28x16，再使用5x5的卷积操作得到了28x28x32的blcok的时候，所需要的multiples的次数。  


1.图a (28x28x32) x (5x5x192) = 120million 「一个output volume所需要的乘积次数 x the number of output values」 

2.图b （28x28x16) x (1x1x192) + (28x28x32) x (5x5x16) = 12.4 million  

从上面👆两个对比可以知道1x1的卷积操作大大的减少了计算量。


### GoogleNet's architecture
首先，为了有一个初步的印象，先截取了GoogleNet的一部分，
![2](https://mohitjainweb.files.wordpress.com/2018/06/googlenet-architecture-showing-the-side-connection.png?w=700)  
我们可以注意到这里有一个「softmax」的分支，整个结构中有两个「softmax」，它相当于辅助分类器，结合code我们可以知道该操作是将中间某一层的输出用作分类，并按一个较小的权重（0.3）加到最终分类结果中起到的是梯度前向传输的作用。论文里是这么交代的：
> By adding auxiliary classifiers connected to these intermediate layers, we would expect to encourage discrimination in the lower stages in the classifier, increase the gradient signal that gets propagated back, and provide additional regularization.  

其次，为了对GoogleNet的组成有一个概念，引用了论文中的表格：
![3](https://img-blog.csdn.net/20170612110458444?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbWFyc2poYW8=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
上面讨论时，已经说过GoogleNet是模块化的，堆叠了多个Inception Module，靠后的Inception Module能够抽取更高阶的抽象的特征。

