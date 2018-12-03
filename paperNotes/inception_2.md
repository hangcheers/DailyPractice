## Batch Normalization
### Internal covariate shift
"Internal"指的是神经网络的隐含层，"Covariate"指的是输入的权重参数化，“Internal Covariate Shift”指的是
在训练的过程中，输入的概率分布不固定，网络的参数在不断的变化，神经网络的隐含层也要不断的去「适应」新的分布。
这个现象会让模型更加难训练，我们也需要更加谨慎的初始化模型参数和学习率。因此作者引入了*Normalization* 来解决这个问题。 

### Normalization
通过规范化的手段，将每层神经网络任意神经元这个输入值的分布“强行拉回”到均值为0，方差为1的分布中.常规的正则化公式为：<a href="https://www.codecogs.com/eqnedit.php?latex=$$\hat{x}^{k}=\frac{x^{k}-E(x^{k})}{\sqrt{var(x^{k})}}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\hat{x}^{k}=\frac{x^{k}-E(x^{k})}{\sqrt{var(x^{k})}}$$" title="$$\hat{x}^{k}=\frac{x^{k}-E(x^{k})}{\sqrt{var(x^{k})}}$$" /></a>
这么做的优点是可以加快收敛的速度，但是缺点是假如该层的各个特征互不相关，简单的正则化操作可能会改变该层的特征表达。
为了确保神经网络里面可以进行恒等变换(identity transform)，我们需要对常规的正则化公式进行改变。由此我们也引入了Batch Normalization:<a href="https://www.codecogs.com/eqnedit.php?latex=$$\mathrm{BN_{\gamma&space;,\beta&space;}}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\mathrm{BN_{\gamma&space;,\beta&space;}}$$" title="$$\mathrm{BN_{\gamma ,\beta }}$$" /></a> 。  <a href="https://www.codecogs.com/eqnedit.php?latex=$$y^{k}=\gamma^{k}\hat{x^{k}}&plus;\beta^{k}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$y^{k}=\gamma^{k}\hat{x^{k}}&plus;\beta^{k}$$" title="$$y^{k}=\gamma^{k}\hat{x^{k}}+\beta^{k}$$" /></a>
其中每一个activation都会引入两个超参数<a href="https://www.codecogs.com/eqnedit.php?latex=$$\mathbf{\gamma&space;,\beta}&space;$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\mathbf{\gamma&space;,\beta}&space;$$" title="$$\mathbf{\gamma ,\beta} $$" /></a> 。这两个参数也是神经网络需要学习的参数，分别起到的是scale和shift的作用。相当于在原来正则化的基础上，再进行了一次线性变化。在mini-batch的训练过程中，BN是规范化了每一层的输入，z=g(BN(W,u))  「g表示的非线性操作，例如：relu」，从而减少了Internal Covariate shift的干扰。
> the inputs to each layer are affected by the parameters of all preceding layers - so that small
changes to the network parameters amplify as the network becomes deeper …… Fixed distribution of inputs to a 
sub-network would have a positive consequences for the layers outside the network, as well.  此外，BN的创新也在于
applied to the sub-networks and layers, incorporate the normalization in the network architecture as well。  

Batch Normalization不仅允许不那么谨慎的初始化，还允许使用更高的learning rate。


## Rethinking the Inception Architecture for Computer vision
因为计算开销、参数量限制了把Inception部署到移动端和一些场景中，在abstract里，作者指出了对网络构架进行改进的思路。
> Here we are exploring ways to scale up networks in ways that aim at utilizing the added computation as efficiently as possible by **factorized convolutions and aggressive regularization**.
### 卷积核的因式分解
Paper里探索了几张将较大的卷积核分解为较小的卷积核的设置方式。例如：5x5的卷积核替换为两个3x3的卷积核；3x3的卷积核替换为1x3和3x1的卷积核。这样具有相同的receptive field的同时可以大大的减小计算开销。

![1](https://tse2.mm.bing.net/th?id=OIP.lEHUl5w_rweJSb-FpNdKeAHaFl&pid=Api)  

![2](http://davidstutz.de/wordpress/wp-content/uploads/2017/03/inception_arch_3.png)  

需要注意的是「nxn的卷积被替换为1xn和nx1的卷积」的这种空间上分解为非对称卷积的做法在前面几层layer的效果不是很好，更适用于中等规格大小的feature map，（m的范围从12到20）。
此外我们还要思考🤔几个问题。  
1.用小卷积核替换大卷积核，是否会带来信息损失(loss of expressiveness)? 不会，只要多次叠加的小卷积核和开始的大卷积核具有相同的receptive field。  
2.如果我们的目标是对计算开销中的线性部分进行因式分解，那么为什么不直接在第一次保持线性激活（linear activation）？因为在实验中表明，非线性激活性能更好。
 

### Label Smoothing Regularization
因为大多数的数据集都存在错误的标签， 但是minimize the cost function on the wrong labels can be harmful。因此在Model Regularization中，可以通过在训练的过程中主动加入噪声作为penalty，这样的模型具有noise Robustness。Label Smoothing Regularization(LSR)是其中的一种regularization的方法。
> Here we propose a mechanism to regularize the classifier layer by estimating the marginalized effect of label-dropout during
training.   

{*举一个University of Waterloo的WAVE LAB的 ME 780中lecture 3：Regularization for deep models的例子来帮助理解*：  
*ground-truth:*   y1_label=[1,0,0,……，0]  
*prediction:*   经过softmax classifier得到的softmax output:  y1_out=[0.87,0.001,0.04……,0.03]. }  
> maximum likelihood learning with softmax classifier and hard targets may actually never converge, the softmax can 
never predict a probability of exactly 0 or 1, so it will continue to learn larger and larger weights, making more 
extreme predictions.  


假设x为training example，p(k|x)为x属于「label k」的概率，q(k|x)为x属于「ground-truth label」的概率。为了方便起见，忽略了p和q在example x上的相关性。

目标函数：「最小化交叉熵」。因为交叉熵衡量的是两个分布（p和q）的相似性，最小化目标函数是为了让预测的label概率分布p(k|x)（即例如上面的softmax的输出）和ground-truth label的概率分布q(k|x)尽可能的接近。「最小化交叉熵」也等价为「最大化似然函数」。但是我们需要对这个目标函数进行了改进。因为在单类情况下，单一的交叉熵导致样本属于某个类别的概率非常大，模型太过与自信自己的判断。这样会导致过拟合，此外还会降低模型的适应能力。为了避免模型过于自信，引入了一个独立于样本分布的变量u(k)，这相当于在ground-truth distribution中加入了噪声，组成一个新的分布。在实验中，使用的是均匀分布(uniform distribution)代替了u(k)
> we propose a mechanism for encouraging the model to be less confident. While this may not be desired if the goal is to maximize the log-likelihood of training labels, it does regularize the model and makes it more adaptable …… we refer to this
change in ground-truth label distribution as label-smoothing regularization, or LSR.
