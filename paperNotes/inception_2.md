## Batch Normalization
### Internal covariate shift
"Internal"指的是神经网络的隐含层，"Covariate"指的是输入的权重参数化，“Internal Covariate Shift”指的是
在训练的过程中，输入的概率分布不固定，网络的参数在不断的变化，神经网络的隐含层也要不断的去「适应」新的分布。
这个现象会让模型更加难训练，我们也需要更加谨慎的初始化模型参数和学习率。因此作者引入了*Normalization* 来解决这个问题。 

BN的基本思想就是：让每个隐层节点的激活输入分布固定下来，通过规范化的手段，将每层神经网络任意
神经元这个输入值的分布“强行拉回”到均值为0，方差为1的分布中。需要「固定」参数的原因，在paper里是这么交代的。
> the inputs to each layer are affected by the parameters of all preceding layers - so that small
changes to the network parameters amplify as the network becomes deeper …… Fixed distribution of inputs to a 
sub-network would have a positive consequences for the layers outside the network, as well.

经过BN后，大部分的activation的值就会落入非线性函数的「线性区域」即「导数非饱和区域」，
这样就可以避免进入梯度饱和区域(即*梯度变化较小的区域*)，这样的话，训练时就可以加快收敛速度。

## Rethinking the Inception Architecture for Computer vision
### Label Smoothing Regularization
因为大多数的数据集都存在wrong labels， 但是minimize the cost function on the wrong labels can be harmful。因此在Model Regularization中，可以通过在训练的过程中主动加入噪声作为penalty，从而模型具有noise Robustness。Label Smoothing Regularization(LSR)是其中的一种方法。
> Here we propose a mechanism to regularize the classifier layer by estimating the marginalized effect of label-dropout during
training.   

{*举一个University of Waterloo的WAVE LAB的 ME 780中lecture 3：Regularization for deep models的例子来帮助理解*：  
*ground-truth:*   y1_label=[1,0,0,……，0]  
*prediction:*   经过softmax classifier得到的softmax output:  y1_out=[0.87,0.001,0.04……,0.03]. }  
> maximum likelihood learning with softmax classifier and hard targets may actually never converge, the softmax can 
never predict a probability of exactly 0 or 1, so it will continue to learn larger and larger weights, making more 
extreme predictions.  


假设x为training example，p(k|x)为x属于「label k」的概率，q(k|x)为x属于「ground-truth label」的概率。为了方便起见，忽略了p和q在example x上的相关性。
> consider the ground-truth distribution over labels q(k|x) for this training example .....   

作者取交叉熵cross entropy作为了目标函数。因为交叉熵是衡量两个分布p和q的相似性，最小化目标函数是为了让预测的label概率分布p(k|x)（即例如上面的softmax的输出）和ground-truth label的概率分布q(k|x)尽可能的接近。



