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
