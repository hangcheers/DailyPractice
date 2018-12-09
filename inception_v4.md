## 前言
Google Research推出的Inception module历经了四次“迭代”，GoogleNet也被成为Inception-v1，加入了batch Normalization的inception 
module成为inception-v2，加入了factorization的称为inception-v3,在和Microsoft Research推出的Resnet 相结合得到了inception-v4。
## Resnet
### introduction
不能简单的增加网络的深度，否则网络深度越深，training error和test error越高。
Kaiming He提出了深度残差学习（deep residual learning framework）,通过网络结构的创新来有效的解决了上述的梯度弥散现象(degradation)。
>1. our extremely deep residual nets are easy to optimize.    
>2. our deep residual nets can easily enjoy accuracy gains from greatly increased depth.
- ***residual representations***  
残差函数  F(x):=H(x)-x
- ***shortcut connection***   

> if the added layers can be constructed as identity mappings, a deeper model should have
training error no greater than its shallower counterpart.
