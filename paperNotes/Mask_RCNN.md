### 前言
看Mask-RCNN的时候遇到了很多以前论文中的概念，特此在这次做个整合,方便更好的理解文章的概念。

### Fully Convolutional Networks
问题：  
作者在文章一开始说的：combines semantic information from a 「deep , coarse」 layer with appearance information from a 「shallow, fine」 layers，那为什么deep和coarse连在一起，shallow和fine来接在一起，不是越deep的层，越有表达力么？

首先来看全连接网络和CNN之间的区别：  
经典的CNN是将卷积层产生的feature map使用全连接层映射为固定长度的特征向量，最后输出的是概率。
FCN将全连接层都变化为卷积层，「E.X.: 将4096 变成1x1x4096」是针对语义分割训练的一个end-to-end, pixel的网络，最后输出的是heatmap热力图。  

文章的创新点在于：
FCN可以接受任意尺寸的输入图像，采用反卷积层对最后一个卷积层的feature map做上采样upsampling，
使其恢复到输入图像的相同尺寸，从而对每一个像素都产生一个预测，同时保留原始输入图像的空间
信息。但是这样得到的结果比较coarser, 一些细节不能恢复。因此，作者采用了skip architecture来优化上采样，即将不同池化层的结果进行上采样，然后结合这些结果来优化输出。
「E.X 第五层的输出32倍放大反卷积到原图大小时比较粗糙，因此作者将第四层输出16倍放大，第3层输出8倍放大，可以从原论文中插图看到越低池化层，越精细」
因此我们也就可以理解了上文的问题。

### Faster-RCNN
步骤：
1. Region proposal 找出物体可能存在的所有位置（all the possible places),  所以这个过程的输出应该是一系列的物体可能出现的（用bounding box框出来）
在这一个过程中high recalls很重要，不然后面的分类也没法分了。  
  PS: Recall=正确识别出来的object/数据库里含有的object，当recall=100%时，表示没有漏检)
但是也存在如果ROI过多时，会影响到processing speed，从而影响到real-time object detection。
此外，RPN网络是一种全连接网络，（FCN已经在上面提到过了哈哈）
2. classification & bounding box regression  
这个过程的输出应该是1. class label 2. bounding box offset(边界框抵消值) 

ROI pooling (Region of Interest pooling)
主要分为三步：
1. 把 region proposal 分为n等分，n=the dimension of the output  2. 找到每个section最大的值  3.把每个最大的提取出来作为output buffer。其主要的优点在于：再一次用了CNN产生的feature map，并且加速了训练/测试的过程。

### Feature Pyramid Network
问题：  
ROI映射到某个feature map是将底层的坐标直接除以stride，显然对于小目标（size比较小）物体来说，到后面的卷积池化时，实际的语义信息就丢失了很多了。FPN解决的是多尺度检测的问题。 

结构：  
FPN利用了CNN层级特征的金字塔形式，同时生成在所有尺度上具有强语义信息的特征金字塔。FPN设计的金字塔结构包括了bottom-up & top-down & lateral connections(横向连接)三种结构。  
1. bottom-up是主干CNN沿前向传输（feed-foward/inference)的时候产生的一系列不同尺度的feature map。通常是每个阶段最深的层有strongest features。  
2. top-down是向上采样（upsampling)  
3. lateral connection帮助融合不同层的语义信息（即融合了bottom-up和top-down的语义信息），达到单尺度单张input，构建multiple scale的特征金字塔。
 此外，使用了1x1的卷积核来起到降低维度的作用。


