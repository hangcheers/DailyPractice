### 前言
看Mask-RCNN的时候遇到了很多以前论文中的概念，特此在这次做个整合,方便更好的理解文章的概念。

### Faster-RCNN
[Faster-RCNN论文地址](https://arxiv.org/abs/1506.01497)  
首先来提Faster-RCNN网络结构，是因为Mask-RCNN是在其基础上改进网络结果「更具体点并列加一个mask branch」而得到的来实现segmentation。
Faster-RCNN在object detection中相当于**baseline system**，也是**benchmark**。主要包括了对目标物体的分类（classification），以及用候选框（bounding box）来对图片中的位置进行定位。在此之前也已有了Fast-RCNN之类的目标检测算法了。文章「Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks」的创新在于解决了Region Proposal生成开销问题。当生成的候选框过多时，processing speed会受到影响，从而没法很好的实现**real-time object detection**。
> we introduce a Region Proposal Network (RPN) that shares full-image convolutional features with the detection network, thus enabling nearly **cost-free** region proposals

在Faster-RCNN中使用RPN来进行候选框的确定，即「Region proposal Network找出物体可能存在的所有位置」，在这一个过程中找全，没有漏检很重要，不然后面的分类也没法分了。即Recall的值要高，「Recall=正确识别出来的object/数据库里含有的object，当recall=100%时，表示没有漏检」。RPN网络是一种全连接网络（FCN在下文有提到哈哈）

RPN预测了object bounds and objectness scores at each position，这给Fast-RCNN起到**类似指哪打哪**的作用了。此外，这里也是文章的另一个**创新点**，通过「sharing the convolutional features」实现了RPN和Fast-RCNN融合到一个网络中去了。  
在这里我们来理解一下「sharing」，RPN从feature map 上选择出了一系列的bounding box，然后Fast-RCNN再次利用了feature map，并用ROI pooling（*主要包括三步：1. 把 region proposal 分为n等分，n=the dimension of the output  2. 找到每个section最大的值  3.把每个最大的提取出来作为output buffer*）来对每个candidate box进行classification 和 bounding box regression，也在一定程度上节省了计算开销，加速了训练过程。
> using the recently popular terminology of neural networks with “attention” mechanisms, the RPN component tells the unified network where to look  
![Faster-RCNN](https://lilianweng.github.io/lil-log/assets/images/faster-RCNN.png)  



### Feature Pyramid Network
[FPN论文地址](https://arxiv.org/abs/1612.03144)

问题：  
ROI映射到某个feature map是将底层的坐标直接除以stride，显然对于小目标（size比较小）物体来说，到后面的卷积池化时，实际的语义信息就丢失了很多了。FPN解决的是多尺度检测的问题。 

结构：  
FPN利用了CNN层级特征的金字塔形式，同时生成在所有尺度上具有强语义信息的特征金字塔。FPN设计的金字塔结构包括了bottom-up & top-down & lateral connections(横向连接)三种结构。  
1. bottom-up是主干CNN沿前向传输（feed-foward/inference)的时候产生的一系列不同尺度的feature map。通常是每个阶段最深的层有strongest features。  
2. top-down是向上采样（upsampling)  
3. lateral connection帮助融合不同层的语义信息（即融合了bottom-up和top-down的语义信息），达到单尺度单张input，构建multiple scale的特征金字塔。
 此外，使用了1x1的卷积核来起到降低维度的作用。
 ![FPN](https://www.pytorchtutorial.com/wp-content/uploads/2018/08/1174793-20170612173455400-159085110.png)
**FPN在mask-RCNN中的用法**
从👆我们也已经看到从单一尺度的图像输入中，FPN可以获取multi-scale的特征图。在Mask-RCNN中作者采用ResNet-FPN作为主干的网络结构。原文是这么描述的。
> Using a ResNet-FPN backbone for feature extraction with Mask R-CNN gives excellent gains in both accuracy and speed
### Fully Convolutional Networks
[FCN论文地址](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) 

「Fully Convolutional Networks for semantic segmentation」一开始说的：
> combines semantic information from a 「deep , coarse」 layer with appearance information from a 「shallow, fine」 layers  

那为什么deep和coarse连在一起，shallow和fine来接在一起，不是越deep的层，越有表达力么？  

**全连接网络和CNN之间的区别**：  
经典的CNN是将卷积层产生的feature map使用全连接层映射为固定长度的特征向量，最后输出的是概率。
FCN将全连接层都变化为卷积层，「E.X.: 将4096 变成1x1x4096」是针对语义分割训练的一个end-to-end, pixel的网络，最后输出的是heatmap热力图。  

**FCN网络结构创新点**：
FCN可以接受任意尺寸的输入图像，采用反卷积层对最后一个卷积层的feature map做上采样upsampling，
使其恢复到输入图像的相同尺寸，从而对每一个像素都产生一个预测，同时保留原始输入图像的空间
信息。但是这样得到的结果比较coarser, 一些细节不能恢复。因此，作者采用了skip architecture来优化上采样，即将不同池化层的结果进行上采样，然后结合这些结果来优化输出。
「E.X 第五层的输出32倍放大反卷积到原图大小时比较粗糙，因此作者将第四层输出16倍放大，第3层输出8倍放大，可以从原论文中插图看到越低池化层，越精细」
因此我们也就可以理解了上文的问题。

**FCN在mask-RCNN中的应用**：
在the mask branch中，FCN被用在每个ROI中进行pixel-to-pixel的分割，这也是mask-RCNN超越了Faster-RCNN的地方。
作者在文章里是这么说的：
> Our method, called Mask-RCNN，extends Faster-RCNN by adding a branch for predicting segmentation masks on each Region of Interest,in parallel with the existing branch for classification and bounding box regression.

### Mask-RCNN
[Mask-RCNN论文地址](https://arxiv.org/abs/1703.06870)  

Mask-RCNN实现的任务要更「难」，因为不再是object detection 而是要达到instance segmentation，细化到区分类别中的不同实例。通俗点说，像素分类的话可以用不同的颜色来区别不同的实例，但是实例分割的时候即使是同一种类的物体，比如都是猫猫，也要区别出橘猫和加菲猫。像FCN中也可以用在实例分割的情景中，但它们的做法是，对每个像素进行multi-class categorization。
作者提出的方法在实例分割中是更有优势的。
> Instead, our method is based on parallel prediction of masks and class labels, which is **simpler and more flexible**.
> In contrast to the segmentation-first level of these methods, Mask R-CNN is based on an **instance first strategy**.  

在上面介绍faster-RCNN时，已经提到了Mask-RCNN增加了分支，来预测物体对应的掩膜(object mask).  

在阅读Mask-RCNN的时候，遇到一个问题「如何来理解**pixel-to-pixel alignment** 」
> we propose a simple, quantiazation-free layer, called *ROIAlign*， that preserves exact spatial locations.   

相比faster-RCNN的ROI Pooling, Mask-RCNN 用的方法是ROIAlign

