## RCNN
RCNN 是开创性的将Classification和Object detection问题连接起来的文章。发现以前很多概念没弄太清楚，都列在下面，加以巩固。  
RCNN中的R表示的是Region，RCNN的全名应该叫做Regions with CNN features. 


**Bounding box regression**  

因为即使classifier识别出了物体所属的class，但是由于定位不准，导致生成框与标定框的IOU低于设定的阈值（比如0.5），还是无法正确的检测出物体，所以微调很有必要。
[RCNN](https://arxiv.org/abs/1311.2524)中的Appendix C里面详细的介绍了Proposal是如何通过「回归」来逼近Ground Truth 从而提高localization的准确度。
> These proposals define the set of *candidate detection* available to our detector..... we demonstrate that a simple bounding-box regression method significantly reduces mislocalizations, whihc are the dominant 
error modes.  

首先我们面对回归问题，我们要知道这个问题是线性的。当生成框Proposal box与标定框Ground Truth的距离比较近的时候，我们才能把其当作线性问题来解决。反之当PG距离比较远的时候，这个方法并不能work。
> we train a linear regression model to predict a new detection window given the pool5 features for a selective search region
proposal.   

线性回归是对自变量和因变量之间关系进行建模的一种回归分析，当其目标是预测结果尽可能的去拟合y，也就是需要拟合出一个预测模型，对于新增的X值，可以用这个模型来预测一个y值,在这种情况下通常用最常见的Least square作为loss function。在本文中，作者也是选择了构建一个线性回归模型来预测新的bounding box。
其次训练过程中的输入变量又是什么？这个也是**创新**之处，从*CNN最后一个pooling层即Pool5*的获取的特征来训练线性回归模型来预测detection window
> Our goal is to learn a **transformation** that maps a proposed box P to a ground-truth box G.

按Appendix C的说法：
 由Proposal box P 和 Ground-truth box G 组成 training pairs ，
 <a href="https://www.codecogs.com/eqnedit.php?latex={P^i,G^i}&space;(i=1,...,N)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?{P^i,G^i}&space;(i=1,...,N)" title="{P^i,G^i} (i=1,...,N)" /></a>   在线性回归模型中分别相当于X和y
 
 其中P和G可以通过四个位置坐标来表示，其中x,y是bounding box的中心点坐标，w,h是bounding box的宽和高。但是我们需要注意的是这时的坐标是图片像素级别「pixel」的坐标。
 <a href="https://www.codecogs.com/eqnedit.php?latex=P^i=(P^i_x,P^i_y,P^i_w,P^i_h),&space;G=(G_x,G_y,G_w,G_h)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P^i=(P^i_x,P^i_y,P^i_w,P^i_h),&space;G=(G_x,G_y,G_w,G_h)" title="P^i=(P^i_x,P^i_y,P^i_w,P^i_h), G=(G_x,G_y,G_w,G_h)" /></a>
 
 从P到G的映射需要两步骤：1.平移 2.尺度缩放。👇是作者所给出的公式，我们对照公式加以解释。  
 <a href="https://www.codecogs.com/eqnedit.php?latex=\widehat{G_x}=P_wd_x(P)&plus;P_x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\widehat{G_x}=P_wd_x(P)&plus;P_x" title="\widehat{G_x}=P_wd_x(P)+P_x" /></a>  其中的<a href="https://www.codecogs.com/eqnedit.php?latex=P_wd_x(P)=\Delta&space;x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P_wd_x(P)=\Delta&space;x" title="P_wd_x(P)=\Delta x" /></a>  可以理解为x方向上的移动，同理也可以得到y方向的移动。 
 <a href="https://www.codecogs.com/eqnedit.php?latex=\widehat{G_w}=P_w(exp(d_w(P)))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\widehat{G_w}=P_w(exp(d_w(P)))" title="\widehat{G_w}=P_w(exp(d_w(P)))" /></a> 可以理解为宽度方向上的缩放，同理也可以得到高度方向上的缩放。因为缩放因子永远是正数，所以这里用的是指数形式。
 
我们注意到上面都出现了d_x(P) 这个也就是线性回归模型所派上用场的地方了，这四个变换是我们需要通过模型来学习的。  
<a href="https://www.codecogs.com/eqnedit.php?latex=d_*(P)=w_*^T\Phi&space;_5(P)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?d_*(P)=w_*^T\Phi&space;_5(P)" title="d_*(P)=w_*^T\Phi _5(P)" /></a>   
其中<a href="https://www.codecogs.com/eqnedit.php?latex=\Phi&space;_5(P)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Phi&space;_5(P)" title="\Phi _5(P)" /></a>是输入Proposal的特征向量，<a href="https://www.codecogs.com/eqnedit.php?latex=w_*" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w_*" title="w_*" /></a>表示的是需要学习的参数。*其中每一个变换都对应一个目标函数*，最终得到的<a href="https://www.codecogs.com/eqnedit.php?latex=d_*(P)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?d_*(P)" title="d_*(P)" /></a>是我们得到的预测值

但是真正的「平移量」*t_x,t_y*和「缩放量」*t_w,t_h*是需要从P和G上求出来的，从<a href="https://www.codecogs.com/eqnedit.php?latex=d_x(P),d_y(P),d_w(P),d_h(P)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?d_x(P),d_y(P),d_w(P),d_h(P)" title="d_x(P),d_y(P),d_w(P),d_h(P)" /></a>得到的是预测<a href="https://www.codecogs.com/eqnedit.php?latex=\widehat{G}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\widehat{G}" title="\widehat{G}" /></a>
在这里我们需要通过Loss function来使得预测值（即预测的平移量+缩放量）和真实值（即真实的平移量+缩放量）的差距缩小，文章是通过最小二乘法进行目标函数优化。最终，bbox regression的好处就是提高了mAP 3%~5%

**仿射变换**  

参考[马同学高等数学](https://www.matongxue.com/madocs/244.html) 里的回答了解到，仿射变换（Affine Transformation）相当于：线性变换+平移。直线经过仿射变换仍然为直线，直线之间的平行线保持不必，点的位置顺序也不会变换。但坐标系的原点发生了变换。
在图像处理中，二维图像的平移、缩放、拉伸、旋转、扭曲、收缩等操作是常见的图像几何变换。在R-CNN中作者是通过该操作来得到固定大小的input size。
> we use a simple technique affine image warping to compute a fixed-size CNN input from each region proposal, regardless of the region's shape.  

**non-maximum suppression**
> the only class-specific computations are a reasonably small matrix-vector product and greedy non-maximum suppression.  

根据[Coursera](https://zh.coursera.org/lecture/convolutional-neural-networks/non-max-suppression-dvrjH)的video，对non-maximum suppression做以下总结：  算法可能会对一个物体生成multiple detections，但是我们只保留置信度最高的预测框即maximum probability此外抑制其他低概率的预测框。
> 1. discard all the predictions of the bounding boxes with Pc less than or equal to some threshold, assume it is 0.6  
> 2. while there are any remaining boxes, pick the box with the largest Pc then output as a prediction.  
> 3. discard any remaining box with IOU >=0.5 with the box output in the previous step.   

从上面的流程中我们也可以知道nms需要设定两个阈值，此外因为nms一次只会去处理一个类别，如果有N个类别时，nms需要执行N次。  

**评价指标**  

在object detection中，经常用mAP(mean Average Precision)来衡量算法的效果。
首先我们要来区分TP(True Positive) & FP(False Positive) & TN(True Negative) & FN(False Negative)  

![1](https://pic4.zhimg.com/80/v2-761706f5b1fe36873ba1bb20c7d1d447_hd.jpg)  



Precison=TP/(TP+FP)，其中TP+FP为总的检测数据，Precision是候选框框出来的有多少是正确的，举个例子对pedestrian进行检测，FP指的是people detected are not really pedestrians，而TP指的是把一开始标注为Ground Truth=1的pedestrian给正确的检测出来了。  
Recall=TP/(TP+FN)，其中TP+FN为Ground Truth=1的数据。在以Precision为纵坐标，Recall为横坐标的坐标系中，Precision会随着Recall的增加总体上呈现下降趋势。而AP衡量的是学习出的模型在单个类别上的好坏，mAP是AP取平均值，衡量的是学习出的模型在所有类别上的好坏。具体的计算方法可以从[代码片段](https://github.com/facebookresearch/Detectron/blob/05d04d3a024f0991339de45872d02f2f50669b3d/lib/datasets/voc_eval.py#L54)更好的理解。

![cv](https://img-blog.csdn.net/20160816132136353)
