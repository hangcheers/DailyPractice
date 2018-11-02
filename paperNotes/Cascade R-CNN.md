## Bounding box regression
因为即使classifier识别出了物体所属的class，但是由于定位不准，导致生成框与标定框的IOU低于设定的阈值（比如0.5），还是无法正确的检测出物体，所以微调很有必要。
[RCNN](https://arxiv.org/abs/1311.2524)中的Appendix C里面详细的介绍了Bounding box是如何通过「回归」来提高localization的准确度。
> we demonstrate that a simple bounding-box regression method significantly reduces mislocalizations, whihc are the dominant 
error modes.   

首先我们面对回归问题，我们要知道这个问题是线性的。
当生成框Proposal box与标定框Ground Truth的距离比较近的时候，我们才能把其当作线性问题来解决。反之当PG距离比较远的时候，这个方法并不能work。
其次训练过程中的输入变量又是什么，期待得到一个什么样的结果。
> the primary difference between the two approaches is that here we regress from features computed by the CNN.   

根据这一句话我们可以知道是对CNN的特征向量进行回归处理。
按Appendix C的说法：
 由Proposal box P 和 Ground-truth box G 组成 training pairs  
 <a href="https://www.codecogs.com/eqnedit.php?latex={P^i,G^i}&space;(i=1,...,N)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?{P^i,G^i}&space;(i=1,...,N)" title="{P^i,G^i} (i=1,...,N)" /></a>   
 
 其中P和G可以通过四个位置坐标来表示，但是我们需要注意的是这时的坐标是图片像素级别「pixel」的坐标。
 <a href="https://www.codecogs.com/eqnedit.php?latex=P^i=(P^i_x,P^i_y,P^i_w,P^i_h),&space;G=(G_x,G_y,G_w,G_h)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P^i=(P^i_x,P^i_y,P^i_w,P^i_h),&space;G=(G_x,G_y,G_w,G_h)" title="P^i=(P^i_x,P^i_y,P^i_w,P^i_h), G=(G_x,G_y,G_w,G_h)" /></a>
 
