## GAN

<a href="https://www.codecogs.com/eqnedit.php?latex={min_{G}max_{D}}V(G,D)=E_{x&space;\sim&space;p_{data}}[logD(x)]&plus;E_{z&space;\sim&space;p_{z}}[log(1-D(z))]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?{min_{G}max_{D}}V(G,D)=E_{x&space;\sim&space;p_{data}}[logD(x)]&plus;E_{z&space;\sim&space;p_{z}}[log(1-D(z))]" title="{min_{G}max_{D}}V(G,D)=E_{x \sim p_{data}}[logD(x)]+E_{z \sim p_{z}}[log(1-D(z))]" /></a>

G：接收一个随机噪声（random noise)作为输入，通过噪声生成图片。G tries to make D(G(z))n near 1，尽可能生成真实的faked samples去欺骗判别网络；D tries to make D(G(z)) near 0, D尽可能把G生成的图片和真实的图片区分出来。
D越大，D(G(z))越小，V(D,G)越大；G越大，D(G(z))越大，V(D,G)越小

为什么G过程输入随机噪声 就可以生成目标图？  
🤔：G是要去学习dataset的概率分布情况。使用正态分布或者均匀分布的随机噪声Z作为generator net的输入是为了G(Z) 产生尽可能多的新的不同的值来保证大致能学到概率分布情况。我看条件gan的有篇论文里有一句酱紫的话「x: the labels fed as the input for conditional adversarial network。a map from x to y but produce deterministic outputs and fail to match any distribution 」这个时候它没有加额外的噪声，就最终不能很好的学习到分布了。

<a href="https://www.codecogs.com/eqnedit.php?latex={x&space;\sim&space;Pdata}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?{x&space;\sim&space;Pdata}" title="{x \sim Pdata}" /></a>  
表示 x来自真实样本的数据。

<a href="https://www.codecogs.com/eqnedit.php?latex=E{_{x&space;\sim&space;data}log(D(x))}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E{_{x&space;\sim&space;data}log(D(x))}" title="E{_{x \sim data}log(D(x))}" /></a>  
表示判别器D对来自真实样本的数据的期望。 

<a href="https://www.codecogs.com/eqnedit.php?latex=D_kL(p\left&space;|&space;\right&space;|q)=\sum_{i=1}^{N}p(x_i)(logp(x_i)-log(q(x_i)))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?D_kL(p\left&space;|&space;\right&space;|q)=\sum_{i=1}^{N}p(x_i)(logp(x_i)-log(q(x_i)))" title="D_kL(p\left | \right |q)=\sum_{i=1}^{N}p(x_i)(logp(x_i)-log(q(x_i)))" /></a>  

KL散度衡量了两个概率分布的近似程度，在GAN中我们使用JS散度「JS散度是在KL散度基础上」衡量生成的概率分布密度和原概率分布密度之间的近似程度   
 
<a href="https://www.codecogs.com/eqnedit.php?latex=D(x)=\cfrac{P_{data}}{P_{data}&plus;P_{g}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?D(x)=\cfrac{P_{data}}{P_{data}&plus;P_{g}}" title="D(x)=\cfrac{P_{data}}{P_{data}+P_{g}}" /></a>  

在收敛点附近时，P_g=P_data，D是一个局部准确的分类器。根据博弈的纳什平衡,在算法内部循环时，该循环会收敛到D(x)=1/2 即 D不能判别P_g和P_data时，训练停止。 


