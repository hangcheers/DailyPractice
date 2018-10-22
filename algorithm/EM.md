### EM与极大似然估计的关系
似然函数是找到一组模型参数使得序列的概率最大，当似然函数最大时，模型参数就最合理。  <a href="https://www.codecogs.com/eqnedit.php?latex=$$\Theta&space;$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\Theta&space;$$" title="$$\Theta $$" /></a> 反映了模型参数，例如在正态分布中，模型参数指的是mean, std

<a href="https://www.codecogs.com/eqnedit.php?latex=$$\Theta&space;^{*}=argmax_{\Theta&space;}&space;\sum_{x}logP(x;\Theta&space;)$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\Theta&space;^{*}=argmax_{\Theta&space;}&space;\sum_{x}logP(x;\Theta&space;)$$" title="$$\Theta ^{*}=argmax_{\Theta } \sum_{x}logP(x;\Theta )$$" /></a>  


EM含有隐变量z。

<a href="https://www.codecogs.com/eqnedit.php?latex=$$\Theta&space;^{*}=argmax_{\Theta&space;}&space;\sum_{x}logP(z|\Theta&space;)\cdot&space;P(x|z;\Theta&space;)$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\Theta&space;^{*}=argmax_{\Theta&space;}&space;\sum_{x}logP(z|\Theta&space;)\cdot&space;P(x|z;\Theta&space;)$$" title="$$\Theta ^{*}=argmax_{\Theta } \sum_{x}logP(z|\Theta )\cdot P(x|z;\Theta )$$" /></a>  

<a href="https://www.codecogs.com/eqnedit.php?latex=$$p(x,z;\Theta&space;)$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$p(x,z;\Theta&space;)$$" title="$$p(x,z;\Theta )$$" /></a> 表示x,z的联合分布，当x理解为定值时，该联合分布可以看为
z的随机变量函数。 <a href="https://www.codecogs.com/eqnedit.php?latex=$$z\epsilon&space;{z_1,z_2...z_k}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$z\epsilon&space;{z_1,z_2...z_k}$$" title="$$z\epsilon {z_1,z_2...z_k}$$" /></a>表示隐变量。 


### 步骤  
<a href="https://www.codecogs.com/eqnedit.php?latex=$$\Theta&space;_{n}\rightarrow\Theta&space;_{n&plus;1}&space;$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\Theta&space;_{n}\rightarrow\Theta&space;_{n&plus;1}&space;$$" title="$$\Theta _{n}\rightarrow\Theta _{n+1} $$" /></a>  
从当前的模型参数，迭代得到下一次的模型参数。通常分为E-step和M-step。
笼统来说，根据初始的模型参数，把missing data猜测出来后得到一些猜出来的数据，更新假设，让观察到的数据更加有可能不断迭代直至收敛，最后得到一个可以解释数据的假设。
注意猜测的时候要尽可能的涵盖所有情况，然后求期望。收敛的条件是由阈值或者迭代次数来决定的。此外，EM算法受到初值的影响，不一定保证可以得到全局最优解，因此在实际运用时可以多选择几个初值，择优录用。

### 混合高斯模型  
<a href="https://www.codecogs.com/eqnedit.php?latex=$$p(x;\Theta&space;)&space;=&space;L(\Theta&space;|x)$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$p(x;\Theta&space;)&space;=&space;L(\Theta&space;|x)$$" title="$$p(x;\Theta ) = L(\Theta |x)$$" /></a>  
x为随机变量，p(x;\Theta)表示在当前模型参数下x的概率值。 

<a href="https://www.codecogs.com/eqnedit.php?latex=$$p(x;\Theta&space;)&space;=\sum_{k=1}^{K}\pi&space;_{k}N(x;\mu&space;_{k},\sigma&space;_{k})$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$p(x;\Theta&space;)&space;=\sum_{k=1}^{K}\pi&space;_{k}N(x;\mu&space;_{k},\sigma&space;_{k})$$" title="$$p(x;\Theta ) =\sum_{k=1}^{K}\pi _{k}N(x;\mu _{k},\sigma _{k})$$" /></a>  

<a href="https://www.codecogs.com/eqnedit.php?latex=$$\pi_{k}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\pi_{k}$$" title="$$\pi_{k}$$" /></a>表示每个高斯模型占总模型的比重。 

混合高斯分布的观测值就是以概率<a href="https://www.codecogs.com/eqnedit.php?latex=$$\pi_k$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\pi_k$$" title="$$\pi_k$$" /></a>抽取一个高斯分布，
这个过程以z来表示，高斯分布<a href="https://www.codecogs.com/eqnedit.php?latex=$$N(x;\mu_{k},\sigma_{k})$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$N(x;\mu_{k},\sigma_{k})$$" title="$$N(x;\mu_{k},\sigma_{k})$$" /></a>去生成观测x,此外该高斯分布也可以理解为条件概率分布<a href="https://www.codecogs.com/eqnedit.php?latex=$$p(x|z;\mu,\sigma)$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$p(x|z;\mu,\sigma)$$" title="$$p(x|z;\mu,\sigma)$$" /></a>










