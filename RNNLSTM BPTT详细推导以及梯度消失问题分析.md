## RNN/LSTM BPTT详细推导以及梯度消失问题分析



## 1. RNN的BPTT

假设RNN的基本方程如下所示

![[公式]](https://www.zhihu.com/equation?tex=h_%7Bt%7D%3D%5Csigma+%28W_x+x_%7Bt%7D%2BW_h+h_%7Bt-1%7D%29%5Ctag%7B1%7D)

![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By_%7Bt%7D%7D%3D%5Csigma+%28W_oh_%7Bt%7D%29%5Ctag%7B2%7D)

损失函数定义如下：

![[公式]](https://www.zhihu.com/equation?tex=L_t+%3D+g%28%5Chat%7By_%7Bt%7D%7D%29+%5Ctag%7B3%7D)

对于一个输入序列 ![[公式]](https://www.zhihu.com/equation?tex=%5Cleft%5C%7B+%28x_t%2Cy_t%29%2Ct%3D1%2C...%2CT+%5Cright%5C%7D) ，其整体损失函数为

![[公式]](https://www.zhihu.com/equation?tex=L%3D%5Csum_%7Bt%3D1%7D%5E%7BT%7D%7BL_t%7D%5Ctag%7B4%7D)

我们接下来分别对 ![[公式]](https://www.zhihu.com/equation?tex=W_h%2CW_x%2CW_o) 进行求导

首先对 ![[公式]](https://www.zhihu.com/equation?tex=W_o) 进行求导，这个比较简单

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+L%7D%7B%5Cpartial+W_o%7D%3D%5Csum_%7Bt%3D1%7D%5E%7BT%7D%7B%5Cfrac%7B%5Cpartial+L_t%7D%7B%5Cpartial+W_o%7D%7D+%3D+%5Csum_%7Bt%3D1%7D%5E%7BT%7D%7B%5Cfrac%7B%5Cpartial+L_t%7D%7B%5Cpartial+y_t%7D%7D+%7B%5Cfrac%7B%5Cpartial+y_t%7D%7B%5Cpartial+W_o%7D%7D+%5Ctag%7B5%7D)

然后对 ![[公式]](https://www.zhihu.com/equation?tex=W_h) 进行求导

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+L%7D%7B%5Cpartial+W_h%7D%3D%5Csum_%7Bt%3D1%7D%5E%7BT%7D%7B%5Cfrac%7B%5Cpartial+L_t%7D%7B%5Cpartial+W_h%7D%7D+%3D+%5Csum_%7Bt%3D1%7D%5E%7BT%7D%7B%5Cfrac%7B%5Cpartial+L_t%7D%7B%5Cpartial+y_t%7D%7D+%7B%5Cfrac%7B%5Cpartial+y_t%7D%7B%5Cpartial+h_t%7D%7D+%7B%5Cfrac%7B%5Cpartial+h_t%7D%7B%5Cpartial+W_h%7D%7D%5Ctag%7B6%7D)

如下公式可知

![[公式]](https://www.zhihu.com/equation?tex=h_%7Bt%7D%3D%5Csigma+%28W_x+x_%7Bt%7D%2BW_h+h_%7Bt-1%7D%29%5Ctag%7B7%7D)

![[公式]](https://www.zhihu.com/equation?tex=h_t) 的计算涉及到 ![[公式]](https://www.zhihu.com/equation?tex=h_%7Bt-1%7D) ,而 ![[公式]](https://www.zhihu.com/equation?tex=h_%7Bt-1%7D) 的计算也涉及到 ![[公式]](https://www.zhihu.com/equation?tex=W_h) ，同样![[公式]](https://www.zhihu.com/equation?tex=h_%7Bt-1%7D) 的计算涉及到 ![[公式]](https://www.zhihu.com/equation?tex=h_%7Bt-2%7D) ,而 ![[公式]](https://www.zhihu.com/equation?tex=h_%7Bt-2%7D) 的计算也涉及到 ![[公式]](https://www.zhihu.com/equation?tex=W_h) ，以此类推，因此需要回溯到t时刻之前的所有时刻，我们需要对公示(6)中的第三项 ![[公式]](https://www.zhihu.com/equation?tex=+%7B%5Cfrac%7B%5Cpartial+h_t%7D%7B%5Cpartial+W_h%7D%7D) 进行展开，下面我们单独对其进行展开如下所示：

![[公式]](https://www.zhihu.com/equation?tex=+%7B%5Cfrac%7B%5Cpartial+h_t%7D%7B%5Cpartial+W_h%7D%7D+%3D+%5Csum_%7Bk%3D1%7D%5E%7Bt%7D+%7B%5Cfrac%7B%5Cpartial+h_t%7D%7B%5Cpartial+h_k%7D%5Cfrac%7B%5Cpartial+h_k%7D%7B%5Cpartial+W_h%7D%7D+%5Ctag%7B8%7D)

同样的道理，公示(8)中的第一项 ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+h_t%7D%7B%5Cpartial+h_k%7D) 的计算如下所示

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+h_t%7D%7B%5Cpartial+h_k%7D+%3D+%5Cfrac%7B%5Cpartial+h_t%7D%7B%5Cpartial+h_%7Bt-1%7D%7D+%5Cfrac%7B%5Cpartial+h_%7Bt-1%7D%7D%7B%5Cpartial+h_%7Bt-2%7D%7D...+%5Cfrac%7B%5Cpartial+h_%7Bk%2B1%7D%7D%7B%5Cpartial+h_k%7D%5Ctag%7B9%7D)

将其带入到公示(8)中即可得到

![[公式]](https://www.zhihu.com/equation?tex=+%7B%5Cfrac%7B%5Cpartial+h_t%7D%7B%5Cpartial+W_h%7D%7D+%3D+%5Csum_%7Bk%3D1%7D%5E%7Bt%7D+%7B%28%5Cprod_%7Bj%3Dk%2B1%7D%5E%7Bt%7D%5Cfrac%7B%5Cpartial+h_j%7D%7B%5Cpartial+h_%7Bj-1%7D%7D%29%5Cfrac%7B%5Cpartial+h_k%7D%7B%5Cpartial+W_h%7D%7D+%5Ctag%7B10%7D)

这样我们把公式(6)中的第三项就展开了，现在带入公式(6)中即可得到：

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+L%7D%7B%5Cpartial+W_h%7D%3D%5Csum_%7Bt%3D1%7D%5E%7BT%7D%7B%5Cfrac%7B%5Cpartial+L_t%7D%7B%5Cpartial+W_h%7D%7D+%3D+%5Csum_%7Bt%3D1%7D%5E%7BT%7D%7B%5Cfrac%7B%5Cpartial+L_t%7D%7B%5Cpartial+y_t%7D%7D+%7B%5Cfrac%7B%5Cpartial+y_t%7D%7B%5Cpartial+h_t%7D%7D+%5Csum_%7Bk%3D1%7D%5E%7Bt%7D+%7B%28%5Cprod_%7Bj%3Dk%2B1%7D%5E%7Bt%7D%5Cfrac%7B%5Cpartial+h_j%7D%7B%5Cpartial+h_%7Bj-1%7D%7D%29%5Cfrac%7B%5Cpartial+h_k%7D%7B%5Cpartial+W_h%7D%7D+%5C%5C+%3D+%5Csum_%7Bt%3D1%7D%5E%7BT%7D+%5Csum_%7Bk%3D1%7D%5E%7Bt%7D%7B+%5Cfrac%7B%5Cpartial+L_t%7D%7B%5Cpartial+y_t%7D%7D+%7B%5Cfrac%7B%5Cpartial+y_t%7D%7B%5Cpartial+h_t%7D%7D+%7B%28%5Cprod_%7Bj%3Dk%2B1%7D%5E%7Bt%7D%5Cfrac%7B%5Cpartial+h_j%7D%7B%5Cpartial+h_%7Bj-1%7D%7D%29%5Cfrac%7B%5Cpartial+h_k%7D%7B%5Cpartial+W_h%7D%7D+%5Ctag%7B11%7D)

按照同样的方式，我们对 ![[公式]](https://www.zhihu.com/equation?tex=W_x) 进行求导

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+L%7D%7B%5Cpartial+W_x%7D%3D+%3D+%5Csum_%7Bt%3D1%7D%5E%7BT%7D+%5Csum_%7Bk%3D1%7D%5E%7Bt%7D%7B+%5Cfrac%7B%5Cpartial+L_t%7D%7B%5Cpartial+y_t%7D%7D+%7B%5Cfrac%7B%5Cpartial+y_t%7D%7B%5Cpartial+h_t%7D%7D+%7B%28%5Cprod_%7Bj%3Dk%2B1%7D%5E%7Bt%7D%5Cfrac%7B%5Cpartial+h_j%7D%7B%5Cpartial+h_%7Bj-1%7D%7D%29%5Cfrac%7B%5Cpartial+h_k%7D%7B%5Cpartial+W_x%7D%7D+%5Ctag%7B12%7D)

------

## 2. RNN梯度消失分析

在上面的推导中，我们对 ![[公式]](https://www.zhihu.com/equation?tex=W_h%2CW_x) 部分的推导公式(11),(12)可以看到，在计算 ![[公式]](https://www.zhihu.com/equation?tex=t) 时刻的损失产生的梯度时，必须回溯之前所有时刻 ![[公式]](https://www.zhihu.com/equation?tex=1%5Csim+t-1) 的信息，并且存在连乘项 ![[公式]](https://www.zhihu.com/equation?tex=%5Cprod_%7Bj%3Dk%2B1%7D%5E%7Bt%7D%5Cfrac%7B%5Cpartial+h_j%7D%7B%5Cpartial+h_%7Bj-1%7D%7D) ，根据公式(1)我们可以计算

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+h_j%7D%7B%5Cpartial+h_%7Bj-1%7D%7D+%3D+%5Csigma%5E%7B%27%7DW_h%5Ctag%7B13%7D)

sigmoid函数的导数大家都很熟悉了，处于 ![[公式]](https://www.zhihu.com/equation?tex=%5B0%2C0.25%5D) 之间，那么会有以下两种情况：

1. 当 ![[公式]](https://www.zhihu.com/equation?tex=W_h) > 4的时候，那么 ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+h_j%7D%7B%5Cpartial+h_%7Bj-1%7D%7D+%3D+%5Csigma%5E%7B%27%7DW_h+%3E+1) ,此时如果 ![[公式]](https://www.zhihu.com/equation?tex=j%2Ck)距离过大，会导致连乘项过多，产生梯度爆炸，趋近于无穷
2. 当 ![[公式]](https://www.zhihu.com/equation?tex=W_h) <4的时候，那么 ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+h_j%7D%7B%5Cpartial+h_%7Bj-1%7D%7D+%3D+%5Csigma%5E%7B%27%7DW_h+%3C+1) ,此时如果 ![[公式]](https://www.zhihu.com/equation?tex=j%2Ck)距离过大，会导致连乘项过多，产生梯度消失，趋近于0

因此当输入序列过长的时候，在求取一个比较远的时刻 ![[公式]](https://www.zhihu.com/equation?tex=t) 的梯度时，需要回溯到前面的所有时刻的信息，由于连乘项的存在，导致前面时刻的信息会缺失，这就是RNN中的梯度消失问题，也是所谓的long-range dependency问题(这样划一个约等号会不会太草率？)；

梯度爆炸问题容易解决，例如采用clip的方式即可。但是梯度消失的问题比较难以解决，我们下面介绍LSTM为什么能够缓解梯度消失问题

------

## 3. LSTM BPTT推导及梯度消失分析

LSTM的公式如下所示

![img](https://pic1.zhimg.com/80/v2-9fbf9b637a9e4865789d13a23245fce4_hd.jpg)





![[公式]](https://www.zhihu.com/equation?tex=i_t+%3D+%5Csigma%28W_i%5Bh_%7Bt-1%7D%3Bx_t%5D%2Bb_i%29%5C%5C+f_t+%3D+%5Csigma%28W_f%5Bh_%7Bt-1%7D%3Bx_t%5D%2Bb_f%29%5C%5C+%5Ctilde%7BC_t%7D+%3D+%5Ctanh%28W_c%5Bh_%7Bt-1%7D%3Bx_t%5D%2Bb_c%29%5C%5C+C_t+%3D+i_t+%5Ccirc+%5Ctilde%7BC_t%7D+%5Coplus+f_t+%5Ccirc+C_%7Bt-1%7D%5C%5C+o_t+%3D+%5Csigma%28W_o%5Bh_%7Bt-1%7D%3Bx_t%5D%2Bb_o%29%5C%5C+h_t+%3D+o_t+%5Ccirc+%5Ctanh%28C_t%29)

其中 ![[公式]](https://www.zhihu.com/equation?tex=C_t) 可以看作之前RNN中的 ![[公式]](https://www.zhihu.com/equation?tex=h_t) ,我们将![[公式]](https://www.zhihu.com/equation?tex=C_t)的计算公式展开如下所示：

![[公式]](https://www.zhihu.com/equation?tex=C_t+%3D++%5Csigma%28W_i%5Bh_%7Bt-1%7D%3Bx_t%5D%2Bb_i%29+%5Ccirc+%5Ctanh%28W_c%5Bh_%7Bt-1%7D%3Bx_t%5D%2Bb_c%29+%5Coplus%5Csigma%28W_f%5Bh_%7Bt-1%7D%3Bx_t%5D%2Bb_f%29+%5Ccirc+C_%7Bt-1%7D%5C%5C+%5Ctag%7B14%7D)

那么需要连乘的部分计算可得：

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+C_j%7D%7B%5Cpartial+C_%7Bj-1%7D%7D+%3D+%5Csigma%28W_f%5Bh_%7Bt-1%7D%3Bx_t%5D%2Bb_f%29%5Ctag%7B15%7D)

从之前的 ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+h_j%7D%7B%5Cpartial+h_%7Bj-1%7D%7D+%3D+%5Csigma%5E%7B%27%7DW_h) 变成了sigmoid函数，范围在[0,1]之间，在实际参数更新中，可以通过控制使得其接近于1，因此多次连乘依然不会产生梯度消失，在 ![[公式]](https://www.zhihu.com/equation?tex=j%2Ck) 距离较大的情况下，依然能够较好的利用 ![[公式]](https://www.zhihu.com/equation?tex=j) 时刻的信息进行梯度计算。

------



## 4. 一些思考

本文由两个问题后续进行提升：

1. LSTM部分的推导并不十分严谨，在RNN BPTT的基础上进行了类比
2. 梯度消失问题以及long-range dependency问题的定义需要明确，本文进行了约等于，是否准确还有待商榷