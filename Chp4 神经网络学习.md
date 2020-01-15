# Chp4 神经网络学习



### 4.4 梯度法

注意：梯度表示的是**各点处**的函数值减少最多的方向。因此，无法保证梯度所指的方向就是**函数**的最小值或者真正前进的方向。

**梯度法（gradient method）**：在梯度法中，函数的取值从当前位置沿着梯度方向前进一定距离，然后在新的地方重新求梯度，再沿着新梯度方向前进，如此反复，不断地沿梯度方向前进。像这样，通过不断地沿梯度方向前进，逐渐减小函数值的过程就是梯度法（gradient method）。

   

#### 梯度计算代码

```python
def numerical_gradient(f,x): #f是函数,x是(x1,x2)组成
    h = 0.0001 #一个极小值
    grad = np.zero_like(x) #生成和x形状相同的数组
    
    for idx in range(x.size): #通过循环分别对x1,x2求偏导
        tmp_val = x[idx] 
        #计算f(x+h)
        x[idx] = tmp_val+h #此时x变成(x1+h,x2)
        fxh1 = f(x)
        
        #计算f(x-h)
        x[idx] = tmp_val-h #此时x变成(x1-h,x2)
        fxh2 = f(x) 
        
        grad[idx] = (fxh1-fxh2)/(2*h)
        x[idx] = tmp_val #还原值
        
     return grad
```



#### 梯度下降法代码

```python
def gradient_descent(f,init_x,lr=0.01,step_num=100):
    x = init_x
    
    for i in range(step_num):
        grad = numerical_gradient(f,x) #计算梯度
        x- = lr*grad #更新参数
        
    return x
```

