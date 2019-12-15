### 概率梯度下降法

在更新参数时，对于N个所有的数据，需要求和。当N很小时没有问题，但是当N变得很大时，on memory没有足够的空间加载data，计算时间变得巨大无比。

**概率梯度下降法**解决：

概率梯度下降法是随机选择一个一个的data，更新参数。
$$w_{k+1} = w_{k}-\eta(y_{n}-t_{n})x_{n} $$

$$b_{k+1} = b_{k}-\eta(y_{n}-t_{n})$$
多行公式：

```math
\displaystyle
\left( \sum\_{k=1}^n a\_k b\_k \right)^2
\leq
\left( \sum\_{k=1}^n a\_k^2 \right)
\left( \sum\_{k=1}^n b\_k^2 \right)
```

```katex
\displaystyle
    \frac{1}{
        \Bigl(\sqrt{\phi \sqrt{5}}-\phi\Bigr) e^{
        \frac25 \pi}} = 1+\frac{e^{-2\pi}} {1+\frac{e^{-4\pi}} {
        1+\frac{e^{-6\pi}}
        {1+\frac{e^{-8\pi}}
         {1+\cdots} }
        }
    }
```

```latex
f(x) = \int_{-\infty}^\infty
    \hat f(\xi)\,e^{2 \pi i \xi x}
    \,d\xi
```



本次Logistic回归使用OR门实现，input的维度为2元，data的数量只有4个。本次使用batch学习，而不是mini-batch梯度下降法。

#### LogisticRegression的模型构建



```python
class LogisticRegression(object):
    
    def __init__(self,input_dim):#构造函数，需要确定维度；初始w,b值随机即可(总之都要学习 )
        self.input_dim = input_dim
        self.w = np.random.normal(size=(input_dim,)) #随机生成一个1行dim列的矩阵
        self.b = 0
        
    def __call__(self,x): #魔法方法call，使实例能作为函数被调用
        return self.forward(x)
    
    def forward(self,x): #向前传递
        return sigmiod(np.matmul(x,self.w)+self.b) #注意x和w的顺序
    
    def compute_gradients(self,x,t): #计算梯度
        y = self.forward(x)
        delta = y-t 
        dw = np.matmul(x.T,delta) 
        db = np.matmul(np.ones(x.shape[0]),delta)
        return dw,db
    
def sigmiod(x):
    return 1/(1+np.exp(-x))
```

注：

比起对data进行一个一个的计算，使用矩阵运算效率更高。

x和self.w的顺序。X是nx1维的矩阵，

`__call__`使实例能作为函数一样被调用。例如：model.forward(x)可以写作model(x)



#### data的准备

这次进行OR门的学习

```python
x = np.array([[0,0],[0,1],[1,0],[1,1]])
t = np.array([0,1,1,1])
```

#### model的构建

```python
model = LogisticRegression(input_dim=2)
```

#### 学习model

1、使交叉熵最小化

2、使用梯度下降法更新参数值

```python
def compute_loss(t,y): #使用cross-entropy计算Loss function
    return (-t*np.log(y) - (1-t)*np.log(1-y)).sum()

def train_step(x,t): #梯度下降法训练，学习率定为0.1，更新参数
    dw,db = model.compute_gradiets(x,t)
    model.w = model.w - 0.1*dw 
    model.b = model.b - 0.1*db
    loss = compute_loss(t,model(x))
    return loss

#进行batch学习
epochs = 100  #对data整体反复学习100次

for epoch in range(epochs):
    train_loss = train_step(x,t) #batch学习
    
    if epoch %10 ==0 or epoch == epochs -1:
        print('epoch: {},loss: {:.3f}'.format(epoch+1,train_loss))
```

结果：

```
epoch: 1,loss: 0.593
epoch: 11,loss: 0.562
epoch: 21,loss: 0.533
epoch: 31,loss: 0.508
epoch: 41,loss: 0.484
epoch: 51,loss: 0.463
epoch: 61,loss: 0.443
epoch: 71,loss: 0.425
epoch: 81,loss: 0.408
epoch: 91,loss: 0.393
epoch: 100,loss: 0.380
```



#### 模型的评价



```

```



