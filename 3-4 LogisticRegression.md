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

x和self.w的顺序。

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

```python
def compute_loss(t,y): #使用cross-entropy计算Loss function
    return (-t*np.log(y) - (1-t)*np.log(1-y)).sum()

def 
```

