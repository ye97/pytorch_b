import torch
#model实现自己的类，继承module模块
class Model(torch.nn.Module):
    #继承他的原因是初始化的调用函数要使用他
    def __init__(self):
        #调用父辈的同名方法都是这样的调用 super（本类名，self）.方法
        # python中的super( test, self).init()
        # 首先找到test的父类（比如是类A），然后把类test的对象self转换为类A的对象，然后“被转换”的类A对象调用自己的__init__函数.
        super(Model,self).__init__()
        #实例中添加linear函数，使用torch中linear函数，返回得到一个linear对象
        self.linear=torch.nn.Linear(1,1)
    def forward(self,x):
        #实现正向传播图 调用实例对象的linear对象，即上面初始化的对象
        return self.linear(x)


model=Model()

#此处loss和optim都是可调用对象，即无参构造，传入参数调用
loss=torch.nn.MSELoss(reduction="sum")
optim=torch.optim.SGD(model.parameters(),lr=0.01)


x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])

for epoch in range(1,1000):
    y=model(x_data)
    cost=loss(y,y_data)
    print(epoch,":",cost.data.item(),cost.data.item())
    optim.zero_grad()
    cost.backward()
    optim.step()
print(model.linear.weight.item())
print(model.linear.bias.item())