import  torch
def forward(w,b,x):
    return w*x+b
def loss(w,b,x,y):
    y_pre=forward(w,b,x)
    return (y-y_pre)**2

from random import random
X=[]
Y=[]
w=torch.Tensor([1])
w.requires_grad=True

b=torch.Tensor([1])
b.requires_grad=True
for i in range(10):
    X.append(random()*10+1)
    Y.append(X[i] * 2 + 0.5)

for epoch in range(10000):
    for x, y in zip(X, Y):
        l=loss(w,b,x,y)
        l.backward()
        w.data=w.data-0.001*w.grad.data
        b.data=b.data-0.001*b.grad.data
        w.grad.data.zero_()
        b.grad.data.zero_()

print(w.item())
print(b.item())