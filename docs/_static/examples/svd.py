from torch.nn import Module
import torch
import torchbearer as tb

class Net(Module):
    def __init__(self, A, B, S):
        super().__init__()
        self.A = torch.nn.Parameter(A)
        self.B = torch.nn.Parameter(B)
        self.S = torch.nn.Parameter(S)
        self.diag_mask = (torch.ones(n, n) - torch.eye(n, n)).to(dev)

    def f(self):
        Sn = self.S*self.diag_mask
        AS = torch.mm(self.A,Sn)
        ASB = torch.mm(AS, self.B)
        return ASB

    def forward(self, _, state):
        state['A'] = self.A
        state['B'] = self.B
        state['S'] = self.S
        return self.f(), self.S

def loss(target):
    diag_mask = (torch.ones(n,n) - torch.eye(n,n)).to(dev)
    def loss_1(out, _):
        y_pred, S = out
        diff = (y_pred - target)**2
        loss = torch.sum(diff.view(y_pred.shape[0], -1))
        S = S * diag_mask
        loss += torch.sum((S.view(S.shape[0], -1))**2)
        return loss
    return loss_1

steps = None

n = 500
m = 20
dev = 'cuda'

traget = (torch.rand(n,n)+10).to(dev)
A = (torch.rand(n,n)).to(dev)
B = (torch.rand(n,n)).to(dev)
S = (torch.rand(n,n)).to(dev)

model = Net(A, B, S)
optim = torch.optim.Adam(model.parameters(), lr=0.001)

lr_anneal = tb.callbacks.ExponentialLR(0.9999999, step_on_batch=True)

tbmodel = tb.Model(model, optim, loss(traget), ['loss']).to(dev)
tbmodel.fit_generator(None, epochs=1, pass_state=True, callbacks=[], train_steps=100000)
# print(model.f())
# print(model.A)
# print(model.B)
print(torch.max(model.S * (torch.ones(n,n) - torch.eye(n,n)).to(dev)))
print(torch.mean(model.S * (torch.ones(n,n) - torch.eye(n,n)).to(dev)))

