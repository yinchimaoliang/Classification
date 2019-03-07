import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt


EPOCH = 1000




class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.hidden = nn.Linear(2,100)
        self.predict = nn.Linear(100,2)
    def forward(self, x):
        out = F.relu(self.hidden(x))
        out = self.predict(out)
        return out




class Main():
    def getData(self):
        n_data = torch.ones(100,2)
        # print(n_data)
        x0 = torch.normal(2 * n_data , 1)#均值和方差
        x1 = torch.normal(-2 * n_data , 1)
        self.x = torch.cat((x0,x1),0).cuda() #按行拼接
        # print(self.x)
        # print(x0)
        y0 = torch.zeros(100)
        y1 = torch.ones(100)
        self.y = torch.cat((y0,y1),0).type(torch.LongTensor)#标签必须为long
        self.y = self.y.cuda()
        # plt.scatter(self.x.data.cpu().numpy()[:,0],self.x.data.cpu().numpy()[:,1])
        # plt.show()

    def main(self):
        net = Net().cuda()
        optimizer = torch.optim.SGD(net.parameters(),lr = 0.01)
        loss_func = nn.CrossEntropyLoss()
        for epoch in range(EPOCH):
            out = net(self.x)
            loss = loss_func(out,self.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            plt.ion()
            if epoch % 100 == 0:
                plt.cla()
                prediction = torch.max(F.softmax(out),1)[1]
                # print(out)
                # print(prediction)

                plt.scatter(self.x[:,0].data.cpu().numpy(),self.x[:,1].cpu().numpy(), c = prediction.data.cpu().numpy(),cmap = "RdYlGn")
                plt.pause(0.3)
        plt.ioff()
        plt.show()





if __name__ == "__main__":
    t = Main()
    t.getData()
    t.main()