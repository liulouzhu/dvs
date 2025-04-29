import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.losses import DMIL_CenterLoss
import option
args = option.paper_args()

def train(model,optimizer,train_loader,epoch):
    model.train()

    criterion = DMIL_CenterLoss(k=4, lambda_center=20).cuda()

    for i, (data, label) in enumerate(train_loader):
        data = data.cuda()
        label = label.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output,label)

        loss.backward()
        optimizer.step()

    # if epoch % 10 == 0:
    #     torch.save(model.state_dict(), f'./ckpt/model_epoch_{epoch}.pth')
