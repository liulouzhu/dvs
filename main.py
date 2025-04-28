import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

from  model.msf import MultiScaleSpikingFusion
from option import paper_args
from load_data import dataset
from utils.losses import DMIL_CenterLoss
from train import train
from test import test


if __name__ == "__main__":
    args = paper_args()

    random.seed(args.seed)  # Set a random seed for reproducibility

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus # Set the GPU device ID
    # torch.cuda.set_device(args.gpus)  # Set the GPU device ID

    train_loader = DataLoader(dataset(args, test_mode=False), batch_size=args.batch_size,shuffle=True)
    test_loader = DataLoader(dataset(args, test_mode=True), batch_size=1, shuffle=False)

    model = MultiScaleSpikingFusion(in_channels=args.feature_size)
    model = model.cuda()

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    best_auc = 0.0

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    for epoch in tqdm(range(1,args.epochs+1)):
        train(model=model, optimizer=optimizer, train_loader=train_loader, epoch=epoch)
        test_auc = test(model=model, test_loader=test_loader, epoch=epoch)
        
        if test_auc > best_auc:
            best_auc = test_auc
            torch.save(model.state_dict(), f'./ckpt/model_{best_auc}.pth')
            print(f"Model saved at epoch {epoch} with AUC: {best_auc:.4f}")
        else:
            print(f"Epoch {epoch} - AUC: {test_auc:.4f} (Best: {best_auc:.4f})")

        



