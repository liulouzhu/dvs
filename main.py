import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from  model.msf import MultiScaleSpikingFusion
from option import paper_args
from load_data import dataset
from utils.losses import DMIL_CenterLoss
from train import train
from test import test


if __name__ == "__main__":
    args = paper_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus # Set the GPU device ID
    # torch.cuda.set_device(args.gpus)  # Set the GPU device ID

    train_loader = DataLoader(dataset(args, test_mode=False, batch_size=args.batch_size//2))
    test_loader = DataLoader(dataset(args, test_mode=True, batch_size=args.batch_size))

    model = MultiScaleSpikingFusion(in_channels=args.feature_size)
    model = model.cuda()

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    best_auc = 0.0

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    for epoch in tqdm(range(1,args.epochs+1)):
        train(model=model, optimizer=optimizer, train_loader=train_loader, epoch=epoch)
        test_auc = test(model=model, test_loader=test_loader, epoch=epoch)
        

        



