import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve,auc,precision_recall_curve

import numpy as np
import os 

def test(model, test_loader):
    model.eval()
    labels = []
    pred = []

    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            labels  += label.cpu().numpy().tolist()
            data = data.cuda()
            label = label.cuda()
            output = model(data)
            pred += output.cpu().numpy().tolist()
            
        fpr, tpr, threshold = roc_curve(labels, pred)
        roc_auc = auc(fpr, tpr)
        print(f"Test AUC: {roc_auc:.4f}")
        # precision, recall, th = precision_recall_curve(labels, pred)
        # pr_auc = auc(recall, precision)
        # print(f"Test PR AUC: {pr_auc:.4f}")

        return roc_auc