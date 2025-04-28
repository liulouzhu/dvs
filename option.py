import argparse
def paper_args():
    parser = argparse.ArgumentParser(description="DVS")
    parser.add_argument('--model', type=str, default='dvs', help='Model name')
    parser.add_argument('--gpus', type=str, default='0', help='GPU device ID')

    parser.add_argument('--train-list', type=str, default='list/UCF_DVS_Train.list', help='Path to training list')
    parser.add_argument('--test-list', type=str, default='list/UCF_DVS_Test.list', help='Path to testing list')
    parser.add_argument('--feature-size', type=int, default=256, help='Feature size')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--save-path', type=str, default='save', help='Path to save model and results')

    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    
    # parser.add_argument('--seed', type=int, default=42, help='Random seed')
    # parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
    # parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension size')


    
    args = parser.parse_args()
    return args