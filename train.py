import argparse
import shutil
from utils.learning.train_part import train
from pathlib import Path
from utils.model.random_seed import seed_fix


def parse():
    parser = argparse.ArgumentParser(description='Train Unet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('-e', '--num-epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('-l', '--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-r', '--report-interval', type=int, default=500, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='test_Unet', help='Name of network')
    parser.add_argument('-p', '--data-path-train', type=Path, default='../Data/train/', help='Directory of train data')
    parser.add_argument('-t', '--test-size', type=float, default=0.2, help='Ratio of validation data')
    parser.add_argument('-m', '--min-delta', type=float, default=0.0, help='minimum change in the monitored quantity to qualify as an improvement')


    parser.add_argument('--in-chans', type=int, default=1, help='Size of input channels for network')
    parser.add_argument('--out-chans', type=int, default=1, help='Size of output channels for network')
    parser.add_argument('--input-key', type=str, default='image_input', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--patient', type=int, default=3, help='number of checks with no improvement after which training will be stopped')
    

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()
    args.exp_dir = '../result' / args.net_name / 'checkpoints'
    args.val_dir = '../result' / args.net_name / 'reconstructions_val'
    args.main_dir = '../result' / args.net_name / __file__
    
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)

    seed_fix(42)

    train(args)
