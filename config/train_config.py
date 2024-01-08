from config import configlib
from types import SimpleNamespace

parser = configlib.add_parser("General train config")

# Data options
parser.add_argument('--dataset_prefix', default='', type=str)
parser.add_argument('--data_split', default=0, type=int, help='train test sono split')
parser.add_argument('--data_dir', default='', type=str)
parser.add_argument('--aug', default='mult', type=str, help='data augmentation. rot, affine, mult')
parser.add_argument('--ds', default=0.5, type=float, help='frame downsampling rate')
parser.add_argument('--seq_ds', default=1, type=int, help='sequence downsampling rate')
parser.add_argument('--seq_len', default=4, type=int, help='input seq length')
parser.add_argument('--labels', default=['hc', 'csp', 'lv'], nargs='+', type=str, help='segmentation tasks')
parser.add_argument('--meta_train', default=0, type=int, 
                    help='0: train, -1: find meta epoch later than 550, 1: best, >1: meta epoch')
parser.add_argument('--meta_test_dev_ratio', default=0.4, type=float, help='ratio of meta test to meta dev; 0.: use multiple ratio')

# General options
parser.add_argument('--arch', default='blo', type=str, help='choose from archs')
parser.add_argument('--gdl', default=0, type=int, help='if 1 add weight and use generalised dice loss')
parser.add_argument('--gdl_weight', default=[0., 0., 0.], nargs='+', type=float, help='weights for generalised dice loss')
parser.add_argument('--start_epoch', default=0, type=int, help='starting epoch number (for restart)')
parser.add_argument('--epochs', default=1, type=int, help='# total epochs to run')
parser.add_argument('--resume', default='', type=str, help='model path for restart')
parser.add_argument('--batch_size', default=2, type=int, help='train batch size')
parser.add_argument('--workers', default=1, type=int, help='num_workers')
parser.add_argument('--save_freq', default=10, type=int, help='model save frequency')
parser.add_argument('--verbose_freq', default=10, type=int, help='print train status frequency')

# lower level options
parser.add_argument('--lower_loss', default='dsc', type=str, help='loss func, dsc, ce-dsc, ce')
parser.add_argument('--lower_lr', default=1e-4, type=float, help='lower level learning rate')

# upper level options
parser.add_argument('--upper_loss', default='mse', type=str, help='loss func, mse, dsc')
parser.add_argument('--upper_lr', default=1e-4, type=float, help='upper level learning rate')
parser.add_argument('--score', default='avg', type=str, help='how to generate gt for upper level, avg, max, N<1')
parser.add_argument('--score_norm', default='rank', type=str, help='how to normalize score, 0-1, softmax, rank')

# config = SimpleNamespace(**configlib.parse())
# print_config(config, parser)