from config import configlib
import importlib
from config.config_utils import print_config
import sys
from types import SimpleNamespace

parser = configlib.add_parser("General test config")

parser.add_argument('--test_epoch',default=0,type=int, help='test certain model; -1: find best')
parser.add_argument('--save_res', default=1, type=int, help='save test result')
parser.add_argument('--meta_test', default=0, type=int, help='0: test; 1: meta test')
parser.add_argument('--meta_test_test_ratio', default=0.4, type=float, help='ratio of meta test to meta dev; 0.: use multiple ratio')

# config = SimpleNamespace(**configlib.parse())
# print_config(config, parser)