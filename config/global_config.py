from config import configlib
import importlib
from config.config_utils import print_config
import sys
from types import SimpleNamespace


spec_list = [
    'test',
    'train'
]

def get_spec():
    command_line = ' '.join(sys.argv)
    for pj in spec_list:
        segment = f"--spec {pj}"
        if segment in command_line:
            return pj
    raise NotImplementedError

# importlib.import_module(f"config.train_config")
# importlib.import_module(f"config.test_config")
importlib.import_module(f"config.{get_spec()}_config")

parser = configlib.add_parser("General config")
parser.add_argument('--spec', default=None, type=str, help=f'the specific config name {spec_list}')
parser.add_argument('--save_prefix', default='', type=str, help='model save path prefix')
parser.add_argument('--exp_name', default='', type=str, help='a special marker for experiment name.')
parser.add_argument('--test_stride', default=0, type=int, help='stride for slicing segment during test')

config = SimpleNamespace(**configlib.parse())
print_config(config, parser)

# set default values for new config
NEW_PARAMETERS = {
    'seq_ds':1, 
    'meta_train':0, 
    'meta_test_dev_ratio':0.4,}
def load_default_config(configs):
    for k, v in NEW_PARAMETERS.items():
        if not hasattr(configs, k):
            setattr(configs, k, v)
    return configs