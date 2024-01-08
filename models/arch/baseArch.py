import time, torch, os, sys
import torch.utils.data as tdata
from datetime import datetime
import pickle as pkl

class BaseArch(object):
    def __init__(self, config):
        # general init
        self.config = config
        self.model_path, self.exp_path, self.res_path = self.set_path()
        self.device = self.set_device()
        if config.data_split == 0:
            self.dev_sono = ['S1', 'S2', 'S3', 'S4', ]
            self.test_sono = ['S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S12']
        else:
            raise NotImplementedError(f'*** Invalid data_split {config.data_split} ***')

        # lower level init
        self.lower_model = None
        self.gdl_weight = None
        self.lower_in_c = self.config.seq_len
        self.lower_out_c = self.config.seq_len * len(self.config.labels)
        if self.config.gdl:
            self.gdl_weight = torch.Tensor(self.config.gdl_weight).to(self.device)

        # upper level init
        self.upper_model = None
        self.upper_in_c = self.config.seq_len
        self.upper_out_c = 1

        # globals
        self.epoch = 0
        self.best_mse = 10000
        self.best_model = ''

    def train(self):
        pass

    def validate(self):
        pass

    def evaluation(self):
        pass

    def set_dataloader(self):
        pass

    def set_path(self):
        if self.config.resume:
            exp_path = os.path.dirname(os.path.dirname(self.config.resume))
        else:
            exp_path = f'log-{self.config.save_prefix}/{self.config.exp_name}'
        model_path = os.path.join(exp_path, 'model')
        res_path = os.path.join(exp_path, 'res')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        return model_path, exp_path, res_path

    def set_device(self):
        if torch.cuda.is_available():
            device = torch.device('cuda')
            gpu_id = torch.cuda.current_device()
            gpu_type = torch.cuda.get_device_name(gpu_id)
            print(f'>>> Using GPU {gpu_type}')
        else:
            device = torch.device('cpu')
            print('>>> Using CPU')
        return device

    def save_config(self):
        with open(os.path.join(self.exp_path, 'config.pkl'), 'wb') as f:
            pkl.dump(self.config, f)

    @staticmethod
    def get_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    @staticmethod
    def calc_acc(output, target):
        """calculate the acc in a mini-batch"""
        pred = output.detach().max(1)[1]
        correct = pred.eq(target.view_as(pred)).sum()
        sample_size = pred.shape[0]
        acc = float(correct * 1.0 / sample_size)
        return acc, sample_size, pred

    @staticmethod
    def logit2pix(mask,thre=0.5):
        mask = mask >= thre
        mask = mask * 1
        return mask.astype(int)