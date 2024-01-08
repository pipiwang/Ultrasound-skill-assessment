from models.arch.baseArch import BaseArch
from models.networks.unet import UNet
from dataset import FrameSeqDataset
import torch.utils.data as tdata
from models.networks.resnet_backbone_modified import resnet18
from loss import WeightedDiceLoss, mult_binary_dice, binary_dice, calc_assd
import torch, time, shutil, glob,  os, ujson
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pickle as pkl
from pathlib import Path
import random

class BloTrainArch(BaseArch):
    def __init__(self, config):
        super(BloTrainArch, self).__init__(config)

        # init lower level
        self.lower_model = UNet(in_channels=self.lower_in_c, out_channels=self.lower_out_c,
                              init_features=32).to(self.device)
        if self.config.lower_loss == 'dsc':
            if self.config.gdl:
                self.lower_criterion = WeightedDiceLoss(weight=self.gdl_weight.repeat(self.config.seq_len))
                print(f'Using generalised dice, weight {self.gdl_weight}')
            else:
                self.lower_criterion = WeightedDiceLoss()
        else:
            raise NotImplementedError('***Invalid lower level loss type.***')
        self.lower_optimizer = optim.Adam(self.lower_model.parameters(), lr=self.config.lower_lr)
        print('>>> Lower level inited.')

        # init upper level
        self.upper_model = resnet18(in_channel=self.upper_in_c, num_classes=self.upper_out_c).to(self.device)
        if self.config.upper_loss == 'mse':
            self.upper_criterion = nn.MSELoss()
        else:
            raise NotImplementedError('***Invalid upper level loss type.***')
        self.upper_optimizer = optim.Adam(self.upper_model.parameters(), lr=self.config.upper_lr)
        print('>>> Upper level inited.')

        if self.config.meta_train:
            # load meta train init model dict
            # find test model pth
            if self.config.meta_train == 1:
                self.best_model = os.path.join(self.exp_path, glob.glob1(self.exp_path, '*best.pth')[0])
            elif self.config.meta_train == -1:
                # find model after 550:
                min_mse  = float('inf')
                selected_model = None

                # List files in the folder
                files = os.listdir(self.model_path)

                # Iterate over the files
                for file_name in files:
                    if file_name.endswith('.pth') and 'meta' not in file_name:
                        # Extract the epoch and loss values from the file name
                        parts = file_name.split('_')
                        epoch = int(parts[1])
                        loss = float(parts[3][:-4])  # Removing the ".pth" extension

                        # Check if the epoch is greater than 550 and the loss is smaller than the current minimum
                        if epoch > 550 and min_mse > loss > 0:
                            min_mse  = loss
                            self.best_model = os.path.join(self.model_path, file_name)
            else:
                model_dir = os.listdir(self.model_path)
                for pth in model_dir:
                    epoch = int(pth.split('_')[1])
                    if epoch == self.config.meta_train:
                        self.best_model = os.path.join(self.model_path, pth)
                        break
            # load meta train model
            checkpoint = torch.load(self.best_model)
            self.lower_model.load_state_dict(checkpoint['lower'])
            self.upper_model.load_state_dict(checkpoint['upper'])
            print(f'>>> Meta train start model {self.best_model}')

    def train(self):
        self.save_config()
        self.set_trainloader()
        print('>>> Bilevel train start <<<')

        for self.epoch in range(self.config.start_epoch, self.config.epochs+self.config.start_epoch):
            start_time = time.time()

            val_dice_array = []
            for step, data in enumerate(self.train_loader):
                # with torch.autograd.set_detect_anomaly(True):
                # upper level step
                upper_data = next(iter(self.val_loader))
                score, upper_loss, val_dice = self.train_upper(upper_data)
                val_dice_array.append(val_dice)

                # lower level
                self.lower_model.train()
                frames = data[0].float().to(self.device)
                masks = data[1].to(self.device)

                self.lower_optimizer.zero_grad()
                out =  self.lower_model(frames)
                score = self.normalise_score(score) # normalise upper level score within a mini-batch
                lower_loss = self.lower_criterion(out, masks, score)
                lower_loss.backward()
                self.lower_optimizer.step()

                if step % self.config.verbose_freq == 0 or step == len(self.train_loader)-1:
                    print(
                        f"Epoch {self.epoch:03} Step {step:02}"
                        f" | lower(seg) loss: {lower_loss.item(): .4f} | upper(SA) loss: {upper_loss.item(): .4f}"
                    )

            val_dice_array = torch.stack(val_dice_array) #(num_step, num_labels)
            val_dice = val_dice_array.mean(dim=(0))

            # validation
            val_mse, train_dice = self.validate()
            # save best model
            if val_mse < self.best_mse:
                self.best_mse = val_mse
                if self.config.meta_train:
                    self.best_model = self.save_checkpoint(f'meta{self.config.meta_test_dev_ratio}',val_mse)
                else:
                    self.best_model = self.save_checkpoint('blo',val_mse)

            end_time = time.time()
            epoch_mins, epoch_secs = self.get_time(start_time, end_time)

            print(
                f"Epoch [{self.epoch:03}/{self.config.epochs+self.config.start_epoch - 1:03}] | time: {epoch_mins}m {epoch_secs}s"
                f" | Val MSE: {val_mse: .4f} | Pseudo Val Dice: {val_dice.cpu().numpy()}"
            )
            if (self.epoch % self.config.save_freq == 0) or (self.epoch == self.config.epochs+self.config.start_epoch - 1):
                if self.config.meta_train:
                    _ = self.save_checkpoint(f'meta{self.config.meta_test_dev_ratio}',val_mse)
                else:
                    _ = self.save_checkpoint('blo',val_mse)

        save_name = self.best_model.split('/')[-1]
        shutil.copy(self.best_model,
                    os.path.join(self.exp_path, save_name.replace('.pth', f'-{self.config.exp_name}-best.pth')))

    def train_upper(self, data):
        self.upper_model.train()

        frames = data[0].float().to(self.device)
        masks = data[1].to(self.device)

        # lower model inference
        self.lower_model.eval()
        with torch.no_grad():
            lower_out = self.lower_model(frames)
            dice_array = mult_binary_dice(lower_out, masks) #(b,seq_len * num_labels)
        upper_target = self.select_dice(dice_array)

        dice_array = dice_array.view(-1, self.config.seq_len, len(self.config.labels))

        self.upper_optimizer.zero_grad()
        score = self.upper_model(frames)
        score = score.squeeze()

        upper_loss = self.upper_criterion(score, upper_target)

        upper_loss.backward(retain_graph=True)
        self.upper_optimizer.step()

        return score, upper_loss, dice_array.mean(dim=(0,1))

    def validate(self):
        """validate upper model on support set (train set)"""

        self.upper_model.eval()
        self.lower_model.eval()

        total_mse = 0
        sample_num = 0

        total_dice = []

        with torch.no_grad():
            for i, data in enumerate(self.train_loader):
                frames = data[0].float().to(self.device)
                masks = data[1].to(self.device)

                # lower model inference
                lower_out = self.lower_model(frames)
                dice_array = mult_binary_dice(lower_out, masks) #(b,seq_len * num_labels)
                upper_target = self.select_dice(dice_array)

                # upper model inference
                out = self.upper_model(frames)
                out = out.squeeze(1)
                mse = nn.functional.mse_loss(out, upper_target, reduction='sum')

                total_mse += mse
                sample_num += out.shape[0]

                total_dice.append(dice_array)
        total_dice = torch.cat(total_dice, dim=0)
        total_dice = total_dice.view(sample_num, self.config.seq_len, len(self.config.labels))

        return 1. * total_mse / sample_num, total_dice.mean(dim=(0,1))

    def evaluation(self):
        # find test model pth
        name = 'blo'
        if self.config.test_epoch == 0:
            self.best_model = os.path.join(self.exp_path, glob.glob1(self.exp_path, f'*{name}*best.pth')[0])
        elif self.config.test_epoch == -1:
            # find model after 550:
            min_mse  = float('inf')
            selected_model = None

            # List files in the folder
            files = os.listdir(self.model_path)

            # Iterate over the files
            for file_name in files:
                if name in file_name and file_name.endswith('.pth'):
                    # Extract the epoch and loss values from the file name
                    parts = file_name.split('_')
                    epoch = int(parts[1])
                    loss = float(parts[3][:-4])  # Removing the ".pth" extension

                    # Check if the epoch is greater than 550 and the loss is smaller than the current minimum
                    if epoch > 550 and min_mse > loss > 0:
                        min_mse  = loss
                        self.best_model = os.path.join(self.model_path, file_name)
                        test_epoch = epoch
            self.config.test_epoch = test_epoch
        else:
            model_dir = os.listdir(self.model_path)
            for pth in model_dir:
                epoch = int(pth.split('_')[1])
                if epoch == self.config.test_epoch and name in pth:
                    self.best_model = os.path.join(self.model_path, pth)
                    break

        # load test model
        if int(self.config.exp_name[:2]) < 3:
            lower_pth = os.path.join(self.model_path,glob.glob1(self.model_path, f'*_{self.config.test_epoch}_lower_*.pth')[0])
            self.lower_model.load_state_dict(torch.load(lower_pth))
            upper_pth = os.path.join(self.model_path,glob.glob1(self.model_path, f'*_{self.config.test_epoch}_upper_*.pth')[0])
            self.upper_model.load_state_dict(torch.load(upper_pth))
            print(f'>>> Using model {upper_pth}')
        else:
            checkpoint = torch.load(self.best_model)
            self.lower_model.load_state_dict(checkpoint['lower'])
            self.upper_model.load_state_dict(checkpoint['upper'])
            print(f'>>> Using model {self.best_model}')
        print('>>> Test model loaded. ')
        self.upper_model.eval()
        self.lower_model.eval()

        total_mse, sample_num = 0, 0
        total_dice = []
        save_res_array = []
        tp_dict = {key: [] for key in self.config.labels}
        assd_dict = {key: [] for key in self.config.labels} 
        top1_cnt, top5_cnt, total_test_cnt = 0, 0, 0
        sp_top1_cnt, sp_top5_cnt = 0, 0
        sp_top1_start_diff = {}

        # load test data
        data_dir = Path(self.config.data_dir).expanduser().resolve()
        with torch.no_grad():
            for sono in self.test_sono:
                with open(data_dir / 'Dataset' / f'{self.config.dataset_prefix}_{sono}.json') as f:
                    test_file = ujson.load(f)
                # test
                for test_idx in range(len(test_file)):
                    print(f'>>> Testing sonographer {sono} file {test_idx+1} ...')
                    self.set_testloader(sono, test_idx)
                    res_dict_array = []

                    for idx, data in enumerate(self.test_loader):
                        frames = data[0].float().to(self.device)
                        # print(frames.shape)
                        masks = data[1].to(self.device)
                        scan_start_id = data[2]
                        sp_flag = data[3]

                        # lower model inference
                        lower_out = self.lower_model(frames)
                        dice_array = mult_binary_dice(lower_out, masks)  # (b,seq_len * num_labels)
                        upper_target = self.select_dice(dice_array)

                        # upper model inference
                        out = self.upper_model(frames)
                        out = out.squeeze(1)
                        mse = nn.functional.mse_loss(out, upper_target, reduction='sum')

                        total_mse += mse
                        sample_num += out.shape[0]

                        total_dice.append(dice_array)

                        for b in range(frames.shape[0]):
                            scan_id = scan_start_id[b].split('_')[0]
                            start_frame_nr = int(scan_start_id[b].split('_')[1])
                            masks_pred = lower_out[b].detach().cpu().numpy()
                            masks_true = masks[b].detach().cpu().numpy()
                            if self.config.save_res == 2: # save dice 
                                res_dict = {
                                    'scan_id': scan_id,
                                    'start_frame_nr': start_frame_nr,
                                    'score_pred': out[b].detach().cpu().item(),
                                    'score_gt': upper_target[b].detach().cpu().item(),
                                    'avg_dice': dice_array[b].mean().detach().cpu().item(),
                                    'dice_array': dice_array[b].detach().cpu().numpy(),
                                    'sp_flag': sp_flag[b]
                                }
                            else: # save all
                                res_dict = {
                                    'scan_id': scan_id,
                                    'start_frame_nr': start_frame_nr,
                                    'masks_pred': masks_pred,
                                    'score_pred': out[b].detach().cpu().item(),
                                    'score_gt': upper_target[b].detach().cpu().item(),
                                    'avg_dice': dice_array[b].mean().detach().cpu().item(),
                                    'sp_flag': sp_flag[b]
                                }
                                
                            res_dict_array.append(res_dict)

                            for c in range(frames.shape[1]):
                                # calculate dice when gt exists:
                                for n, label in enumerate(tp_dict.keys()):
                                    index = c*len(tp_dict.keys())+n
                                    # check if gt exists
                                    if masks_true[index].sum() > 0:
                                        dsc = binary_dice(masks_pred[index], masks_true[index])
                                        tp_dict[label].append(dsc)
                                        if masks_pred[index].sum() > 0:
                                            assd = calc_assd(masks_pred[index], masks_true[index])
                                            assd_dict[label].append(assd)

                            if sp_flag[b] == 1:
                                sp_res_dict = res_dict

                    save_res_array.extend(res_dict_array)
                    sorted_res = sorted(res_dict_array, key=lambda x: x['score_pred'], reverse=True)
                    top_1 = sorted_res[0]
                    top_5 = sorted_res[:5]
                    avg_top5_dice = sum(res['avg_dice'] for res in top_5) / len(top_5)
                    if top_1['avg_dice'] > dice_array.mean().item():
                        top1_cnt += 1
                    if avg_top5_dice > dice_array.mean().item():
                        top5_cnt += 1
                    total_test_cnt += 1
                    sp_rank = 0
                    for rank, res_dict in enumerate(sorted_res):
                        if res_dict['sp_flag'] == 1:
                            sp_rank = rank
                            sp_dict = res_dict
                            break
                    if sp_rank == 0:
                        sp_top1_cnt += 1
                    if sp_rank < 5:
                        sp_top5_cnt += 1

                    print(f"top dice: {top_1['avg_dice']}, top 5 avg dice: {avg_top5_dice}, "
                          f"sp dice: {sp_dict['avg_dice']}, average dice: {dice_array.mean().item()}")
                    print(f"top dice start frame {top_1['start_frame_nr']}, sp start frame {sp_dict['start_frame_nr']}")
                    sp_top1_start_diff[scan_id] = abs(top_1['start_frame_nr'] - sp_dict['start_frame_nr'])
            
            save_res_array.append(tp_dict)
            save_res_array.append(assd_dict)
            if self.config.save_res:
                
                with open(os.path.join(self.res_path, 
                                       f'{self.config.exp_name}_epoch{self.config.test_epoch}_res{self.config.save_res}.pkl'), 'wb') as f:
                    pkl.dump(save_res_array, f)
                print(f'>>> res {self.config.save_res} saved.')    

            total_dice = torch.cat(total_dice, dim=0)
            total_dice = total_dice.view(sample_num, self.config.seq_len, len(self.config.labels))
            mean_dice = total_dice.mean(dim=(0,1))

            print('Dice score when gt exists:')
            for label, dsc_list in tp_dict.items():
                print(label, np.mean(np.array(dsc_list)), len(dsc_list))
            print('False Positives, All negatives, and False positive rates: ')
            print(f'{top1_cnt*1./total_test_cnt} top 1 good, {top5_cnt * 1. / total_test_cnt} top 5 good')
            print(f'{sp_top1_cnt/total_test_cnt} top 1 matches sp segment, {sp_top5_cnt/total_test_cnt} top 5 contains sp segment')
            print(f'>>> Test results for model epoch {self.config.test_epoch}: \n'
                  f'Total MSE: {1. * total_mse / sample_num} \n')
            for idx, entry in enumerate(self.config.labels):
                print(entry, mean_dice[idx].cpu().numpy())
            print(f'average sp and top1 start frame difference: {np.mean(list(sp_top1_start_diff.values()))}')
            print(sp_top1_start_diff)

    def meta_eval_test(self):
        # find test model pth
        name = f'meta{self.config.meta_test_dev_ratio}'
        if self.config.test_epoch == -1:
            # find meta eval model:
            min_mse  = float('inf')
            selected_model = None

            # List files in the folder
            files = os.listdir(self.model_path)

            # Iterate over the files
            for file_name in files:
                if name in file_name and file_name.endswith('.pth'):
                    # Extract the epoch and loss values from the file name
                    parts = file_name.split('_')
                    epoch = int(parts[1])
                    loss = float(parts[3][:-4])  # Removing the ".pth" extension

                    # Check if the epoch is greater than 550 and the loss is smaller than the current minimum
                    if epoch > 550 and min_mse > loss > 0:
                        min_mse  = loss
                        self.best_model = os.path.join(self.model_path, file_name)
                        test_epoch = epoch
            self.config.test_epoch = test_epoch
        elif self.config.test_epoch == -2:
            # find the un-finetuned best model:
            name = 'blo'
            min_mse  = float('inf')
            selected_model = None

            # List files in the folder
            files = os.listdir(self.model_path)

            # Iterate over the files
            for file_name in files:
                if name in file_name and file_name.endswith('.pth'):
                    # Extract the epoch and loss values from the file name
                    parts = file_name.split('_')
                    epoch = int(parts[1])
                    loss = float(parts[3][:-4])  # Removing the ".pth" extension

                    # Check if the epoch is greater than 550 and the loss is smaller than the current minimum
                    if 550< epoch < 4001 and min_mse > loss > 0:
                        min_mse  = loss
                        self.best_model = os.path.join(self.model_path, file_name)
                        test_epoch = epoch
            self.config.test_epoch = test_epoch
        else:
            model_dir = os.listdir(self.model_path)
            for pth in model_dir:
                epoch = int(pth.split('_')[1])
                if epoch == self.config.test_epoch and name in pth:
                    self.best_model = os.path.join(self.model_path, pth)
                    break

        # load test model
        checkpoint = torch.load(self.best_model)
        self.lower_model.load_state_dict(checkpoint['lower'])
        self.upper_model.load_state_dict(checkpoint['upper'])
        print(f'>>> Using model {self.best_model}')
        print('>>> Test model loaded. ')
        self.upper_model.eval()
        self.lower_model.eval()

        total_mse, sample_num = 0, 0
        total_dice = []
        save_res_array = []
        tp_dict = {key: [] for key in self.config.labels}
        assd_dict = {key: [] for key in self.config.labels} 
        top1_cnt, top5_cnt, total_test_cnt = 0, 0, 0
        sp_top1_cnt, sp_top5_cnt = 0, 0
        sp_top1_start_diff = {}

        # load test data
        data_dir = Path(self.config.data_dir).expanduser().resolve()
        with torch.no_grad():
            data_list  = []
            for sono in self.test_sono:
                with open(data_dir / 'Dataset' / f'{self.config.dataset_prefix}_{sono}.json') as f:
                    data_list.extend(ujson.load(f))
            # random split meta test developement set
            random.seed(42)
            random.shuffle(data_list)
            cutoff = int(len(data_list) * self.config.meta_test_dev_ratio)
            if cutoff % 2 == 1:
                cutoff += 1
            test_file = data_list[cutoff:]
            print(f'meta eval test set size: {len(test_file)}')
            # test
            for test_idx in range(len(test_file)):
                print(f'>>> Testing file {test_idx+1} ...')
                self.set_testloader(sono, test_idx)
                res_dict_array = []

                for idx, data in enumerate(self.test_loader):
                    frames = data[0].float().to(self.device)
                    masks = data[1].to(self.device)
                    scan_start_id = data[2]
                    sp_flag = data[3]

                    # lower model inference
                    lower_out = self.lower_model(frames)
                    dice_array = mult_binary_dice(lower_out, masks)  # (b,seq_len * num_labels)
                    upper_target = self.select_dice(dice_array)

                    # upper model inference
                    out = self.upper_model(frames)
                    # print(frames.shape,out.shape)
                    out = out.squeeze(1)
                    mse = nn.functional.mse_loss(out, upper_target, reduction='sum')

                    total_mse += mse
                    sample_num += out.shape[0]

                    total_dice.append(dice_array)

                    # res dict
                    for b in range(frames.shape[0]):
                        scan_id = scan_start_id[b].split('_')[0]
                        start_frame_nr = int(scan_start_id[b].split('_')[1])
                        masks_pred = lower_out[b].detach().cpu().numpy()
                        masks_true = masks[b].detach().cpu().numpy()
                        if self.config.save_res == 2: # save dice 
                            res_dict = {
                                'scan_id': scan_id,
                                'patient_id': sono,
                                'start_frame_nr': start_frame_nr,
                                # 'masks_pred': masks_pred,
                                # 'masks_true': masks_true,
                                'score_pred': out[b].detach().cpu().item(),
                                'score_gt': upper_target[b].detach().cpu().item(),
                                'avg_dice': dice_array[b].mean().detach().cpu().item(),
                                'dice_array': dice_array[b].detach().cpu().numpy(),
                                'sp_flag': sp_flag[b]
                            }
                        else: # save all
                            res_dict = {
                                'scan_id': scan_id,
                                'start_frame_nr': start_frame_nr,
                                'masks_pred': masks_pred,
                                # 'masks_true': masks_true,
                                'score_pred': out[b].detach().cpu().item(),
                                'score_gt': upper_target[b].detach().cpu().item(),
                                'avg_dice': dice_array[b].mean().detach().cpu().item(),
                                'sp_flag': sp_flag[b]
                            }
                        res_dict_array.append(res_dict)

                        for c in range(frames.shape[1]):
                            # calculate dice when gt exists:
                            for n, label in enumerate(tp_dict.keys()):
                                index = c*len(tp_dict.keys())+n
                                # check if gt exists
                                if masks_true[index].sum() > 0:
                                    dsc = binary_dice(masks_pred[index], masks_true[index])
                                    tp_dict[label].append(dsc)
                                    if masks_pred[index].sum() > 0:
                                            assd = calc_assd(masks_pred[index], masks_true[index])
                                            assd_dict[label].append(assd)

                        if sp_flag[b] == 1:
                            sp_res_dict = res_dict

                save_res_array.extend(res_dict_array)
                sorted_res = sorted(res_dict_array, key=lambda x: x['score_pred'], reverse=True)
                top_1 = sorted_res[0]
                top_5 = sorted_res[:5]
                avg_top5_dice = sum(res['avg_dice'] for res in top_5) / len(top_5)
                if top_1['avg_dice'] > dice_array.mean().item():
                    top1_cnt += 1
                if avg_top5_dice > dice_array.mean().item():
                    top5_cnt += 1
                total_test_cnt += 1
                sp_rank = 0
                sp_dict = None
                for rank, res_dict in enumerate(sorted_res):
                    if res_dict['sp_flag'] == 1:
                        sp_rank = rank
                        sp_dict = res_dict
                        break
                if sp_rank == 0:
                    sp_top1_cnt += 1
                if sp_rank < 5:
                    sp_top5_cnt += 1

                print(f"top dice: {top_1['avg_dice']}, top 5 avg dice: {avg_top5_dice}")
                print(f"sp dice: {sp_dict['avg_dice']}, average dice: {dice_array.mean().item()}")
                print(f"top dice start frame {top_1['start_frame_nr']}, sp start frame {sp_dict['start_frame_nr']}")
                sp_top1_start_diff[scan_id] = abs(top_1['start_frame_nr'] - sp_dict['start_frame_nr'])
            
            save_res_array.append(tp_dict)
            save_res_array.append(assd_dict)
            if self.config.save_res:
                # save tp dict
                with open(os.path.join(self.res_path, 
                                       f'{self.config.exp_name}_epoch{self.config.test_epoch}_meta{self.config.meta_test_dev_ratio}_res{self.config.save_res}.pkl'), 'wb') as f:
                    pkl.dump(save_res_array, f)
                print(f'>>> res {self.config.save_res} saved.')   

            total_dice = torch.cat(total_dice, dim=0)
            total_dice = total_dice.view(sample_num, self.config.seq_len, len(self.config.labels))
            mean_dice = total_dice.mean(dim=(0,1))

            print('Dice score when gt exists:')
            for label, dsc_list in tp_dict.items():
                print(label, np.mean(np.array(dsc_list)), len(dsc_list))
            print('False Positives, All negatives, and False positive rates: ')
            print(f'{top1_cnt*1./total_test_cnt} top 1 good, {top5_cnt * 1. / total_test_cnt} top 5 good')
            print(f'{sp_top1_cnt/total_test_cnt} top 1 matches sp segment, {sp_top5_cnt/total_test_cnt} top 5 contains sp segment')
            print(f'>>> Test results for model epoch {self.config.test_epoch}: \n'
                  f'Total MSE: {1. * total_mse / sample_num} \n')
            for idx, entry in enumerate(self.config.labels):
                print(entry, mean_dice[idx].cpu().numpy())
            print(f'average sp and top1 start frame difference: {np.mean(list(sp_top1_start_diff.values()))}')
            print(sp_top1_start_diff)

    def select_dice(self, lower_dsc):
        lower_dsc = lower_dsc.view(-1, self.config.seq_len, len(self.config.labels))
        lower_dsc = lower_dsc.mean(dim=2) # (b,seq_len)

        if self.config.score == 'avg':
            dice = lower_dsc.mean(dim=1)
        elif self.config.score == 'max':
            dice =  lower_dsc.max(dim=1)[0]
        elif self.config.score.startswith('0.'):
            N = float(self.config.score)
            lower_dsc = torch.topk(lower_dsc, int(N*self.config.seq_len), dim=1)[0]
            dice = lower_dsc.mean(dim=1)
        else:
            raise NotImplementedError('*** Invalid sys input for gt-score generation ***')
        return dice

    def normalise_score(self, dice):
        if self.config.score_norm == '0-1':
            min_val = dice.min()
            max_val = dice.max()
            dice = (dice - min_val) / (max_val - min_val)
        elif self.config.score_norm == 'softmax':
            dice = torch.softmax(dice, dim=0)
        elif self.config.score_norm == 'rank':
            sorted_indices = torch.argsort(dice)
            rank_scores = torch.linspace(0, 1, len(dice))
            assigned_scores = torch.zeros_like(dice)
            for i, index in enumerate(sorted_indices):
                assigned_scores[index] = rank_scores[i]
            dice = assigned_scores
        else:
            raise NotImplementedError('*** Invalid sys input for score normalization method ***')

        return dice

    def set_trainloader(self):
        if self.config.meta_train:
            print(self.test_sono)
            train_set = FrameSeqDataset(config=self.config, phase='meta_train', sono_list=self.test_sono,
                                        meta_test_dev_ratio=self.config.meta_test_dev_ratio)
            self.train_loader = tdata.DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True,
                                                 num_workers=self.config.workers, drop_last=False)
            print(f'>>> Meta evaluation train set ready. ')
            val_set = FrameSeqDataset(config=self.config, phase='meta_val', sono_list=self.test_sono,
                                        meta_test_dev_ratio=self.config.meta_test_dev_ratio)
            self.val_loader = tdata.DataLoader(val_set, batch_size=self.config.batch_size,
                                               sampler=tdata.RandomSampler(val_set),
                                               num_workers=self.config.workers, drop_last=False)
            print(f'>>> Meta evaluation validation set ready. ')
            assert len(self.train_loader) == len(self.val_loader), "Meta eval train and val loader should have same length."
            assert len(self.train_loader)<2, "Meta eval train loader should only have one batch."
        else:
            train_set = FrameSeqDataset(config=self.config, phase='train', sono_list=self.dev_sono)
            self.train_loader = tdata.DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True,
                                                num_workers=self.config.workers, drop_last=True)
            print('>>> Train set ready.')
            val_set = FrameSeqDataset(config=self.config, phase='val', sono_list=self.dev_sono)
            self.val_loader = tdata.DataLoader(val_set, batch_size=self.config.batch_size,
                                            sampler=tdata.RandomSampler(val_set),
                                            num_workers=self.config.workers, drop_last=True)
            print('>>> Validation set ready.')

    def set_testloader(self, sono, test_idx):
        if self.config.meta_test:
            self.test_set = FrameSeqDataset(config=self.config, phase='meta_test', sono_list=self.test_sono, 
                                            test_index=test_idx, meta_test_dev_ratio=self.config.meta_test_dev_ratio)
            # print(f'meta test set size: {len(self.test_set)}')
            self.test_loader = tdata.DataLoader(self.test_set, batch_size=128, shuffle=False,
                                             num_workers=self.config.workers, drop_last=False)
        else:
            self.test_set = FrameSeqDataset(config=self.config, phase='test', sono_list=sono, test_index=test_idx)
            # note that during test, a batch contains all sequences from one scan, hence the large batch size
            self.test_loader = tdata.DataLoader(self.test_set, batch_size=128, shuffle=False,
                                                num_workers=self.config.workers, drop_last=False)
            print('>>> Test set ready.')


    def save_checkpoint(self, name, metric=-1):
        print(f'>>> Saving model {name} at epoch {self.epoch}')
        t = time.localtime()
        timestamp = time.strftime('%b-%d-%Y-%H%M', t)
        checkpoint = {
            'epoch': self.epoch,
            'lower': self.lower_model.state_dict(),
            'upper': self.upper_model.state_dict(),
        }
        file_name = os.path.join(self.model_path,f'{timestamp}_{self.epoch}_{name}_{metric:04.4f}.pth')
        torch.save(checkpoint, file_name)
        return file_name
    