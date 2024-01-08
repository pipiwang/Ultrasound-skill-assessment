import torch
import torch.utils.data as tdata
import ujson
import random
from pathlib import Path
import torchio as tio

class FrameSeqDataset(tdata.Dataset):
    def __init__(self, config, phase, sono_list, test_index=0, meta_test_dev_ratio=0.4):
        assert phase in ['train', 'val', 'test', 'meta_train', 'meta_test', 'meta_val'], f"*** phase [{phase}] incorrect ***"
        self.config = config
        self.phase = phase
        self.data_dir = Path(config.data_dir).expanduser().resolve()
        self.meta_test_dev_ratio = meta_test_dev_ratio

        data_list = []
        if self.phase == 'test':
            with open(self.data_dir / 'Dataset' / f'{self.config.dataset_prefix}_{sono_list}.json') as f:
                data_list.extend(ujson.load(f))
            self.data_list = self.slice_test_data(data_list[test_index])
            self.scan_id = data_list[test_index]['scan_id']
            self.sp = data_list[test_index]['sp']
        elif self.phase == 'train' or self.phase == 'val':
            for sono in sono_list:
                with open(self.data_dir / 'Dataset' / f'{self.config.dataset_prefix}_{sono}.json') as f:
                    data_list.extend(ujson.load(f))
            # split query and support set
            patient_list = []
            for segment in data_list:
                patient_list.append(segment['patient_id'])
            patient_list = list(dict.fromkeys(patient_list))
            random.seed(42)
            random.shuffle(patient_list)
            self.data_list = []
            for segment in data_list:
                if self.phase == 'train':
                    if segment['patient_id'] in patient_list[:int(len(patient_list) * 0.5)]:
                        self.data_list.append(segment)
                elif self.phase == 'val':
                    if segment['patient_id'] in patient_list[int(len(patient_list) * 0.5):]:
                        self.data_list.append(segment)
        else:
            for sono in sono_list:
                with open(self.data_dir / 'Dataset' / f'{self.config.dataset_prefix}_{sono}.json') as f:
                    data_list.extend(ujson.load(f))
            # random split meta test developement set
            random.seed(42)
            random.shuffle(data_list)
            cutoff = int(len(data_list) * self.meta_test_dev_ratio)
            if cutoff % 2 == 1:
                cutoff += 1
            meta_test_dev = data_list[:cutoff]
            meta_test_test = data_list[cutoff:]
            print(f'meta eval dev {len(meta_test_dev)}, meta eval test {len(meta_test_test)}')
            if self.phase == 'meta_train' or self.phase == 'meta_val':
                if self.phase == 'meta_train':
                    self.data_list =  meta_test_dev[:int(len(meta_test_dev) * 0.5)]
                elif self.phase == 'meta_val':
                    self.data_list =  meta_test_dev[int(len(meta_test_dev) * 0.5):]
            elif self.phase == 'meta_test':
                self.data_list = self.slice_test_data(meta_test_test[test_index])
                self.scan_id = meta_test_test[test_index]['scan_id']
                self.sp = meta_test_test[test_index]['sp']
            else:
                raise NotImplementedError(f'*** Invalid phase {self.phase} for dataset! ***')
    

    def slice_test_data(self, segment):
        frame_nr_array = segment['frame_nr_array'][::self.config.seq_ds]
        frame_nr_array = sorted(frame_nr_array, reverse=True) # reverse the frame nr to avoid deleting the last seq (probs contains sp)
        if self.config.test_stride == 0:
            self.config.test_stride = self.config.seq_len
        data_list = [frame_nr_array[i:i+self.config.seq_len] for i in range(0, len(frame_nr_array)-self.config.test_stride, self.config.test_stride)]
        if len(data_list[-1]) != self.config.seq_len:
            del data_list[-1]
        data_list = sorted([sorted(seq) for seq in data_list])
        return data_list

    def transform(self, subject):
        trans = tio.Compose([
            tio.Resize((self.config.ds * subject.frame01.shape[1], self.config.ds * subject.frame01.shape[2], -1)),
            tio.RemapLabels({255: 1}),
            tio.ZNormalization(include=[key for key in subject if key.startswith('frame')])
        ])
        return trans(subject)

    def augmentation(self, subject):
        if self.config.aug == 'affine':
            aug = tio.RandomAffine(scales=(0.85, 1), degrees=10, )
        elif self.config.aug == 'hflip':
            aug = tio.RandomFlip(flip_probability=1, axes=('LR',))
        elif self.config.aug == 'mult':
            aug = tio.Compose([
                tio.RandomAffine(scales=(0.85, 1), degrees=10, p=0.5),
                tio.RandomFlip(flip_probability=1, axes=('LR',), p=0.5)
            ])
        elif self.config.aug == 'none':
            aug = None
        else:
            raise NotImplementedError("*** Unknown augmentation type! ***")
        if aug is not None:
            subject = aug(subject)
        return subject

    def __getitem__(self, index):
        empth_path = self.data_dir / 'empty_mask.png'

        if self.phase == 'test' or self.phase == 'meta_test':
            frame_nr_seq = self.data_list[index]
            # print(f'frame_nr_seq {len(frame_nr_seq)}')
        else:
            segment = self.data_list[index]
            self.scan_id = segment['scan_id']
            self.sp = segment['sp']
            frame_nr_seq = segment['frame_nr_array'][::self.config.seq_ds]
            start_idx = random.randint(0, len(frame_nr_seq) - self.config.seq_len)
            frame_nr_seq = frame_nr_seq[start_idx: start_idx + self.config.seq_len]
        start_frame = frame_nr_seq[0]

        sp_flag = 0
        
        for fr in self.sp:
            if min(frame_nr_seq) <= fr <= max(frame_nr_seq):
                sp_flag = 1
                break

        subject_dict = {}
        for fr_index,frame_nr in enumerate(frame_nr_seq):
            # get frame
            subject_dict[f'frame{fr_index:02d}'] = tio.ScalarImage(self.data_dir / self.scan_id /
                                    'frames' / f'frame{frame_nr:06d}_1008x784_576x448.jpg')
            # get masks
            for mask_idx, mask_label in enumerate(self.config.labels):
                mask_path = self.data_dir / self.scan_id / 'masks_strict' \
                            / f'frame{frame_nr:06d}_1008x784_576x448_{mask_label}.png'
                if mask_path.exists():
                    subject_dict[f'frame{fr_index:02d}_{mask_label}'] = tio.LabelMap(mask_path)
                else:
                    subject_dict[f'frame{fr_index:02d}_{mask_label}'] = tio.LabelMap(empth_path)

        subject = tio.Subject(subject_dict)
        subject = self.transform(subject)

        if not self.phase == 'test' and random.uniform(0,1) > 0.5:
            subject = self.augmentation(subject)

        frame_array, mask_array = [], []
        for fr_index in range(self.config.seq_len):
            frame_array.append(subject[f'frame{fr_index:02d}'].data.float().squeeze())
            for mask_idx, mask_label in enumerate(self.config.labels):
                mask_array.append(subject[f'frame{fr_index:02d}_{mask_label}'].data.float().squeeze())
        frame_array = torch.stack(frame_array)
        mask_array = torch.stack(mask_array)

        frame_array = frame_array.permute((0,2,1))
        mask_array = mask_array.permute((0,2,1))

        scan_start_id = f"{self.scan_id}_{start_frame:06d}"

        return frame_array, mask_array, scan_start_id, sp_flag


    def __len__(self):
        return len(self.data_list)

