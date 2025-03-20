
from data_provider.data_loader_ROVI import Dataset_Performance, Dataset_SOH, Dataset_SOHReg, \
    Dataset_1D, Dataset_1DVar
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

import torch.multiprocessing

data_dict = {
    'Performance': Dataset_Performance,
    'SOH': Dataset_SOH,
    '1D': Dataset_1D,
    '1DVar': Dataset_1DVar,
    'SOHReg':Dataset_SOHReg,
}

sharing_strategy = "file_system"
torch.multiprocessing.set_sharing_strategy(sharing_strategy)

def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    scaler = args.scaler

    if flag == 'test' or flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            worker_init_fn=set_worker_sharing_strategy,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            worker_init_fn=set_worker_sharing_strategy,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    elif args.task_name == 'regression':
                
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            size=[args.seq_len, 0, 1],
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            worker_init_fn=set_worker_sharing_strategy
            #collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            target=args.target,
            freq=freq,
            scaler_path=scaler
            )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            worker_init_fn=set_worker_sharing_strategy,
            drop_last=drop_last)
        return data_set, data_loader
