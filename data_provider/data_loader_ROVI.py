import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
import random
import tqdm
from datetime import datetime
from joblib import dump, load

warnings.filterwarnings('ignore')

sharing_strategy = "file_system"
torch.multiprocessing.set_sharing_strategy(sharing_strategy)


class Dataset_ROVI(Dataset):
    """Base ROVI dataset class"""
    def __init__(self, root_path, flag='train', size=None,
                 data_path='./',
                 target='param', scale=False, freq='h', 
                 nsplits=10, scaler_path=None, scaler_cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        
        # init
        self.flag = flag
        assert flag in ['train', 'test', 'val', 'pred']
        type_map = {'train': 0, 'val': 1, 'test': 2, 'pred': 3}
        self.set_type = type_map[flag]

        self.target = target
        self.scale = scale
        self.freq = freq
        self.nsplits = nsplits
        self.scaler_path = scaler_path

        if scaler_cols is None:
            self.scaler_cols = [self.target]
        else:
            self.scaler_cols = scaler_cols


        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        fit_scaler = False
        try:
            self.scaler = load(self.scaler_path)
            print(f'Loaded {self.scaler_path}...')
        except:
            self.scaler = StandardScaler()
            fit_scaler = True
        
        fns = glob.glob(f'{self.root_path}/*/*.csv')
                        
        dfs = []
        for fn in tqdm.tqdm(fns):
            
            if self.flag != 'pred' and '/pred/' in fn:
                continue
            
            if not fit_scaler and self.flag == 'pred' and not '/pred/' in fn:
                continue
            
            file_id = fn.split('/')[-1]
            try:
                df_raw = pd.read_csv(fn)
            except:
                print(fn)
            
            df_raw['instance'] = file_id
            
            if 'train' in fn:
                df_raw['split'] = 'train'
            elif 'val' in fn:
                df_raw['split'] = 'val'
            elif 'test' in fn:
                df_raw['split'] = 'test'
            elif 'pred' in fn:
                df_raw['split'] = 'pred'

            dfs.append(df_raw)

        df_raw = pd.concat(dfs)

        if fit_scaler:
            train_idx = df_raw['split'] == 'train'
            train_data = df_raw.loc[train_idx]
            
            self.scaler.fit(train_data[self.scaler_cols].values)
            dump(self.scaler, self.scaler_path, compress=True)
        else:
            print('Not training scaler!')

        self.df_raw = df_raw[df_raw['split'] == self.flag]
        
        print(self.df_raw['split'].value_counts())

    def __getitem__(self, index):
                
        id = int(index)
                
        seq_x = self.data_x[id]
        seq_y = self.data_y[id]

        if hasattr(self, 'mask_x'):
            mask_x = self.mask_x[id]
            mask_y = self.mask_y[id]
        else:
            mask_x = np.ones(seq_x.shape[0])
            mask_y = np.ones(seq_y.shape[0])
        
        seq_x_mark = self.data_stamp_x[id]
        seq_y_mark = self.data_stamp_y[id]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, mask_x, mask_y, self.instance_ids[id]

    def __len__(self):
        return len(self.instance_ids)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_1DVarOld(Dataset_ROVI):
    """Forecasting of variable length individual degredation parameter curves"""
    def __init__(self, root_path, flag='train', size=None,
                 data_path='./',
                 target='param', scale=False,freq='h', 
                 nsplits=10, scaler_path=None):
        super().__init__(root_path, flag=flag, size=size, data_path=data_path,
                              target=target, scale=scale,freq=freq,nsplits=nsplits,
                              scaler_path=scaler_path)
        
        self.__process_data__()
   

    def __process_data__(self):

        self.data_x = []
        self.data_y = []
        self.mask_x = []
        self.mask_y = []
        self.data_stamp_x = []
        self.data_stamp_y = []
        self.instance_ids = []
        self.splits = []
        
        for g,grp in tqdm.tqdm(self.df_raw.groupby('instance')):
            
            if self.flag != 'pred':
                splits = np.random.choice(range(30,470),self.nsplits)
            else:
                splits = np.arange(30,len(grp) - 30)
                splits = np.arange(30,len(grp) - 30,50)

            for split in splits:
                inst_df = grp.copy()                 

                if self.flag != 'pred':
                    n_before = 500 - split
                    n_after = split
                else:
                    n_before = max([500 - split,0])
                    n_after = max([500 - (len(inst_df) - split),0])

                #df_before = inst_df.iloc[split - n_before:split]
                df_before = inst_df.head(n_before)
                df_before[[self.target]] = 0 
                df_before['mask'] = 0.0
                
                #df_after = inst_df.iloc[split:split + n_after]
                df_after = inst_df.tail(n_after)
                df_after[[self.target]] = 0 
                df_after['mask'] = 0.0
                
                inst_df['mask'] = 1.0

                inst_df = pd.concat([df_before,inst_df,df_after]).reset_index(drop=True)
                                
                if self.flag == 'pred':
                    start = max([0, split - 500])
                    end = min([n_before + split + 500,len(inst_df)])
                    inst_df = inst_df.iloc[start:end].reset_index()
                
                                                
                inst_df['timestamp'] = pd.date_range(datetime.today(), periods=len(inst_df), freq='12H').tolist()

                df_data = inst_df[[self.target]]

                if self.scale:
                    data = self.scaler.transform(df_data.values)
                else:
                    data = df_data.values


                df_stamp = inst_df[['timestamp']]
                df_stamp['date'] = pd.to_datetime(df_stamp['timestamp'])
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                data_stamp = df_stamp.drop(columns = ['timestamp','date']).values
                
                s_begin = 0
                s_end = 500
                r_begin = 500
                r_end = 1000


                self.instance_ids.append(f'{g}_{split}')

                
                self.data_x.append(data[s_begin:s_end])
                self.data_y.append(data[r_begin:r_end])  
                self.mask_x.append(inst_df['mask'].values[s_begin:s_end])
                self.mask_y.append(inst_df['mask'].values[r_begin:r_end])
                self.data_stamp_x.append(data_stamp[s_begin:s_end])
                self.data_stamp_y.append(data_stamp[r_begin:r_end])
                self.splits.append(split)
                



class Dataset_1D(Dataset_ROVI):
    """Forecasting of a single degredation curve"""
    def __init__(self, root_path, flag='train', size=None,
                  data_path='./',
                 target='param', scale=True, freq='h',
                 scaler_path=None):
        super().__init__(root_path, flag=flag, size=size, data_path=data_path,
                              target=target, scale=scale,freq=freq,
                              scaler_path=scaler_path)
        self.__process_data__()


    def __process_data__(self):
        self.data_x = []
        self.data_y = []
        self.data_stamp_x = []
        self.data_stamp_y = []
        self.instance_ids = []
        for g,grp in self.df_raw.groupby('instance'):         


            if len(grp) < self.seq_len + self.pred_len:
                len_append = self.seq_len + self.pred_len - len(grp)
                append = pd.concat([grp.head(1)]*len_append,axis=0).reset_index(drop=True)
                append['timestamp'] = pd.date_range(start=grp['timestamp'].iloc[-1],freq='24H',periods=len(append))
                append[self.target] = 0
                
                grp = pd.concat([grp,append],axis=0).reset_index(drop=True)

    
            df_data = grp[[self.target]]

            if self.scale:
                data = self.scaler.transform(df_data.values)
            else:
                data = df_data.values

            ## index as stamp
            #data_stamp = np.arange(len(grp))

            df_stamp = grp[['timestamp']]
            df_stamp['date'] = pd.to_datetime(df_stamp['timestamp'])
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(columns = ['timestamp','date']).values
            
            s_begin = 0
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len


            self.instance_ids.append(g)

            
            self.data_x.append(data[s_begin:s_end])
            self.data_y.append(data[r_begin:r_end])  
            self.data_stamp_x.append(data_stamp[s_begin:s_end])
            self.data_stamp_y.append(data_stamp[r_begin:r_end])
        
class Dataset_1DVar(Dataset_ROVI):
    """Forecasting of variable length individual degredation parameter curves"""
    def __init__(self, root_path, flag='train', size=None,
                 data_path='./',
                 target='param', scale=False,freq='h', 
                 nsplits=2, scaler_path=None):
        super().__init__(root_path, flag=flag, size=size, data_path=data_path,
                              target=target, scale=scale,freq=freq,nsplits=nsplits,
                              scaler_path=scaler_path)
        
        self.__process_data__()
   

    def __process_data__(self):

        self.data_x = []
        self.data_y = []
        self.data_stamp_x = []
        self.data_stamp_y = []
        self.instance_ids = []
        self.splits = []
        
        for g,grp in tqdm.tqdm(self.df_raw.groupby('instance')):

            if len(grp) < self.seq_len + self.pred_len:
                len_append = self.seq_len + self.pred_len - len(grp)
                append = pd.concat([grp.head(1)]*len_append,axis=0).reset_index(drop=True)
                append['timestamp'] = pd.date_range(start=grp['timestamp'].iloc[-1],freq='24H',periods=len(append))
                append[self.target] = 0
                
                grp = pd.concat([grp,append],axis=0).reset_index(drop=True)
            
            
            if self.flag != 'pred':
                splits = np.random.choice(range(50,self.seq_len),self.nsplits)
            else:
                splits = np.arange(0,self.seq_len,1)

            for split in splits:
                inst_df = grp.copy()                 

                n_before = self.seq_len - split

                df_before = inst_df.head(n_before)
                df_before[[self.target]] = grp[self.target].values[0]                

                inst_df = pd.concat([df_before,inst_df]).reset_index(drop=True)
                                
                inst_df = inst_df.head(self.seq_len + self.pred_len)
                                                
                inst_df['timestamp'] = pd.date_range(start='2024-12-18 13:09:16.959331',freq='24H', periods=len(inst_df)).tolist()

                df_data = inst_df[[self.target]]

                if self.scale:
                    data = self.scaler.transform(df_data.values)
                else:
                    data = df_data.values


                df_stamp = inst_df[['timestamp']]
                df_stamp['date'] = pd.to_datetime(df_stamp['timestamp'])
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                data_stamp = df_stamp.drop(columns = ['timestamp','date']).values
                
                s_begin = 0
                s_end = s_begin + self.seq_len
                r_begin = s_end - self.label_len
                r_end = r_begin + self.label_len + self.pred_len

                self.instance_ids.append(f'{g}_{split}')
                
                self.data_x.append(data[s_begin:s_end])
                self.data_y.append(data[r_begin:r_end])  
                self.data_stamp_x.append(data_stamp[s_begin:s_end])
                self.data_stamp_y.append(data_stamp[r_begin:r_end])
                self.splits.append(split)

class Dataset_SOH(Dataset_ROVI):
    """Multivariate state of health forecasting based on input performance and/or SOH parameter data."""
    def __init__(self, root_path, flag='train', size=None,
                 data_path='./',
                 target='V', scale=True, freq='h',
                 scaler=None,
                 #input_cols=['C_cap','Dis_cap','C_energy','Dis_energy','E_eff','C','sigma_m'],
                 #output_cols=['alpha_n', 'a', 'sigma_m', 'k_n', 'k_m', 'C', 'D_n', 'mu_n', 'E_n','k_eff', 'i_app', 'Q']
                 input_cols=['C_cap','Dis_cap','C_energy','Dis_energy','E_eff'],
                 output_cols=['sigma_m', 'k_n', 'C']):
        super().__init__(root_path, flag=flag, size=size, data_path=data_path,
                              target=target, scale=scale,freq=freq,
                              scaler=scaler,scaler_cols=output_cols)
        
        self.input_cols = input_cols
        self.output_cols = output_cols
        
        self.__process_data__()
        

    def __process_data__(self):
        self.data_x = []
        self.data_y = []
        self.data_stamp_x = []
        self.data_stamp_y = []
        self.instance_ids = []
        for g,grp in self.df_raw.groupby('instance'):
                    
            df_data_in = grp[self.output_cols]
            df_data_out = grp[self.output_cols]

            if self.scale:
                data_in = df_data_in.values 
                data_out = self.scaler.transform(df_data_out.values)
            else:
                data_in = df_data_in.values
                data_out = df_data_out.values


            df_stamp = grp[['timstamp']]
            df_stamp['date'] = pd.to_datetime(df_stamp['timstamp'])
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(columns = ['timstamp','date']).values
            
            s_begin = 0
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len


            self.instance_ids.append(g)

            
            self.data_x.append(data_in[s_begin:s_end])
            self.data_y.append(data_out[r_begin:r_end])  
            self.data_stamp_x.append(data_stamp[s_begin:s_end])
            self.data_stamp_y.append(data_stamp[r_begin:r_end])
        

class Dataset_SOHReg(Dataset_ROVI):
    """
    Regression of final value of degredation parameters from input performance curves.
    """
    def __init__(self, root_path, flag='train', size=None,
                  data_path='./',
                 target='V', scale=True, freq='h', 
                 scaler=None,
                 max_seq_len=500,
                 input_cols=['C_cap','Dis_cap','C_energy','Dis_energy','E_eff'],
                 output_cols=['sigma_m', 'C', 'E_n','i_app']):
        
        super().__init__(root_path, flag=flag, size=size, data_path=data_path,
                        target=target, scale=scale,freq=freq,
                        scaler=scaler,scaler_cols=input_cols + output_cols)
        
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.max_seq_len = max_seq_len
        
        self.__process_data__()
        

    def __process_data__(self):

        self.data_x = []
        self.data_y = []
        self.instance_ids = []
        
        for g,grp in self.df_raw.groupby('instance'):
                    
            if self.scale:
                grp = self.scaler.transform(grp[self.input_cols + self.output_cols])
                grp = pd.DataFrame(grp,columns = self.input_cols + self.output_cols)
                    
            feature_df = grp[self.input_cols].copy()
            labels_df = grp[self.output_cols].copy()

            data_in = feature_df.values
            data_out = labels_df.values
            
            s_begin = 0
            s_end = s_begin + self.seq_len

            self.instance_ids.append(g)
            
            self.data_x.append(data_in[s_begin:s_end])
            self.data_y.append(data_out[s_end,:])


    def __getitem__(self, index):
                
        id = int(index)
        
        seq_x = self.data_x[id]
        label = self.data_y[id]
        
        padding = np.ones_like(seq_x)

        return seq_x, label, padding, self.instance_ids[id]


    
class Dataset_Performance(Dataset_ROVI):
    """Forecast performance metrics from input cycles."""
    def __init__(self, root_path, flag='train', size=None,
                 data_path='./',
                 target='V', scale=True, freq='h', 
                 scaler=None,
                 cols=['C_cap','Dis_cap','C_energy','Dis_energy','E_eff','sigma_m', 'C', 'E_n','i_app'],
                 ):
        super().__init__(root_path, flag=flag, size=size, data_path=data_path,
                        target=target, scale=scale,freq=freq,
                        scaler=scaler,scaler_cols=cols)
        
        self.cols = cols
        
        self.__process_data__()

    def __process_data__(self):

        self.data_x = []
        self.data_y = []
        self.data_stamp_x = []
        self.data_stamp_y = []
        self.instance_ids = []
        for g,grp in self.df_raw.groupby('instance'):
                    
                    
            data = grp[self.cols].values
            

            df_stamp = grp[['timstamp']]
            df_stamp['date'] = pd.to_datetime(df_stamp['timstamp'])
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(columns = ['timstamp','date']).values
            
            s_begin = 0
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

            self.instance_ids.append(g)
            
            self.data_x.append(data[s_begin:s_end])
            self.data_y.append(data[r_begin:r_end])  
            self.data_stamp_x.append(data_stamp[s_begin:s_end])
            self.data_stamp_y.append(data_stamp[r_begin:r_end])