from data_provider.data_factory_ROVI import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pdb
from sklearn.metrics import r2_score
import pandas as pd

warnings.filterwarnings('ignore')


class Exp_Regression(Exp_Basic):
    def __init__(self, args):
        super(Exp_Regression, self).__init__(args)

    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag='train')
        test_data, test_loader = self._get_data(flag='test')
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = len(train_data.input_cols)
        self.args.num_class = len(train_data.output_cols)
        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion  = nn.MSELoss()
        #criterion = nn.CrossEntropyLoss()
        return criterion
    
    def make_output_df(self, all_ids, trues, preds, dataset):
        
        temp_trues = np.concatenate( [np.ones( (trues.shape[0],len(dataset.input_cols)) ), trues], axis=1 )
        temp_preds = np.concatenate( [np.ones( (trues.shape[0],len(dataset.input_cols))), preds], axis=1 )
    
        
        output = pd.concat([pd.DataFrame({'id':all_ids}),
                            pd.DataFrame(dataset.inverse_transform(temp_trues)),
                            pd.DataFrame(dataset.inverse_transform(temp_preds))],
                           axis=1)
        output_columns = ['id']
        output_columns += [f'{c}_true_in' for c in dataset.input_cols]
        output_columns += [f'{c}_true' for c in dataset.output_cols]
        output_columns += [f'{c}_pred_in' for c in dataset.input_cols]
        output_columns += [f'{c}_pred' for c in dataset.output_cols]
        output.columns = output_columns
        output = output[[c for c in output.columns if '_in' not in c]]
        
        return(output)

    def vali(self, vali_data, vali_loader, criterion, setting, split='val', best_score=np.inf):
        total_loss = []
        preds = []
        trues = []
        all_ids = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask, ids) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.long().squeeze().cpu())
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)
                all_ids += ids

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        #probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        #predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        
        r2 = {}
        for i in range(4):
            r2[i] = r2_score(trues[:,i].cpu().numpy(),preds[:,i].cpu().numpy())
        
        predictions = preds.cpu().numpy()
        trues = trues.cpu().numpy()
        #accuracy = cal_accuracy(predictions, trues)
    
        output = self.make_output_df(all_ids, trues,predictions, vali_data)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if total_loss < best_score:
            print(f'Saving {split} predictions...')
            output.to_csv(f'{folder_path}/{split}_predictions.csv',index=False)


        self.model.train()
        return total_loss, r2

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        best_score = np.inf
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask, ids) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)
                                
                loss = criterion(outputs, label.float().squeeze(-1))
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            
            vali_loss, vali_r2 = self.vali(vali_data, vali_loader, criterion, setting, 'val', best_score=best_score)
            test_loss, test_r2 = self.vali(test_data, test_loader, criterion, setting, 'test')
            
            if vali_loss < best_score:
                best_score = vali_loss

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali R2: {4} Test Loss: {5:.3f} Test R2: {6}"
                .format(epoch + 1, train_steps, train_loss, vali_loss, vali_r2, test_loss, test_r2))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if (epoch + 1) % 5 == 0:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        all_ids = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask, ids) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                preds.append(outputs.detach())
                trues.append(label)
                all_ids += ids

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)

        #probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        #predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        predictions = preds.cpu().numpy()
        trues = trues.cpu().numpy()
        #accuracy = cal_accuracy(predictions, trues)
        rmse = np.sqrt(np.mean(np.power(trues.flatten() - predictions.flatten(),2)))

        output = self.make_output_df(all_ids,trues,predictions, test_data)


        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        output.to_csv(f'{folder_path}/test_predictions.csv',index=False)

        #print('accuracy:{}'.format(accuracy))
        file_name='result_regression.txt'
        f = open(os.path.join(folder_path,file_name), 'a')
        f.write(setting + "  \n")
        f.write('rmse:{}'.format(rmse))
        f.write('\n')
        f.write('\n')
        f.close()
        return
