from data_provider.data_factory import data_provider_period, data_provider_trend
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate_period, adjust_learning_rate_trend, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw,accelerated_dtw
from utils.augmentation import run_augmentation,run_augmentation_single
from decimal import Decimal
import matplotlib.pyplot as plt
import threading
import math
import gc

from thop import profile

import random

warnings.filterwarnings('ignore')



class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model_period(self):
        model_period = self.model_dict[self.args.model_period].Model(self.args).float()
        return model_period
    
    def _build_model_trend(self):

        model_trend = self.model_dict[self.args.model_trend].Model(self.args).float()
        return model_trend

    def _get_data_period(self, flag, root_dir_period):
        data_set, data_loader = data_provider_period(self.args, flag, root_dir_period)
        return data_set, data_loader
    
    def _get_data_trend(self, flag, root_dir_trend):
        data_set, data_loader = data_provider_trend(self.args, flag, root_dir_trend)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-4)
        return model_optim

    def _select_criterion(self):
        # criterion = nn.MSELoss()
        criterion = nn.L1Loss()
        return criterion

    def vali(self, criterion):

        loss_records = []

        for folder_idx in range(1, 2): 
            print(f"\n Processing Folder WD_{folder_idx}...")

            # root_dir_period = os.path.join(root_dir_all, f'period_{folder_idx}')
            # root_dir_trend = os.path.join(root_dir_all, f'trend_{folder_idx}')

            root_dir_period = self.args.root_path_period
            root_dir_trend = self.args.root_path_trend

            file_list_period = [f for f in os.listdir(root_dir_period)]

            k = 0
            val_datas_period = []
            test_datas_period = []

            for file in file_list_period: 
                self.args.data_path = file
                test_data_period, test_loader_period = self._get_data_period(flag='test', root_dir_period=root_dir_period)

                test_datas_period.append(test_data_period)

            test_data_period =  torch.utils.data.ConcatDataset(test_datas_period)
            vali_data_period = test_data_period

            test_loader_period = torch.utils.data.DataLoader(test_data_period, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=False)
            vali_loader_period = test_loader_period

            train_steps = len(vali_loader_period)

            file_list_trend = file_list_period

            val_datas_trend = []
            test_datas_trend = []
                                            

            for file in file_list_trend: 
                self.args.data_path = file
                test_data_trend, test_loader_trend = self._get_data_trend(flag='test', root_dir_trend=root_dir_trend)

                test_datas_trend.append(test_data_trend)

            test_data_trend =  torch.utils.data.ConcatDataset(test_datas_trend)
            vali_data_trend = test_data_trend

            test_loader_trend = torch.utils.data.DataLoader(test_data_trend, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=False)
            vali_loader_trend = test_loader_trend


            vali_outputs_period, vali_trues_period = [], []
            vali_outputs_trend, vali_trues_trend = [], []

            vali_loss_period, vali_loss_trend = [], []
            self.model_period.eval()
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader_period):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float()

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model_period(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model_period(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model_period(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model_period(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features_period == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    pred = outputs.detach().cpu()
                    true = batch_y.detach().cpu()

                    vali_outputs_period.append(pred)
                    vali_trues_period.append(true)

                    loss_period = criterion(pred, true)

                    vali_loss_period.append(loss_period)
                
            self.model_period.train()

            self.model_trend.eval()
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader_trend):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float()

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model_trend(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model_trend(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model_trend(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model_trend(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features_trend == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    pred = outputs.detach().cpu()
                    true = batch_y.detach().cpu()

                    vali_outputs_trend.append(pred)
                    vali_trues_trend.append(true)

                    loss_trend = criterion(pred, true)

                    vali_loss_trend.append(loss_trend)

            self.model_trend.train()

            loss_total_list = []
            for op, ot, tp, tt in zip(vali_outputs_period, vali_outputs_trend, vali_trues_period, vali_trues_trend):
                pred_total = op + ot
                true_total = tp + tt
                loss_total = criterion(pred_total, true_total)
                loss_total_list.append(loss_total.item())

            loss_total = np.average(loss_total_list)
            loss_period = np.average(vali_loss_period)
            loss_trend = np.average(vali_loss_trend)

            alpha = 4 # α
            beta = 0.1   # β
            final_loss = loss_total + alpha * loss_trend + beta * loss_period

            loss_records.append((final_loss))

        avg_losses = np.mean(loss_records, axis=0)
        final_loss_avg= avg_losses

        
        return final_loss_avg

    def train(self, setting):
        
        def set_seed(seed):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False  

        set_seed(42)  

        start_time = time.time()
        print("开始时间",start_time)
        self.args.start_time = start_time


        path = os.path.join(self.args.checkpoints, setting)

        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        model_optim_period = optim.Adam(self.model_period.parameters(), lr=self.args.learning_rate_period, weight_decay=1e-4)
        model_optim_trend = optim.Adam(self.model_trend.parameters(), lr=self.args.learning_rate_trend, weight_decay=1e-4)
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        best_val_loss = float('inf')
        best_model_path = ''

        self.model_trend.train()
        self.model_period.train()

        for epoch in range(self.args.train_epochs):
            torch.cuda.reset_peak_memory_stats()  

            loss_records = []

            for folder_idx in range(1, 2):  
                print(f"\n[Epoch {epoch + 1}] Processing Folder WD_{folder_idx}...")

                # root_dir_period = os.path.join(root_dir_all, f'period_{folder_idx}')
                # root_dir_trend = os.path.join(root_dir_all, f'trend_{folder_idx}')

                root_dir_period = self.args.root_path_period
                root_dir_trend = self.args.root_path_trend

                file_list_period = [f for f in os.listdir(root_dir_period)]

                k = 0
                train_datas_period = []
                val_datas_period = []
                test_datas_period = []

                for file in file_list_period: 
                    self.args.data_path = file
                    train_data_period, train_loader_period = self._get_data_period(flag='train', root_dir_period=root_dir_period)

                    train_datas_period.append(train_data_period)


                train_data_period =  torch.utils.data.ConcatDataset(train_datas_period)

                train_loader_period = torch.utils.data.DataLoader(train_data_period, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, pin_memory=False)

                train_steps = len(train_loader_period)

                file_list_trend = file_list_period

                k = 0
                train_datas_trend = []
                val_datas_trend = []
                test_datas_trend = []
                                                

                for file in file_list_trend: 
                    self.args.data_path = file
                    train_data_trend, train_loader_trend = self._get_data_trend(flag='train', root_dir_trend=root_dir_trend)
                    train_datas_trend.append(train_data_trend)

                train_data_trend =  torch.utils.data.ConcatDataset(train_datas_trend)

                train_loader_trend = torch.utils.data.DataLoader(train_data_trend, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, pin_memory=False)

                outputs_period, trues_period = [], []
                outputs_trend, trues_trend = [], []

                train_loss_period, train_loss_trend = [], []


                iter_count = 0
                train_loss = []

                epoch_time = time.time()
                
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader_period):
                    iter_count += 1
                    model_optim_period.zero_grad()
                    batch_x = batch_x.float().to(self.device)
                    # print(batch_x.shape) #torch.Size([32, 60, 8])
                    batch_y = batch_y.float().to(self.device)
                    # print(batch_y.shape) #torch.Size([32, 61, 8])
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                    # print(batch_x_mark.shape,batch_x_mark.shape) #torch.Size([32, 60, 4]) torch.Size([32, 60, 4])
                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model_period(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model_period(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                            f_dim = -1 if self.args.features_period == 'MS' else 0
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                            loss_period = criterion(outputs, batch_y)
                            train_loss_period.append(loss_period.item())
                    else:
                        if self.args.output_attention:
                            outputs = self.model_period(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model_period(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features_period == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        outputs_period.append(outputs)

                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        trues_period.append(batch_y)

                        loss_period = criterion(outputs, batch_y)
                        train_loss_period.append(loss_period.item())

                    if (i + 1) % 1000 == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss_period.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                    loss_period.backward()
                    model_optim_period.step()

                
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader_trend):
                    iter_count += 1
                    model_optim_trend.zero_grad()
                    batch_x = batch_x.float().to(self.device)
                    # print(batch_x.shape) #torch.Size([32, 60, 8])
                    batch_y = batch_y.float().to(self.device)
                    # print(batch_y.shape) #torch.Size([32, 61, 8])
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                    # print(batch_x_mark.shape,batch_x_mark.shape) #torch.Size([32, 60, 4]) torch.Size([32, 60, 4])
                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model_trend(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model_trend(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                            f_dim = -1 if self.args.features_trend == 'MS' else 0
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                            loss_trend = criterion(outputs, batch_y)
                            train_loss_period.append(loss_trend.item())
                    else:
                        if self.args.output_attention:
                            outputs = self.model_trend(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model_trend(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features_trend == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        # print("outputs")
                        # print(outputs)
                        outputs_trend.append(outputs)
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        # print("batch_y")
                        # print(batch_y)
                        trues_trend.append(batch_y)
                        loss_trend = criterion(outputs, batch_y)
                        train_loss_trend.append(loss_trend.item())

                    if (i + 1) % 1000 == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss_trend.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                    loss_trend.backward()
                    model_optim_trend.step()

                loss_total_list = []
                for op, ot, tp, tt in zip(outputs_period, outputs_trend, trues_period, trues_trend):

                    pred_total = op + ot
                    true_total = tp + tt
                    loss_total = criterion(pred_total, true_total)
                    loss_total_list.append(loss_total.item())


                loss_total = np.average(loss_total_list)
                loss_period = np.average(train_loss_period)
                loss_trend = np.average(train_loss_trend)


                alpha = 4  
                beta = 0.1   
                final_loss = loss_total + alpha * loss_trend + beta * loss_period

                loss_records.append((loss_period, loss_trend, loss_total, final_loss))

                peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
                peak_mem_str = f"Epoch {epoch + 1} Peak GPU Memory: {peak_memory:.0f} MiB"
                print(peak_mem_str)
                with open("cost.txt", "a") as f:
                    f.write(peak_mem_str + "\n")

                
            vali_loss_avg = self.vali(criterion)

            avg_losses = np.mean(loss_records, axis=0)
            loss_period_avg, loss_trend_avg, loss_total_avg, final_loss_avg = avg_losses

            print(f"\n[Epoch {epoch + 1}] AVERAGE -> Final Loss: {final_loss_avg:.6f} | Total: {loss_total_avg:.6f} | Trend: {loss_trend_avg:.6f} | Period: {loss_period_avg:.6f} | Vali: {vali_loss_avg:.6f}")

            with open("losses.txt", "a") as f:
                f.write(f"Epoch {epoch + 1}\n")
                f.write(f"Avg Period Loss: {loss_period_avg:.6f}\n")
                f.write(f"Avg Trend Loss: {loss_trend_avg:.6f}\n")
                f.write(f"Avg Total Loss: {loss_total_avg:.6f}\n")
                f.write(f"Avg Final Loss: {final_loss_avg:.6f}\n")
                f.write(f"Avg Vali Loss: {vali_loss_avg:.6f}\n\n")

            model_period_path = os.path.join(path, f'checkpoint_period_epoch_{epoch + 1}.pth')
            torch.save(self.model_period.state_dict(), model_period_path)
            model_trend_path = os.path.join(path, f'checkpoint_trend_epoch_{epoch + 1}.pth')
            torch.save(self.model_trend.state_dict(), model_trend_path)

            if vali_loss_avg < best_val_loss:
                best_val_loss = vali_loss_avg
                best_model_period_path = model_period_path
                best_model_trend_path = model_trend_path

            adjust_learning_rate_period(model_optim_period, epoch + 1, self.args)
            adjust_learning_rate_trend(model_optim_trend, epoch + 1, self.args)

        print(f"Loading best model based on validation MSE from {best_model_period_path}")
        self.args.bestcheckpoints_period = best_model_period_path
        self.args.bestcheckpoints_trend = best_model_trend_path
        self.model_period.load_state_dict(torch.load(best_model_period_path))
        self.model_trend.load_state_dict(torch.load(best_model_trend_path))

        return self.model_period, self.model_trend

##################################################################################################

    def test(self, setting, test=0):

        all_preds_period = []
        all_trues_period = []
        all_preds_trend = []
        all_trues_trend = []

        if test == 0:

            end_time1 = time.time()

            start_time1 = self.args.start_time

            total_time1 = end_time1 - start_time1
            hours1, rem1 = divmod(total_time1, 3600)
            minutes1, seconds1 = divmod(rem1, 60)
    
            f = open("time.txt", 'a')
            f.write(setting + "  \n")
            f.write('hours:{}, minutes:{}, seconds:{}'.format(int(hours1), int(minutes1), int(seconds1)))
            f.write('\n')
            f.write('\n')
            f.close()


        print("test_period........")
        root_dir_period = self.args.root_path_period
        file_list = [f for f in os.listdir(root_dir_period)]
        test_datas = []
        mae_period_total = 0
        mse_period_total = 0
        rmse_period_total = 0
        mape_period_total = 0
        r2_period_total = 0

        path = os.path.join(self.args.checkpoints, setting)
        best_model_path = os.path.join(path, f'checkpoint_period_epoch_10.pth')

        iii = 0
        for file in file_list:

            iii += 1 
            print("读取文件：", file)
            self.args.data_path = file

            test_data, test_loader = self._get_data_period(flag='test', root_dir_period=root_dir_period)
            test_datas.append(test_data)

            #pdb.set_trace()
            import torch
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)
            
            if test:
                self.model_period.load_state_dict(torch.load(best_model_path), strict=False)   

            preds = []
            trues = []

            self.model_period.eval()
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):

                    #pdb.set_trace()
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    # encoder - decoder
                    
                    # print("batch_x")
                    # print(batch_x.shape)
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model_period(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model_period(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model_period(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model_period(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    # print("output1")
                    # print(outputs.shape)
                    f_dim = -1 if self.args.features_period == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, :]
                    batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                    # print("output2")
                    # print(outputs.shape)
                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()

                    if test_data.scale and self.args.inverse:
                        shape = outputs.shape

                        output = outputs.reshape(-1, outputs.shape[-1])
                        output = test_data.inverse_transform(output)

                        outputs = output.reshape(outputs.shape)

                        batch_ys = batch_y.reshape(-1, batch_y.shape[-1])
                        batch_ys = test_data.inverse_transform(batch_ys)
                        batch_y = batch_ys.reshape(batch_y.shape)
            
                    outputs = outputs[:, :, f_dim:]
                    batch_y = batch_y[:, :, f_dim:]

                    pred = outputs
                    true = batch_y

                    preds.append(pred)
                    trues.append(true)

            preds = np.concatenate(preds,axis=0)
            trues = np.concatenate(trues,axis=0)
            preds = preds.squeeze(axis=-1)
            trues = trues.squeeze(axis=-1)


            directory = 'Rovegliana_WD_MS_Mamba_dmdf256_el1_dl1_fc1_ex2_dc2_nh4_train_period'

            folder_path = './results/' + directory + '/' + file + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            # dtw calculation
            if self.args.use_dtw:
                dtw_list = []
                manhattan_distance = lambda x, y: np.abs(x - y)
                for i in range(preds.shape[0]):
                    x = preds[i].reshape(-1,1)
                    y = trues[i].reshape(-1,1)
                    if i % 100 == 0:
                        print("calculating dtw iter:", i)
                    d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                    dtw_list.append(d)
                dtw = np.array(dtw_list).mean()
            else:
                dtw = -999
                

            mae, mse, rmse, mape, mspe, smape, r2 = metric(preds, trues)
            print('mse:{}, mae:{}, rmse:{}, mape:{}, smape:{}, r2:{}'.format(mse, mae, rmse, mape, smape, r2))
            np.save(folder_path + 'metrics.npy', np.array([mse, mae, rmse, mape, smape, r2]))
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)
            
            mse_period_total += mse
            mae_period_total += mae
            rmse_period_total += rmse
            mape_period_total += mape
            r2_period_total += r2

            if (
                math.isnan(mae) or 
                math.isnan(mse) or 
                math.isnan(rmse) or 
                math.isnan(mape) or 
                math.isnan(r2)
            ):
                with open("Nan_file.txt", "a") as f:
                    f.write(f"period NaN detected in file: {file}\n")
            
            all_preds_period.append(preds)
            all_trues_period.append(trues)
            
        mse_period_average = mse_period_total / iii
        mae_period_average = mae_period_total / iii
        rmse_period_average = rmse_period_total / iii
        mape_period_average = mape_period_total / iii
        r2_period_average = r2_period_total / iii
        

        print("test_trend........")
        root_dir_trend = self.args.root_path_trend

        file_list = file_list
        test_datas = []
        mae_trend_total = 0
        mse_trend_total = 0
        rmse_trend_total = 0
        mape_trend_total = 0
        r2_trend_total = 0

        path = os.path.join(self.args.checkpoints, setting)
        best_model_path = os.path.join(path, f'checkpoint_trend_epoch_10.pth')

        iii = 0
        for file in file_list:

            iii += 1 
            print("读取文件：", file)
            self.args.data_path = file

            test_data, test_loader = self._get_data_trend(flag='test', root_dir_trend=root_dir_trend)
            test_datas.append(test_data)

            import torch
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)
            
            if test:
                self.model_trend.load_state_dict(torch.load(best_model_path), strict=False)   

            preds = []
            trues = []

            self.model_trend.eval()
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):

                    #pdb.set_trace()
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    # encoder - decoder
                    
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model_trend(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model_trend(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model_trend(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model_trend(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features_trend == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, :]
                    batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()
                    #pdb.set_trace()
                    if test_data.scale and self.args.inverse:
                        shape = outputs.shape

                        output = outputs.reshape(-1, outputs.shape[-1])
                        output = test_data.inverse_transform(output)

                        outputs = output.reshape(outputs.shape)

                        batch_ys = batch_y.reshape(-1, batch_y.shape[-1])
                        batch_ys = test_data.inverse_transform(batch_ys)
                        batch_y = batch_ys.reshape(batch_y.shape)
            
                    outputs = outputs[:, :, f_dim:]
                    batch_y = batch_y[:, :, f_dim:]

                    pred = outputs
                    true = batch_y

                    preds.append(pred)
                    trues.append(true)
   
            #pdb.set_trace()
            preds = np.concatenate(preds,axis=0)
            trues = np.concatenate(trues,axis=0)

            preds = preds.squeeze(axis=-1)
            trues = trues.squeeze(axis=-1)

            # # Define the directory and file path
            directory = 'Rovegliana_WD_MS_Mamba_dmdf256_el1_dl1_fc1_ex2_dc2_nh4_train_trend'

            folder_path = './results/' + directory + '/' + file + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            # dtw calculation
            if self.args.use_dtw:
                dtw_list = []
                manhattan_distance = lambda x, y: np.abs(x - y)
                for i in range(preds.shape[0]):
                    x = preds[i].reshape(-1,1)
                    y = trues[i].reshape(-1,1)
                    if i % 100 == 0:
                        print("calculating dtw iter:", i)
                    d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                    dtw_list.append(d)
                dtw = np.array(dtw_list).mean()
            else:
                dtw = -999

            mae, mse, rmse, mape, mspe, smape, r2 = metric(preds, trues)
            print('mse:{}, mae:{}, rmse:{}, mape:{}, smape:{}, r2:{}'.format(mse, mae, rmse, mape, smape, r2))
            np.save(folder_path + 'metrics.npy', np.array([mse, mae, rmse, mape, smape, r2]))
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)
            
            mse_trend_total += mse
            mae_trend_total += mae
            rmse_trend_total += rmse
            mape_trend_total += mape
            r2_trend_total += r2


            if (
                math.isnan(mae) or 
                math.isnan(mse) or 
                math.isnan(rmse) or 
                math.isnan(mape) or 
                math.isnan(r2)
            ):

                with open("Nan_file.txt", "a") as f:
                    f.write(f"trend NaN detected in file: {file}\n")

            all_preds_trend.append(preds)
            all_trues_trend.append(trues)
            
            
        mse_trend_average = mse_trend_total / iii
        mae_trend_average = mae_trend_total / iii
        rmse_trend_average = rmse_trend_total / iii
        mape_trend_average = mape_trend_total / iii
        r2_trend_average = r2_trend_total / iii



        print("bestcheckpoints_period")
        print(self.args.bestcheckpoints_period)

        print('mse_period:',mse_period_average)
        print('mae_period:',mae_period_average)
        print('rmse_period:',rmse_period_average)
        print('mape_period:',mape_period_average)
        print('r2_period:', r2_period_average)
 
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse_period:{}, mae_period:{}, rmse_period:{}, mape_period:{}, r2_period:{}'.format(mse_period_average, mae_period_average, rmse_period_average, mape_period_average, r2_period_average))
        f.write('\n')
        f.write('\n')
        f.close()


        
        print('mse_trend:',mse_trend_average)
        print('mae_trend:',mae_trend_average)
        print('rmse_trend:',rmse_trend_average)
        print('mape_trend:',mape_trend_average)
        print('r2_trend:', r2_trend_average)

        end_time = time.time()
        print("结束时间", end_time)

        start_time = self.args.start_time

        total_time = end_time - start_time
        hours, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)
 
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse_trend:{}, mae_trend:{}, rmse_trend:{}, map_trende:{}, r2_trend:{}'.format(mse_trend_average, mae_trend_average, rmse_trend_average, mape_trend_average, r2_trend_average))
        f.write('\n')
        f.write('\n')
        f.close()


        maes, mses, rmses, mapes, mspes, smapes, r2s = [], [], [], [], [], [], []

        for p_period, p_trend, t_period, t_trend in zip(all_preds_period, all_preds_trend, all_trues_period, all_trues_trend):
            pred = p_period + p_trend  
            true = t_period + t_trend  

            mae, mse, rmse, mape, mspe, smape, r2 = metric(pred, true)

            maes.append(mae)
            mses.append(mse)
            rmses.append(rmse)
            mapes.append(mape)
            mspes.append(mspe)
            smapes.append(smape)
            r2s.append(r2)

        avg_mae = np.mean(maes)
        avg_mse = np.mean(mses)
        avg_rmse = np.mean(rmses)
        avg_mape = np.mean(mapes)
        avg_mspe = np.mean(mspes)
        avg_smape = np.mean(smapes)
        avg_r2 = np.mean(r2s)

        print(f"MAE: {avg_mae}, MSE: {avg_mse}, RMSE: {avg_rmse}, MAPE: {avg_mape}, R2: {avg_r2}")

        return

