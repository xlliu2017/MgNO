import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from utilities3 import *
import argparse
import operator
from functools import reduce
from functools import partial

from timeit import default_timer
import logging
from Adam import Adam
from models import MgNO_NS




def train(model, model_type, optimizer, scheduler, trainLossFunc, train_loader, train_l2_step, train_l2_full, dataOpt): #
        
    model.train()
    for xx, yy in train_loader:
        loss = 0
        xx, yy = xx.cuda(), yy.cuda()
        if model_type in ('FNO', 'UNO', 'FFNO', 'MWT', 'LSM'):
            for t in range(0, dataOpt['T'], dataOpt['step']):
                x = xx[..., t:t + dataOpt['T_in']]
                y = yy[..., t:t + dataOpt['step']]
                im = model(x)
                loss += trainLossFunc(im, y)

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)
        elif model_type == 'FMM':
            for t in range(0, dataOpt['T'], dataOpt['step']):
                x = xx[..., t:t + dataOpt['T_in']]
                y = yy[..., t:t + dataOpt['step']]
                im = model(x.permute(0, 3, 1, 2))
                loss += trainLossFunc(im, y)

        else:
            for t in range(0, dataOpt['T'], dataOpt['step']):
                x = xx[:, t:t + dataOpt['T_in'], ...]
                y = yy[:, t:t + dataOpt['step'], ...]
                im = model(x)
                loss += trainLossFunc(im, y)
                # h1_loss, train_f_l2loss, f_l2x, f_l2y = h1loss(im, y)
                # loss += h1_loss
                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), dim=1)

                
        
        train_l2_step += loss.item()


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    lr = optimizer.param_groups[0]['lr']
    scheduler.step()
    return  train_l2_step, 0, lr

def train_full_2(model, model_type, optimizer, scheduler, trainLossFunc, train_loader, train_l2_step, train_l2_full, dataOpt): #
        
    model.train()
    for xx, yy in train_loader:
        loss = 0
        # xx, yy = xx.cuda(), yy.cuda()
        if model_type in ('FNO', 'UNO', 'MWT', 'FFNO', 'LSM'):
            im = model(xx)
            loss += trainLossFunc(im, yy)

        elif model_type == 'FMM':
            for t in range(0, dataOpt['T'], dataOpt['step']):
                x = xx[..., t:t + dataOpt['T_in']]
                y = yy[..., t:t + dataOpt['step']]
                im = model(x.permute(0, 3, 1, 2))
                loss += trainLossFunc(im, y)
 
        else:        
            im = model(xx)
            loss += trainLossFunc(im, yy)
        
        train_l2_step += loss.item()


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    lr = optimizer.param_groups[0]['lr']
    scheduler.step()
    return  train_l2_step, 0, lr


def trainRNNly(model, model_type, optimizer, scheduler, trainLossFunc, train_loader, train_l2_step, train_l2_full, dataOpt): #
        
    model.train()
    for xx, yy in train_loader:
        loss = 0
        xx, yy = xx.cuda(), yy.cuda()
        if model_type in ('FNO', 'UNO', 'FFNO', 'MWT', 'LSM'):
            for t in range(0, dataOpt['T'], dataOpt['step']):
                
                y = yy[..., t:t + dataOpt['step']]
                im = model(xx)
                loss += trainLossFunc(im, y)

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., dataOpt['step']:], im), dim=-1)
        elif model_type == 'FMM':
            for t in range(0, dataOpt['T'], dataOpt['step']):
                
                y = yy[..., t:t + dataOpt['step']]
                im = model(xx.permute(0, 3, 1, 2))
                loss += trainLossFunc(im, y)
                # h1_loss, train_f_l2loss, f_l2x, f_l2y = h1loss(im, y)
                # loss += h1_loss
                if t == 0:
                    pred = im.unsqueeze(dim=-1)
                else:
                    pred = torch.cat((pred, im.unsqueeze(dim=-1)), -1)

                xx = torch.cat((xx[..., dataOpt['step']:], im.unsqueeze(dim=-1)), dim=-1)
        else:
            for t in range(0, dataOpt['T'], dataOpt['step']):
                
                y = yy[:, t:t + dataOpt['step'], ...]
                im = model(xx)
                loss += trainLossFunc(im, y)
                # h1_loss, train_f_l2loss, f_l2x, f_l2y = h1loss(im, y)
                # loss += h1_loss
                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), dim=1)

                xx = torch.cat((xx[:, dataOpt['step']:, ...], im), dim=1)

        train_l2_step += loss.item()
        l2_full = trainLossFunc(pred, yy)
        train_l2_full += l2_full.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    return  train_l2_step, train_l2_full, lr

@torch.no_grad()
def test(model, model_type, trainLossFunc, test_loader, test_l2_full, test_l2_full_2, test_l2_step, dataOpt): #
    
    for xx, yy in test_loader:
        loss = 0
        xx, yy = xx.cuda(), yy.cuda()
        if model_type in ('FNO', 'UNO', 'MWT', 'FFNO', 'LSM'):
            for t in range(0, dataOpt['T'], dataOpt['step']):
                y = yy[..., t:t + dataOpt['step']]
                im = model(xx)
                loss += trainLossFunc(im, y)

                if t == 0:
                    pred = im
                    loss1 = loss.clone().detach()
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., dataOpt['step']:], im), dim=-1)
        elif model_type == 'FMM':
            for t in range(0, dataOpt['T'], dataOpt['step']):
                y = yy[..., t:t + dataOpt['step']]
                im = model(xx.permute(0, 3, 1, 2))
                loss += trainLossFunc(im, y)

                if t == 0:
                    if dataOpt['step']==1:
                        pred = im.unsqueeze(dim=-1)
                    else:
                        pred = im
                    loss1 = loss.clone().detach()
                else:
                    if dataOpt['step']==1:
                        pred = torch.cat((pred, im.unsqueeze(dim=-1)), -1)
                    else:
                        pred = torch.cat((pred, im), -1)

                if dataOpt['step']==1:
                    xx = torch.cat((xx[..., dataOpt['step']:], im.unsqueeze(dim=-1)), dim=-1)
                else:
                    xx = torch.cat((xx[..., dataOpt['step']:], im), dim=-1)

        else:
            for t in range(0, dataOpt['T'], dataOpt['step']):
                y = yy[:, t:t + dataOpt['step'], ...]
                im = model(xx)
                loss += trainLossFunc(im, y)

                if t == 0:
                    pred = im
                    loss1 = loss.clone().detach()
                else:
                    pred = torch.cat((pred, im), dim=1)

                xx = torch.cat((xx[:, dataOpt['step']:, ...], im), dim=1)

        test_l2_step += loss.item()
        test_l2_full += trainLossFunc(pred, yy).item()
        test_l2_full_2 += loss1.item()
    return test_l2_full, test_l2_full_2, test_l2_step

def objective(modelOpt, dataOpt, model_type='MgNO_NS', model_save=True):
    ################################################################
    # configs
    ################################################################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_PATH = getSavePath(dataOpt['data'], model_type)
    MODEL_PATH_PARA = getSavePath(dataOpt['data'], model_type, flag='para')
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                    datefmt='%a %d %b %Y %H:%M:%S',
                    filename=MODEL_PATH,
                    filemode='w')
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(model_type)
    logging.info(modelOpt)
    logging.info(dataOpt)
    
    ################################################################
    # load data
    ################################################################

    train_a, train_u, test_a, test_u = getNavierDataSet3(dataOpt, device, return_normalizer=None, GN=None, normalizer=None)
    train_a = train_a.permute(0, 3, 1, 2).contiguous().to(device)
    train_u = train_u.permute(0, 3, 1, 2).contiguous().to(device)
    test_a = test_a.permute(0, 3, 1, 2).contiguous().to(device)
    test_u = test_u.permute(0, 3, 1, 2).contiguous().to(device)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=dataOpt['batch_size']*10, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=dataOpt['batch_size'], shuffle=False)

    ################################################################
    # training and evaluation
    ################################################################
    

    model = MgNO_NS(**modelOpt)
    model = model.to(device)
    logging.info(count_params(model))
    print(model)
    optimizer = Adam(model.parameters(), lr=dataOpt['learning_rate'], weight_decay=dataOpt['weight_decay'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                               max_lr=dataOpt['learning_rate'], 
                               div_factor=dataOpt['div_factor'], 
                               final_div_factor=dataOpt['final_div_factor'],
                               pct_start=0.1,
                               steps_per_epoch=1, 
                               epochs=dataOpt['epochs'])

    if dataOpt['loss_type'] == 'L2':
        trainLossFunc = LpLoss(size_average=False)
    else:
        trainLossFunc = HsLoss(d=2, p=2, k=1, size_average=False, a=[2.,], res=dataOpt['r'], return_freq=False, return_l2=False, truncation=True)
        trainLossFunc.cuda(device)
    l2loss = LpLoss(size_average=False)   
    trainFun = train_full_2
 
    for ep in range(dataOpt['epochs']):
        
        train_l2_step, train_l2_full, test_l2_step, test_l2_full, test_l2_full_2 = 0, 0, 0, 0, 0
        t1 = default_timer()
        train_l2_step, train_l2_full, lr = trainFun(model, model_type, optimizer, scheduler, trainLossFunc, train_loader, train_l2_step, train_l2_full, dataOpt)
        test_l2_full, test_l2_full_2, test_l2_step = test(model, model_type, l2loss, test_loader, test_l2_full, test_l2_full_2, test_l2_step, dataOpt)
        t2 = default_timer()
        desc = f" Epoch: {ep:3d}"
        desc += f"| time: {t2 - t1:.1f} "
        desc += f" | current lr: {lr:.3e}"
        desc += f"| train_l2_step: {train_l2_step / dataOpt['ntrain'] / (dataOpt['T'] / dataOpt['step']):.3e} "
        desc += f"| train_l2_full: {train_l2_full / dataOpt['ntrain']:.3e} "
        desc += f"| test_l2_step: {test_l2_step / dataOpt['ntest'] / (dataOpt['T'] / dataOpt['step']):.3e} "
        desc += f"| test_l2_full: {test_l2_full / dataOpt['ntest']:.3e} "
        desc += f"| test_l2_full_2: {test_l2_full_2 / dataOpt['ntest']:.3e} "
        
        logging.info(desc)
    if model_save:
        torch.save(model, MODEL_PATH_PARA)  
 

    

if __name__ == "__main__":
    

    import navier
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--data", type=str, default="1e-5", help="data name, darcy, darcy20c6, darcy15c10, darcyF, darcy_contin"
    )
    parser.add_argument(
            "--model_type", type=str, default="MgNO_NS", help="UNO, FNO, MgNO_NS"
    )
    parser.add_argument(
            "--epochs", type=int, default=500, help="number of epochs")
    parser.add_argument(
            "--batch_size", type=int, default=50, help="batch size"
    )
    parser.add_argument(
            "--lr", type=float, default=6e-4, help="learning rate")
    parser.add_argument(
            "--final_div_factor", type=float, default=100, help="final_div_factor")
    parser.add_argument(
            "--div_factor", type=float, default=2, help="div_factor")
    parser.add_argument(
            "--weight_decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument(
            "--loss_type", type=str, default="l2", help="loss type, l2, h1")

    parser.add_argument(
            "--num_layer", type=int, default=5, help="number of layers"
    )
    parser.add_argument(
            "--num_channel_u", type=int, default=32, help="number of channels in u"
    )
    parser.add_argument(
            "--num_channel_f", type=int, default=1, help="number of channels in f"
    )
    parser.add_argument(
            '--num_iteration', type=list, nargs='+', default=[[1,0], [1,0], [1,0], [2,0], [2,0]],  help='number of iterations in each layer')
    parser.add_argument('--padding_mode', type=str, default='circular', help='padding mode')
    parser.add_argument('--mlp_hidden_dim', type=int, default=0)
    parser.add_argument('--bias', action='store_true',)
    args = parser.parse_args()
    args = vars(args)

    for i in range(len(args['num_iteration'])):
        for j in range(len(args['num_iteration'][i])):
            args['num_iteration'][i][j] = int(args['num_iteration'][i][j])
    dataOpt = {}
    dataOpt['data'] = args['data']
    dataOpt['path'] =  getPath(args['data'], flag=None)#'/ibex/ai/home/liux0t/Xinliang/FMM/data/ns_V1e-3_N5000_T50.mat' #'/ibex/ai/home/liux0t/Xinliang/FMM/data/ns_V1e-4_N10000_T30.mat'#'/ibex/ai/home/liux0t/Xinliang/FMM/data/NavierStokes_V1e-5_N1200_T20.mat' ##
    dataOpt['ntrain'] = 1000
    dataOpt['ntest'] = 100
    dataOpt['batch_size'] = 50
    dataOpt['epochs'] = 500
    dataOpt['T_in'] = 1
    dataOpt['T_out'] = 1
    dataOpt['T'] = 10
    dataOpt['step'] = 1
    dataOpt['r'] = 64
    dataOpt['sampling'] = 1
    dataOpt['full_train'] = True
    dataOpt['full_train_2'] = True
    dataOpt['loss_type'] = 'L2'
    dataOpt['GN'] = False
    dataOpt['learning_rate'] = args['lr']
    dataOpt['final_div_factor'] = args['final_div_factor']
    dataOpt['div_factor'] = args['div_factor']
    dataOpt['weight_decay'] = args['weight_decay']

    

    modelOpt = {}
    modelOpt['num_layer'] = args['num_layer']
    modelOpt['in_chans'] = 1
    modelOpt['num_channel_u'] = args['num_channel_u']
    modelOpt['num_channel_f'] = args['num_channel_f']
    modelOpt['num_classes'] = 1
    modelOpt['num_iteration'] = args['num_iteration']
    modelOpt['mlp_hidden_dim'] = args['mlp_hidden_dim']
    modelOpt['output_dim'] = 1
    modelOpt['padding_mode'] = args['padding_mode']
    modelOpt['bias'] = args['bias']

    navier.objective(modelOpt, dataOpt, model_type=args['model_type'],)