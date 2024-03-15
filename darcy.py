import torch
import torch.nn as nn
import torch.nn.functional as F

from models import MgNO_DC, MgNO_DC_smooth
import os, logging
import numpy as np
import matplotlib.pyplot as plt

from utilities3 import *
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import argparse
torch.set_printoptions(threshold=100000)


    
def objective(dataOpt, modelOpt, optimizerScheduler_args,
                tqdm_disable=True, 
              log_if=False, validate=False, model_type='MgNO', 
              model_save=False, tune_if=False,):
    
    ################################################################
    # configs
    ################################################################

    print(os.path.basename(__file__))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_PATH = getSavePath(dataOpt['data'], model_type)
    MODEL_PATH_PARA = getSavePath(dataOpt['data'], model_type, flag='para')
    if log_if:
        logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                        datefmt='%a %d %b %Y %H:%M:%S',
                        filename=MODEL_PATH,
                        filemode='w')
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.info(model_type)
        logging.info(f"dataOpt={dataOpt}")
        logging.info(f"modelOpt={modelOpt}")
        logging.info(f"optimizerScheduler_args={optimizerScheduler_args}")
  
    
    ################################################################
    # load data and data normalization
    ################################################################
   
    if dataOpt['data'] in {'darcy', 'darcy20c6', 'darcy15c10', 'darcyF', 'darcy_contin', 'a4f1'}:
        
        x_train, y_train, x_normalizer, y_normalizer = getDarcyDataSet(dataOpt, flag='train', return_normalizer=True)
        x_test, y_test = getDarcyDataSet(dataOpt, flag='test', return_normalizer=False, normalizer=x_normalizer)
        x_val, y_val = getDarcyDataSet(dataOpt, flag='val', return_normalizer=False, normalizer=x_normalizer)
        
    elif dataOpt['data'] == 'helm':
        x_train, y_train, x_test, y_test, x_val, y_val, x_normalizer, y_normalizer = getHelmDataset(dataOpt)
    elif dataOpt['data'] == 'pipe':
        x_train, y_train, x_test, y_test, x_val, y_val = getPipeDataset(dataOpt)
    else: 
        raise NameError('dataset not exist')

    if modelOpt['normalizer']:
        modelOpt['normalizer'] = y_normalizer

    if x_train.ndim == 3:
        x_train = x_train[:, np.newaxis, ...]
        x_test = x_test[:, np.newaxis, ...]
        x_val = x_val[:, np.newaxis, ...]

    train_loader = DataLoader(TensorDataset(x_train.contiguous().to(device), y_train.contiguous().to(device)), batch_size=dataOpt['batch_size'], shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test.contiguous().to(device), y_test.contiguous().to(device)), batch_size=dataOpt['batch_size'], shuffle=False)
    val_loader = DataLoader(TensorDataset(x_val.contiguous().to(device), y_val.contiguous().to(device)), batch_size=dataOpt['batch_size'], shuffle=False)

    ################################################################
    # training and evaluation
    ################################################################
    if dataOpt['data'] == 'darcy20c6':
        model = MgNO_DC(**modelOpt).to(device)
    elif dataOpt['data'] == 'darcy':
        model = MgNO_DC_smooth(**modelOpt).to(device)
    elif dataOpt['data'] == 'pipe':
        model = MgNO_DC(**modelOpt).to(device)

    if log_if:    
        logging.info(count_params(model))
    optimizer, scheduler = getOptimizerScheduler(model.parameters(), **optimizerScheduler_args)
    
    h1loss = HsLoss(d=2, p=2, k=1, size_average=False, res=y_train.size(1),)
    h1loss.cuda(device)
    if dataOpt['data'] == 'helm':
        h1loss = HSloss_d()
    l2loss = LpLoss(size_average=False)  
    ############################
    def train(train_loader):
        model.train()
        train_l2, train_h1 = 0, 0
        train_f_dist = torch.zeros(y_train.size(1))

        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)
            if dataOpt['loss_type']=='h1':
                with torch.no_grad():
                    train_l2loss = l2loss(out, y)

                train_h1loss, train_f_l2loss, f_l2x, f_l2y = h1loss(out, y)
                train_h1loss.backward()
            else:
                with torch.no_grad():
                    train_h1loss, train_f_l2loss, f_l2x, f_l2y = h1loss(out, y)

                train_l2loss = l2loss(out, y)
                train_l2loss.backward()

            optimizer.step()
            train_h1 += train_h1loss.item()
            train_l2 += train_l2loss.item()
            train_f_dist += sum(torch.squeeze(torch.abs(f_l2x-f_l2y))).cpu()

        train_l2/= len(dataOpt['dataSize']['train'])
        train_h1/= len(dataOpt['dataSize']['train'])
        train_f_dist/= len(dataOpt['dataSize']['train'])
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()
         
        return lr, train_l2, train_h1, train_f_dist
            
    @torch.no_grad()
    def test(test_loader):
        model.eval()
        test_l2, test_h1 = 0., 0.

        with torch.no_grad():
            for x, y in test_loader:
                out = model(x)
                test_l2 += l2loss(out, y).item()
                test_h1 += h1loss(out, y)[0].item()
        test_l2/= len(dataOpt['dataSize']['test'])
        test_h1/= len(dataOpt['dataSize']['test'] )          
        
        return  test_l2, test_h1

        
    ############################  
    ###### start to train ######
    ############################
    
    train_h1_rec, train_l2_rec, test_l2_rec, test_h1_rec = [], [], [], []
    val_l2_rec, val_h1_rec = [], [],
    
    best_l2, best_test_l2, best_test_h1, arg_min_epoch = 1.0, 1.0, 1.0, 0  
    with tqdm(total=optimizerScheduler_args['epochs'], disable=tqdm_disable) as pbar_ep:
                            
        for epoch in range(optimizerScheduler_args['epochs']):
            desc = f"epoch: [{epoch+1}/{optimizerScheduler_args['epochs']}]"
            lr, train_l2, train_h1, train_f_dist = train(train_loader)
            test_l2, test_h1 = test(test_loader)
            val_l2, val_h1 = test(val_loader)
            
            train_l2_rec.append(train_l2); train_h1_rec.append(train_h1) 
            test_l2_rec.append(test_l2); test_h1_rec.append(test_h1)
            val_l2_rec.append(val_l2); val_h1_rec.append(val_h1)

            if val_l2 < best_l2:
                best_l2 = val_l2
                arg_min_epoch = epoch
                best_test_l2 = test_l2
                best_test_h1 = test_h1
           
            desc += f" | current lr: {lr:.3e}"
            desc += f"| train l2 loss: {train_l2:.3e} "
            desc += f"| train h1 loss: {train_h1:.3e} "
            desc += f"| test l2 loss: {test_l2:.3e} "
            desc += f"| test h1 loss: {test_h1:.3e} "
            desc += f"| val l2 loss: {val_l2:.3e} "
            desc += f"| val h1 loss: {val_h1:.3e} "
           
            pbar_ep.set_description(desc)
            pbar_ep.update()
            if log_if:
                logging.info(desc) 


        if log_if: 
            logging.info(f" test h1 loss: {best_test_h1:.3e}, test l2 loss: {best_test_l2:.3e}")
                 
        if log_if:
            logging.info('train l2 rec:')
            logging.info(train_l2_rec)
            logging.info('train h1 rec:')
            logging.info(train_h1_rec)
            logging.info('test l2 rec:')
            logging.info(test_l2_rec)
            logging.info('test h1 rec:')
            logging.info(test_h1_rec)
            
 
            
    if model_save:
        torch.save(model, MODEL_PATH_PARA)
            
    return test_l2







if __name__ == "__main__":

    import darcy
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--data", type=str, default="darcy20c6", help="data name, darcy, darcy20c6, darcy15c10, darcyF, darcy_contin")
    parser.add_argument(
            "--model_type", type=str, default="MgNO_DC", help="FNO, MgNO")
    parser.add_argument(
            "--epochs", type=int, default=500, help="number of epochs")
    parser.add_argument(
            "--batch_size", type=int, default=8, help="batch size")
    parser.add_argument(
            "--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument(
            "--final_div_factor", type=float, default=100, help="final_div_factor")
    parser.add_argument(
            "--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument(
            "--loss_type", type=str, default="h1", help="loss type, l2, h1")
    parser.add_argument(
            "--GN", action='store_true', help="use normalized x")
    parser.add_argument(
            "--sample_x", action='store_true', help="sample x")
    parser.add_argument(
            "--sampling_rate", type=int, default=2, help="sampling rate")
    parser.add_argument(
            "--normalizer", action='store_true', help="use normalizer")
    parser.add_argument(
            "--normalizer_type", type=str, default="GN", help="PGN, GN")
    parser.add_argument(
            "--num_layer", type=int, default=5, help="number of layers")
    parser.add_argument(
            "--num_channel_u", type=int, default=24, help="number of channels for u")
    parser.add_argument(
            "--num_channel_f", type=int, default=1, help="number of channels for f")
    parser.add_argument(
            '--num_iteration', type=list, nargs='+', default=[[1,0], [1,0], [1,0], [2,0], [2,0]], help='number of iterations in each layer')
    parser.add_argument(
            '--padding_mode', type=str, default='zeros', help='padding mode')
  

    args = parser.parse_args()
    args = vars(args)

    for i in range(len(args['num_iteration'])):
        for j in range(len(args['num_iteration'][i])):
            args['num_iteration'][i][j] = int(args['num_iteration'][i][j])
        

    if  args['sample_x']:
        if args['data'] in {'darcy', 'darcy20c6', 'darcy15c10', 'darcyF', 'darcy_contin'}:
            args['sampling_rate'] = 2
        elif args['data'] == 'a4f1':
            args['sampling_rate'] = 4
        elif args['data'] == 'helm':
            args['sampling_rate'] = 1
        elif args['data'] == 'pipe':
            args['sampling_rate'] = 1


    
  
        
    dataOpt = {}
    dataOpt['data'] = args['data']
    dataOpt['sampling_rate'] = args['sampling_rate']
    dataOpt['sample_x'] = args['sample_x']
    dataOpt['batch_size'] = args['batch_size']
    dataOpt['loss_type']=args['loss_type']
    dataOpt['loss_weight'] = [2,]
    dataOpt['normalizer_type'] = args['normalizer_type']
    dataOpt['GN'] = args['GN']
    dataOpt = getDataSize(dataOpt)

    modelOpt = {}
    modelOpt['num_layer'] = args['num_layer']
    modelOpt['num_channel_u'] = args['num_channel_u']
    modelOpt['num_channel_f'] = args['num_channel_f']
    modelOpt['num_classes'] = 1
    modelOpt['num_iteration'] = args['num_iteration']
    modelOpt['in_chans'] = 1
    modelOpt['normalizer'] = args['normalizer'] 
    modelOpt['output_dim'] = 1
    modelOpt['activation'] = 'gelu'    
 

    optimizerScheduler_args = {}
    optimizerScheduler_args['optimizer_type'] = 'adam'
    optimizerScheduler_args['lr'] = args['lr']
    optimizerScheduler_args['weight_decay'] = args['weight_decay']
    optimizerScheduler_args['epochs'] = args['epochs']
    optimizerScheduler_args['final_div_factor'] = args['final_div_factor']
    optimizerScheduler_args['div_factor'] = 2

    darcy.objective(dataOpt, modelOpt, optimizerScheduler_args, model_type=args['model_type'],
    validate=True, tqdm_disable=True, log_if=True, 
    model_save=True, )



    

    