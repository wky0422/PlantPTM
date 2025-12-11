#!/bin/python
# -*- coding:utf-8 -*- 

import os
import os.path as osp
import time
import argparse
import torch
import random
import re
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from transformers import logging
logging.set_verbosity_error()

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from Bio import SeqIO
from tqdm import tqdm
from Metrics import *
from Dataset import *
from Model import *

def print_results(data, desc=['Epoch', 'Acc', 'th','Rec/Sn', 'Pre', 'F1', 'Spe', 'MCC', 'AUROC', 'AUPRC', 'TN', 'FP', 'FN', 'TP']):
    widths = []
    
    for i, header in enumerate(desc):
        if i < len(data):
            data_str = f"{data[i]:.4f}" if isinstance(data[i], float) else f"{data[i]}"
            width = max(len(header), len(data_str))
        else:
            width = len(header)
        widths.append(width + 2)
    
    header_line = ''
    
    for i, header in enumerate(desc):
        header_line += f"{header:<{widths[i]}}"
    
    print(header_line)
    data_line = ''
    
    for i, value in enumerate(data):
        if i < len(widths):
            value_str = f"{value:.4f}" if isinstance(value, float) else f"{value}"
            data_line += f"{value_str:<{widths[i]}}"
    
    print(data_line)

def random_run(SEED=2024):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed initialization: {SEED}!")

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train_one_epoch(loader, model, device, optimizer, criterion):
    model.train()
    train_step_loss = []
    train_total_acc = 0
    step = 1
    train_total_loss = 0
    
    for ind,(data) in enumerate(loader):
        input_ids, attention_mask, feature, plm_feature, label = data
        feature = feature.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        label = label.to(device)
        pred = model(input_ids=input_ids, attention_mask=attention_mask, feature=feature, plm_feature=plm_feature.to(device) if plm_feature is not None else None)
        logits = pred.squeeze()
        loss = criterion(logits, label.float())
        acc = (logits.round() == label).float().mean()
        model.zero_grad()
        loss.backward()
        optimizer.step()
        train_total_loss += loss.item()
        train_step_loss.append(loss.item())
        train_total_acc += acc
        step += 1
    
    avg_train_acc = train_total_acc / step
    avg_train_loss = train_total_loss / step
    return train_step_loss, avg_train_acc, avg_train_loss, step

def test_binary(model, loader, criterion, device):
    model.eval()
    criterion.to(device)
    test_probs = []
    test_targets = []
    valid_total_acc = 0
    valid_total_loss = 0
    valid_step = 1
    
    for ind,(data) in enumerate(loader):
        input_ids, attention_mask, feature, plm_feature, label = data
        feature = feature.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        label = label.to(device)
        pred = model(input_ids=input_ids, attention_mask=attention_mask, feature=feature, plm_feature=plm_feature.to(device) if plm_feature is not None else None)
        logits = pred.squeeze()
        loss = criterion(logits, label.float())
        acc = (logits.round() == label).float().mean()
        valid_total_loss += loss.item()
        valid_total_acc += acc.item()
        test_probs.extend(logits.cpu().detach().numpy())
        test_targets.extend(label.cpu().detach().numpy())
        valid_step += 1
    
    avg_valid_loss = valid_total_loss / valid_step
    avg_valid_acc = valid_total_acc / valid_step
    test_probs = np.array(test_probs)
    test_targets = np.array(test_targets)
    return test_probs, test_targets, avg_valid_loss, avg_valid_acc

def arg_parse(args_list=None):
    fasta_root_dir = "/PlantPTM/Data/fasta"
    pssm_root_dir = "/PlantPTM/Data/pssm"
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default='S-Acylation', type=str,
                        help="Project name for saving model checkpoints and best model.")
    parser.add_argument("--train", default='S-Acylation_train.fasta', type=str,
                        help="Training data file path.")
    parser.add_argument("--test", default='S-Acylation_test.fasta', type=str,
                        help="Testing data file path.")
    parser.add_argument("--pssm", default='S-Acylation', type=str,
                        help="Folder where PSSM files are stored, relative to root_dir_pssm or as an absolute path.")
    parser.add_argument("--model", default='Results/Models', type=str,
                        help="Folder for model storage and logits.")
    parser.add_argument("--result", default='Results', type=str,
                        help="Result folder for model training and evaluation.")
    parser.add_argument("--plm_dir", default='/PlantPTM/Data/bert_embedding/S-Acylation', type=str,
                        help="Folder with precomputed PLM embeddings.")
    parser.add_argument("--PLM", default='/PlantPTM/PLM/prot_bert', type=str,
                        help="The folder containing the PLM model to be used.")
    parser.add_argument("--PLM_type", default='bert', type=str, choices=['bert', 'esm2'],
                        help="Type of PLM to use: 'bert' or 'esm2'.")
    parser.add_argument("--PLM_dim", default=1024, type=int, choices=['1024', '1280'],
                        help="Hidden dimension of PLM. (1024 for Bert, 1280 for ESM2)")
    
    parser.add_argument('--epoch', type=int, default=200, metavar='[Int]',
                        help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=0.001, metavar='[Float]',
                        help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.001, metavar='[Float]',
                        help='Learning rate.')
    parser.add_argument('--batch', type=int, default=32, metavar='[Int]',
                        help='Batch size cutting threshold for Dataloader.')
    parser.add_argument('--cpu', '-cpu', type=int, default=4, metavar='[Int]',
                        help='CPU processors for data loading.')
    parser.add_argument('--gpu', '-gpu', type=int, default=3, metavar='[Int]',
                        help='GPU device ID to be used.')
    parser.add_argument('--emd_dim', '-ed', type=int, default=256, metavar='[Int]',
                        help='Word embedding dimension.')
    parser.add_argument('--hidden_dim', '-hd', type=int, default=256, metavar='[Int]',
                        help='Hidden dimension.')
    parser.add_argument('--out_dim', '-od', type=int, default=128, metavar='[Int]',
                        help='Out dimension for each track.')
    parser.add_argument('--kernel_size', '-ks', type=int, default=9, metavar='[Float]',
                        help='kernel_size.')
    parser.add_argument('--gru_nlayer', '-gn', type=int, default=2, metavar='[Int]',
                        help='Number of BiGRU layer.')
    parser.add_argument('--dropout', '-dp', type=float, default=0.5, metavar='[Float]',
                        help='Dropout rate.')
    parser.add_argument('--encoder', type=str, default='cnn,gru,fea,plm', metavar='[Str]',
                        help='Encoder list separated by comma chosen from cnn,gru,fea,plm.')
    parser.add_argument('--seed', type=int, default=2024, metavar='[Int]',
                        help='Random seed.')
    parser.add_argument('--patience', type=int, default=20, metavar='[Int]',
                        help='Early stopping patience.')
    parser.add_argument('--overfitting', type=int, default=1, metavar='[Int]',
                        help='Early stopping patience for overfitting detection. (valid_loss > train_loss)')
    
    if args_list is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args_list)
    
    args.train = args.train if os.path.isabs(args.train) else os.path.join(fasta_root_dir, args.train)
    args.test = args.test if os.path.isabs(args.test) else os.path.join(fasta_root_dir, args.test)
    args.pssm = args.pssm if os.path.isabs(args.pssm) else os.path.join(pssm_root_dir, args.pssm)
    return args

def build_dataloaders(args):
    print("Building dataloaders (one-time setup).")
    SEED = args.seed
    random_run(SEED)
    pretrained_model = args.PLM 
    encoder_list = args.encoder.split(',') 
    manual_fea = [PSSM, One_Hot]
    fea_dim = 41
    
    if 'plm' in encoder_list:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model, do_lower_case=False, use_fast=False, local_files_only=True)
    else:
        class DummyTokenizer:
            def __init__(self):
                AA = 'ARNDCQEGHILKMFPSTWYVX'
                self.vocab = {aa: i+1 for i, aa in enumerate(AA)}
                self.vocab['<pad>'] = 0
                self.vocab['<cls>'] = len(AA) + 1
                self.vocab['<sep>'] = len(AA) + 2
                self.vocab_size = len(self.vocab)
            
            def encode_plus(self, text, add_special_tokens=True, padding='max_length', return_token_type_ids=False, pad_to_max_length=True, truncation=True, max_length=None, return_tensors='pt'):
                tokens = text.split()
                token_ids = [self.vocab.get(token, self.vocab.get('X', 0)) for token in tokens]
                
                if add_special_tokens:
                    token_ids = [self.vocab['<cls>']] + token_ids + [self.vocab['<sep>']]
                if max_length and len(token_ids) < max_length:
                    token_ids += [self.vocab['<pad>']] * (max_length - len(token_ids))
                elif max_length and len(token_ids) > max_length:
                    token_ids = token_ids[:max_length]
                
                attention_mask = [1 if token_id != self.vocab['<pad>'] else 0 for token_id in token_ids]
                
                if return_tensors == 'pt':
                    import torch
                    return {'input_ids': torch.tensor([token_ids]), 'attention_mask': torch.tensor([attention_mask])}
                
                return {'input_ids': token_ids, 'attention_mask': attention_mask}
        
        tokenizer = DummyTokenizer()
    
    pssm = args.pssm
    
    if args.plm_dir:
        os.makedirs(args.plm_dir, exist_ok=True)
        print(f"PLM features will be loaded from: {args.plm_dir}")
    
    train_file = args.train
    seqlist = [record for record in SeqIO.parse(train_file, "fasta")]
    train_list, valid_list = random_split(seqlist, 0.2, seed=SEED)
    train_ds = PlantPTMDatasetSeq(train_list, tokenizer, pssm=pssm, feature=manual_fea, plm_dir=args.plm_dir)    
    valid_ds = PlantPTMDatasetSeq(valid_list, tokenizer, pssm=pssm, feature=manual_fea, plm_dir=args.plm_dir)
    test_file = args.test
    test_list = [record for record in SeqIO.parse(test_file, "fasta")]
    test_ds = PlantPTMDatasetSeq(test_list, tokenizer, pssm=pssm, feature=manual_fea, plm_dir=args.plm_dir)
    window_size = test_ds.win_size
    print("Dataloaders built successfully!")
    return tokenizer, window_size, train_ds, valid_ds, test_ds

def create_dataloaders_with_batch_size(train_ds, valid_ds, test_ds, batch_size, cpu, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=cpu, prefetch_factor=2, worker_init_fn=seed_worker, generator=g)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=True, num_workers=cpu, prefetch_factor=2, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=cpu, prefetch_factor=2, worker_init_fn=seed_worker, generator=g)
    return train_loader, valid_loader, test_loader

def run_training(args, tokenizer, window_size, train_ds, valid_ds, test_ds):
    save_model = True
    project = args.project
    SEED = args.seed
    random_run(SEED)
    embedding_dim = args.emd_dim
    hidden_dim = args.hidden_dim
    out_dim = args.out_dim
    lr = args.learning_rate
    wd = args.weight_decay
    num_epochs = args.epoch
    batch_size = args.batch
    cpu = args.cpu
    gpu = args.gpu
    model_dir = osp.join(args.result, f"{project}/Models")
    os.makedirs(model_dir, exist_ok=True)
    result_dir = osp.join(args.result, f"{project}")
    os.makedirs(result_dir, exist_ok=True)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    pretrained_model = args.PLM 
    encoder_list = args.encoder.split(',') 
    kernel_size = args.kernel_size
    n_layers = args.gru_nlayer
    dropout = args.dropout
    train_loader, valid_loader, test_loader = create_dataloaders_with_batch_size(train_ds, valid_ds, test_ds, batch_size, cpu, SEED)
    Bert_encoder = None
    
    if 'plm' in encoder_list:
        if args.plm_dir:
            print("Using precomputed PLM features, skipping PLM model load.")
            Bert_encoder = None
        else:
            print("Loading PLM model for real-time PLM feature extraction.")
            Bert_encoder = AutoModel.from_pretrained(pretrained_model, local_files_only=True, output_attentions=False).to(device)
    else:
        print("PLM encoder not included, skipping PLM model load.")
        Bert_encoder = None
    
    model = PlantPTMNetSeq(Bert_encoder=Bert_encoder, vocab_size=tokenizer.vocab_size, encoder_list=encoder_list, win_size=window_size, embedding_dim=embedding_dim, fea_dim=41, hidden_dim=hidden_dim, out_dim=out_dim, kernel_size=kernel_size, n_layers=n_layers, dropout=dropout).to(device)
    params = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    print("Model Trainable Parameter: "+ str(params/1024/1024) + 'Mb.' + "\n")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.BCELoss().to(device)
    result_list = []
    all_train_loss_list = []
    best_auc = 0
    best_epoch = 0
    patience = 0
    max_patience = args.patience
    overfitting_patience = 0
    max_overfitting_patience = args.overfitting
    last_loss = float('inf')
    desc = ['Project        ', 'Epoch', 'Acc', 'th', 'Rec/Sn', 'Pre', 'F1', 'Spe', 'MCC', 'AUROC', 'AUPRC', 'TN', 'FP', 'FN', 'TP']
    loss_file_path = osp.join(result_dir, 'loss.csv')
    file_exists = osp.exists(loss_file_path)
    
    with open(loss_file_path, 'a') as f:
        if not file_exists:
            f.write("Epoch\tTrain Loss\tValid Loss\tTest Loss\n")
    
    for epoch in range(num_epochs):
        start = time.perf_counter()
        train_step_loss, train_acc, train_loss, step = train_one_epoch(train_loader, model, device, optimizer, criterion)
        all_train_loss_list.extend(train_step_loss)
        end = time.perf_counter()
        print(f"\nEpoch {epoch+1} | {(end - start):.4f}s | Train | Loss: {train_loss: .6f}| Train acc: {train_acc:.4f}")
        start = time.perf_counter()
        valid_probs, valid_labels, valid_loss, valid_acc = test_binary(model, valid_loader, criterion, device)
        end = time.perf_counter()
        print(f"Epoch {epoch+1} | {(end - start):.4f}s | Valid | Valid loss: {valid_loss:.6f}| Valid acc: {valid_acc:.4f}")
        
        if valid_loss > train_loss:
            overfitting_patience += 1
            print(f"Current valid_loss: {valid_loss:.6f} > train_loss: {train_loss:.6f}, overfitting detected {overfitting_patience} / {max_overfitting_patience} epoch.")
        else:
            overfitting_patience = 0
        
        acc_, th_, rec_, pre_, f1_, spe_, mcc_, auc_, pred_class, auprc_, tn, fp, fn, tp = eval_metrics(valid_probs, valid_labels)
        result_info = [project+'_val', epoch, (tn+tp)/(tn+tp+fp+fn), th_, rec_, pre_, f1_, spe_, mcc_, auc_, auprc_, tn, fp, fn, tp]
        result_list.append(result_info)
        print_results(result_info, desc)
        
        if valid_loss > last_loss+0.1:
            patience += 1
        
        if patience > max_patience:
            break
        
        if overfitting_patience >= max_overfitting_patience:
            print(f"\nOverfitting has been detected {max_overfitting_patience} epoch in a row, and training has been stopped prematurely.\n")
            break
        
        if valid_loss <= last_loss:
            last_loss = valid_loss
            best_auc = auc_
            best_acc = acc_
            best_epoch = epoch
            start = time.perf_counter()
            test_probs, test_labels, test_loss, test_acc = test_binary(model, test_loader, criterion, device)
            end = time.perf_counter()
            print(f"Epoch {epoch+1} | {(end - start):.4f}s | Test | Test loss: {test_loss:.6f}| Test acc: {test_acc:.4f}")
            acc_, th_, rec_, pre_, f1_, spe_, mcc_, auc_, pred_class, auprc_, tn, fp, fn, tp = eval_metrics(test_probs, test_labels)
            result_info = [project+'_test', epoch, (tn+tp)/(tn+tp+fp+fn), th_, rec_, pre_, f1_, spe_, mcc_, auc_, auprc_, tn, fp, fn, tp]
            print_results(result_info, desc)
            best_test_probs = test_probs
            best_test_labels = test_labels
            best_result = result_info
        
        with open(loss_file_path, 'a') as f:
            f.write(f"{epoch}\t{train_loss:.6f}\t{valid_loss:.6f}\t{test_loss:.6f}\n")
    
    if save_model:
        save_path = osp.join(model_dir, f"{project}_best_model_epoch.pt")
        torch.save(model.state_dict(), save_path)
    
    print('\nBest result:\n')
    print_results(best_result, desc)
    loss_df = pd.DataFrame(all_train_loss_list)
    loss_df.columns = ['Loss']
    loss_df.to_csv(osp.join(result_dir, 'all_train_step_loss.txt'), sep='\t')
    logit_df = pd.DataFrame([best_test_labels, best_test_probs]).transpose()
    logit_df.columns = ['Label', 'Logit']
    logit_df.to_csv(osp.join(result_dir, f"logits_results.txt"), sep='\t', index=False)
    epoch_df = pd.DataFrame(result_list)
    epoch_df.columns = desc
    epoch_df.to_csv(osp.join(result_dir, f"epoch_result.csv"), sep='\t', index=False)
    epoch_df = pd.DataFrame([best_result])
    epoch_df.columns = desc
    epoch_df.to_csv(osp.join(result_dir, f"best_result.csv"), sep='\t', index=False)

if __name__=='__main__':
    args = arg_parse()
    print(args)
    tokenizer, window_size, train_ds, valid_ds, test_ds = build_dataloaders(args)
    run_training(args, tokenizer, window_size, train_ds, valid_ds, test_ds)
