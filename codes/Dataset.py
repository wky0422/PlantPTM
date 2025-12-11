#!/bin/python
# -*- coding:utf-8 -*- 

import os
import torch
import random
import re
import sys
import numpy as np
from tqdm import tqdm

def extract_protein_id(header):
    return header.split('|')[0]

def random_split(datalist, ratio, seed=42):
    random.seed(seed)
    random.shuffle(datalist)
    num_samples = len(datalist)
    split_num = int(num_samples * float(ratio))
    large_list = datalist[split_num:]
    small_list = datalist[:split_num]
    return large_list, small_list

def PSSM(fastas, pssm_path, **kw):
    pssmMatrix = []
    proteinSeq = ''

    with open(pssm_path) as f:
        records = f.readlines()[3: -6]
        
        for array in records:
            array = array.strip().split()
            pssmMatrix.append(array[2:22])
            proteinSeq += array[1]
    
    encodings = []
    
    for sequence in fastas:
        clean_sequence = sequence.replace('X', '')
        pos = proteinSeq.find(clean_sequence)
        
        if pos == -1:
            print(f"Warning: Unable to match the corresponding peptide segment in the PSSM file of the given protein.\n")
        else:
            for p in range(len(sequence)):
                code = []
                
                if sequence[p] == 'X':
                    code += [0] * 20
                else:
                    code += pssmMatrix[pos]
                    pos += 1
                
                encodings.append(code)
    
    arr = np.array(encodings, dtype=np.int64)
    return arr

def One_Hot(fastas, **kw):
    AA = 'ARNDCQEGHILKMFPSTWYVX'
    encodings = []

    for sequence in fastas:
        for aa in sequence:
            if aa not in AA:
                aa = 'X'
            
            if aa == 'X':
                code = [0 for _ in range(len(AA))]
                encodings.append(code)
                continue
            
            code = []
            
            for aa1 in AA:
                tag = 1 if aa == aa1 else 0
                code.append(tag)
            
            encodings.append(code)
    
    arr = np.array(encodings)
    return arr

class PlantPTMDatasetSeq(object):
    def __init__(self, seqlist, tokenizer, pssm=None, feature=None, plm_dir=None, window_size=25):
        self.seq_list = []
        self.label_list = []
        self.feature_list = []
        self.plm_feature_list = []
        self.tokenizer = tokenizer
        self.pssm = pssm
        self.feature = feature
        self.plm_dir = plm_dir
        self.window_size = window_size
        self.win_size = window_size * 2 + 1
        self.plm_cache = {}
        
        for record in tqdm(seqlist):
            seq = str(record.seq)
            desc = record.id.split('|')
            
            if len(desc) < 5:
                print(f"Warning: The format of {record.id} is incorrect.\n")
                continue
            
            protein_id, label, dataset_type, position, protein_length = desc[0], int(desc[1]), desc[2], int(desc[3]), int(desc[4])
            pssm_path = os.path.join(self.pssm, protein_id + '.pssm')
            
            if not os.path.exists(pssm_path):
                print(f"Error: The PSSM file for protein {protein_id} does not exist.\n")
                continue
            
            fea = self._get_encoding(seq, pssm_path, feature)
            self.feature_list.append(fea)
            self.label_list.append(int(label))
            self.seq_list.append(seq)
            
            if plm_dir and hasattr(self, 'plm_dir') and self.plm_dir:
                plm_feature = self._get_plm_feature_slice(protein_id, position, protein_length)
                self.plm_feature_list.append(plm_feature)
            else:
                self.plm_feature_list.append(None)
    
    def _get_plm_feature_slice(self, protein_id, center_position, protein_length):
        if protein_id not in self.plm_cache:
            plm_path = os.path.join(self.plm_dir, f"{protein_id}.npy")
            
            if os.path.exists(plm_path):
                try:
                    full_plm_feature = np.load(plm_path)
                    self.plm_cache[protein_id] = full_plm_feature
                except Exception as e:
                    print(f"Error: The PLM feature file for protein {protein_id} is incorrect: {e}.\n")
                    self.plm_cache[protein_id] = None
            else:
                print(f"Warning: PLM feature file for protein {protein_id} cannot be found in {plm_path}.\n")
                self.plm_cache[protein_id] = None
        
        full_plm_feature = self.plm_cache[protein_id]
        
        if full_plm_feature is None:
            return np.zeros((self.win_size, 1280))
        
        center_idx = center_position - 1
        start_idx = max(0, center_idx - self.window_size)
        end_idx = min(len(full_plm_feature), center_idx + self.window_size + 1)
        sliced_feature = full_plm_feature[start_idx:end_idx]
        target_length = self.win_size
        
        if len(sliced_feature) < target_length:
            pad_before = max(0, self.window_size - center_idx)
            pad_after = max(0, center_idx + self.window_size + 1 - len(full_plm_feature))
            
            if pad_before > 0 or pad_after > 0:
                padded_feature = np.zeros((target_length, sliced_feature.shape[1]))
                start_pos = pad_before
                end_pos = start_pos + len(sliced_feature)
                padded_feature[start_pos:end_pos] = sliced_feature
                sliced_feature = padded_feature
        
        if len(sliced_feature) != target_length:
            if len(sliced_feature) > target_length:
                sliced_feature = sliced_feature[:target_length]
            else:
                temp = np.zeros((target_length, sliced_feature.shape[1]))
                temp[:len(sliced_feature)] = sliced_feature
                sliced_feature = temp
        
        return sliced_feature
    
    def __getitem__(self, index):
        seq = self.seq_list[index]
        seq = [token for token in re.sub(r"[UZOB*]", "X", seq.rstrip('*'))]
        max_len = len(seq)
        encoded = self.tokenizer.encode_plus(' '.join(seq), add_special_tokens = True, padding = 'max_length', return_token_type_ids = False, pad_to_max_length = True, truncation = True, max_length = max_len, return_tensors = 'pt')
        input_ids = encoded['input_ids'].flatten()
        attention_mask = encoded['attention_mask'].flatten()
        plm_feature = (torch.tensor(self.plm_feature_list[index], dtype=torch.float) if self.plm_feature_list[index] is not None else None)
        return (input_ids, attention_mask, torch.tensor(self.feature_list[index], dtype=torch.float), plm_feature, torch.tensor(self.label_list[index], dtype=torch.long))
    
    def __len__(self):
        return len(self.seq_list)
    
    def _get_encoding(self, seq, pssm_path, feature=[PSSM, One_Hot]):
        alphabet = 'ARNDCQEGHILKMFPSTWYVX'
        sample = ''.join([re.sub(r"[UZOB*]", "X", token) for token in seq])
        max_len = len(sample)
        all_fea = []
        
        for encoder in feature:
            fea = encoder([sample], pssm_path=pssm_path)
            assert fea.shape[0] == max_len
            all_fea.append(fea)
        
        return np.hstack(all_fea)
    