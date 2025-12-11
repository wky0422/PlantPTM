#!/bin/python
# -*- coding:utf-8 -*-

import os
import os.path as osp
import re
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import shutil
import subprocess
import random
from transformers import logging
logging.set_verbosity_error()

from Bio import SeqIO
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from Model import PlantPTMNetSeq
from Dataset import random_split, PSSM, One_Hot

matplotlib.use('agg')

def set_seed(seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except:
        torch.set_deterministic(True)

PTM_MAP = {
    'Ngly': ('N-Glycosylation', 'Ngly', 'N'),
    'Sacy': ('S-Acylation', 'Sacy', 'C'),
    'Khib': ('2-Hydroxyisobutyrylation', 'Khib', 'K'),
    'Kcro':  ('Crotonylation', 'Kcro', 'K'),
    'Ksucc':  ('Succinylation', 'Ksucc', 'K'),
    'Kmal':  ('Malonylation', 'Kmal', 'K'),
    'Kac':  ('Acetylation', 'Kac', 'K'),
    'Kub':  ('Ubiquitination', 'Kub', 'K'),
    'pho':  ('Phosphorylation', 'pho', 'STY')
}

def load_thresholds(threshold_file='Threshold.txt'):
    thresholds = {}
    
    if not os.path.exists(threshold_file):
        print(f"Warning: Threshold file {threshold_file} not found. Using default thresholds.")
        return None
    
    try:
        with open(threshold_file, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        
        i = 0
        while i < len(lines):
            if not lines[i]:
                i += 1
                continue
            
            ptm_type = lines[i].strip()
            i += 1
            
            if i + 3 < len(lines):
                try:
                    threshold_values = [
                        float(lines[i]),
                        float(lines[i + 1]),
                        float(lines[i + 2]),
                        float(lines[i + 3])
                    ]
                    thresholds[ptm_type] = threshold_values
                    i += 4
                except ValueError as e:
                    print(f"Warning: Invalid threshold values for {ptm_type}: {e}")
                    i += 4
            else:
                print(f"Warning: Incomplete threshold data for {ptm_type}")
                break
        
        print(f"Successfully loaded thresholds for {len(thresholds)} PTM types")
        return thresholds
    
    except Exception as e:
        print(f"Error loading threshold file: {e}")
        return None

def get_confidence_level(score, thresholds):
    if thresholds is None or len(thresholds) != 4:
        if score >= 0.108:
            return 'Extremely high'
        elif score >= 0.0675:
            return 'High'
        elif score >= 0.0405:
            return 'Medium'
        elif score >= 0.027:
            return 'Low'
        else:
            return 'Non-PTM'
    
    extremely_high_threshold, high_threshold, medium_threshold, low_threshold = thresholds
    
    if score >= extremely_high_threshold:
        return 'Extremely high'
    elif score >= high_threshold:
        return 'High'
    elif score >= medium_threshold:
        return 'Medium'
    elif score >= low_threshold:
        return 'Low'
    else:
        return 'Non-PTM'

def PSSM_file(*args, **kwargs):
    raise NotImplementedError("PSSM file is a placeholder.")

def read_existing_pssm(pssm_file_path):
    try:
        pssm_matrix = []
        
        with open(pssm_file_path, 'r') as f:
            lines = f.readlines()
            content_lines = lines[3:-6]
            
            for line in content_lines:
                cols = line.strip().split()
                
                if len(cols) >= 22:
                    pssm_matrix.append([float(x) for x in cols[2:22]])
        
        pssm_matrix = np.array(pssm_matrix, dtype=np.float32)
        print(f"Successfully read PSSM file: {pssm_file_path}, matrix size: {pssm_matrix.shape}.")
        return pssm_matrix
    
    except Exception as e:
        print(f"Failed to read PSSM file {pssm_file_path}: {e}.")
        return None

def find_pssm_file(seq_id, pssm_folder):
    if not osp.exists(pssm_folder):
        return None
    
    possible_names = [
        f"{seq_id}.pssm",
        f"{seq_id}.txt",
        f"{seq_id}.out",
        f"{seq_id}_pssm.txt",
        f"{seq_id}.pssm.txt"
    ]
    
    for filename in possible_names:
        filepath = osp.join(pssm_folder, filename)
        
        if osp.exists(filepath):
            return filepath
    
    for file in os.listdir(pssm_folder):
        if seq_id in file and (file.endswith('.pssm') or file.endswith('.txt') or file.endswith('.out')):
            return osp.join(pssm_folder, file)
    
    return None

def generate_pssm_matrix(seq_id, seq, uniref_db_path, psiblast_path, num_iterations=3, evalue=0.001):
    tmp_dir = tempfile.mkdtemp()
    fasta_file = osp.join(tmp_dir, "temp.fasta")
    pssm_file = osp.join(tmp_dir, "temp.pssm")
    
    with open(fasta_file, "w") as f:
        f.write(f">{seq_id}\n{seq}\n")
    
    print(f"Generating PSSM feature: {seq_id}.")
    cmd = [
        psiblast_path,
        "-query", fasta_file,
        "-db", uniref_db_path,
        "-num_iterations", str(num_iterations),
        "-evalue", str(evalue),
        "-out_ascii_pssm", pssm_file,
        "-num_threads", "32"
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pssm_matrix = []
    
    with open(pssm_file) as f:
        lines = f.readlines()[3:-6]
        
        for line in lines:
            cols = line.strip().split()
            pssm_matrix.append([float(x) for x in cols[2:22]])
    
    pssm_matrix = np.array(pssm_matrix, dtype=np.float32)
    shutil.rmtree(tmp_dir)
    return pssm_matrix

def get_or_generate_pssm(seq_id, seq, pssm_folder=None, uniref_db_path=None, psiblast_path=None):
    pssm_matrix = None
    
    if pssm_folder:
        pssm_file_path = find_pssm_file(seq_id, pssm_folder)
        
        if pssm_file_path:
            pssm_matrix = read_existing_pssm(pssm_file_path)
            
            if pssm_matrix is not None:
                print(f"Use pre stored PSSM files: {pssm_file_path}.")
                return pssm_matrix
            else:
                print(f"Failed to read pre stored PSSM file, will generate temporarily: {seq_id}.")
        
        else:
            print(f"Failed to find pre stored PSSM file, will generate temporarily: {seq_id}.")
    
    if uniref_db_path and psiblast_path:
        pssm_matrix = generate_pssm_matrix(seq_id, seq, uniref_db_path, psiblast_path)
        print(f"Successfully generated temporary PSSM: {seq_id}.")
    else:
        raise ValueError(f"Unable to obtain the PSSM matrix for protein {seq_id}. Neither the pre stored PSSM folder nor the parameters required to generate PSSM were provided.")
    
    return pssm_matrix

def _get_encoding(seq, pssm_path=None, pssm_matrix=None, feature=[PSSM, One_Hot]):
    sample = ''.join([re.sub(r"[UZOB*]", "X", token) for token in seq])
    max_len = len(sample)
    all_fea = []
    
    for encoder in feature:
        if encoder.__name__ == "PSSM_file" and pssm_matrix is not None:
            fea = pssm_matrix
        else:
            fea = encoder([sample], pssm_path=pssm_path)
        assert fea.shape[0] == max_len
        all_fea.append(fea)
    
    return np.hstack(all_fea)

def generate_protein_bert_features(seq_id, seq, tokenizer, bert_model, device, max_length=1022):
    if len(seq) > max_length:
        overlap = 50
        step_size = max_length - overlap
        embeddings = []
        positions = []
        
        for start in range(0, len(seq), step_size):
            end = min(start + max_length, len(seq))
            chunk = seq[start:end]
            spaced_sequence = ' '.join(list(chunk))
            inputs = tokenizer(spaced_sequence, return_tensors="pt", padding=True, truncation=True, add_special_tokens=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = bert_model(**inputs)
            
            chunk_embedding = outputs.last_hidden_state[0, 1:-1, :].cpu().numpy()
            embeddings.append(chunk_embedding)
            positions.append((start, start + len(chunk)))
        
        full_embedding = merge_overlapping_embeddings(embeddings, positions, len(seq), overlap)
    
    else:
        spaced_sequence = ' '.join(list(seq))
        inputs = tokenizer(spaced_sequence, return_tensors="pt", padding=True, truncation=True, add_special_tokens=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = bert_model(**inputs)
        
        full_embedding = outputs.last_hidden_state[0, 1:-1, :].cpu().numpy()
    
    return full_embedding

def merge_overlapping_embeddings(embeddings, positions, total_length, overlap):
    final_embedding = np.zeros((total_length, embeddings[0].shape[1]))
    weight_sum = np.zeros(total_length)
    
    for i, (embedding, (start, end)) in enumerate(zip(embeddings, positions)):
        actual_length = min(len(embedding), end - start)
        
        for j in range(actual_length):
            pos_in_full = start + j
            
            if pos_in_full < total_length:
                if i == 0:
                    weight = 1.0 if j < len(embedding) - overlap//2 else 0.5
                elif i == len(embeddings) - 1:
                    weight = 1.0 if j > overlap//2 else 0.5
                else:
                    if j < overlap//2 or j >= len(embedding) - overlap//2:
                        weight = 0.5
                    else:
                        weight = 1.0
                
                final_embedding[pos_in_full] += embedding[j] * weight
                weight_sum[pos_in_full] += weight
    
    for i in range(total_length):
        if weight_sum[i] > 0:
            final_embedding[i] /= weight_sum[i]
    
    return final_embedding

def extract_plm_feature_slice(full_plm_feature, center_position, window_size=25):
    win_size = window_size * 2 + 1
    
    if full_plm_feature is None:
        return np.zeros((win_size, 1024))
    
    center_idx = center_position - 1
    start_idx = max(0, center_idx - window_size)
    end_idx = min(len(full_plm_feature), center_idx + window_size + 1)
    sliced_feature = full_plm_feature[start_idx:end_idx]
    target_length = win_size
    
    if len(sliced_feature) < target_length:
        pad_before = max(0, window_size - center_idx)
        pad_after = max(0, center_idx + window_size + 1 - len(full_plm_feature))
        
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

def get_peptide(pos, window_size, seq):
    pos = pos-1
    half_window = window_size // 2
    start = pos - half_window
    end = pos + half_window + 1
    left_padding = ''
    
    if start < 0:
        left_padding = 'X' * abs(start)
        start = 0
    
    right_padding = ''
    
    if end > len(seq):
        right_padding = 'X' * (end - len(seq))
        end = len(seq)
    
    peptide_ = seq[start:end]
    peptide = left_padding + peptide_ + right_padding
    peptide = peptide[:window_size]
    
    if len(peptide) < window_size:
        peptide += 'X' * (window_size - len(peptide))
    
    return peptide

def get_target_sites(seq, ptm_type, window_size=51):
    if ptm_type not in PTM_MAP:
        raise ValueError(f"Unsupported PTM type: {ptm_type}.")
    
    full_name, short_name, target_residues = PTM_MAP[ptm_type]
    peplist = []
    
    for residue in target_residues:
        pattern = re.escape(residue)
        
        for match in re.finditer(pattern, seq):
            pos = match.start() + 1
            pep = get_peptide(pos, window_size, seq)
            
            if pep is not None:
                peplist.append([f'seq|Pred|{pos}|{len(seq)}', pep, pos, residue])
    
    peplist.sort(key=lambda x: x[2])
    print(f"Find {len(peplist)} {target_residues} sites in the protein sequence for {full_name} prediction.")
    return peplist

def get_all_k(seq, window_size=51):
    return get_target_sites(seq, '2hib', window_size)

def load_model_parameters(param_file):
    params = {}
    
    with open(param_file, 'r') as f:
        for line in f:
            key, value = line.strip().split('=')
            try:
                params[key] = float(value) if '.' in value else int(value)
            except ValueError:
                params[key] = value
    
    return params

def predict_engine(seq_path, model_root, ptm_type, plm_model_path, window_size=51, batch_size=128, gpu=0, threshold_file='Threshold.txt', pssm_folder=None, uniref_db=None, psiblast_path=None):
    set_seed(2024)
    if torch.cuda.is_available() and gpu >= 0:
        device = torch.device(f"cuda:{gpu}")
        print(f"Using GPU: {gpu}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    if device.type == 'cpu':
        torch.set_num_threads(4)
    
    if ptm_type not in PTM_MAP:
        raise ValueError(f"Invalid PTM type: {ptm_type}.")
    
    all_thresholds = load_thresholds(threshold_file)
    ptm_thresholds = None
    if all_thresholds and ptm_type in all_thresholds:
        ptm_thresholds = all_thresholds[ptm_type]
        print(f"Using thresholds for {ptm_type}: {ptm_thresholds}")
    else:
        print(f"Warning: No specific thresholds found for {ptm_type}, using default thresholds")
    
    full_name, short_name, target_residues = PTM_MAP[ptm_type]
    model_dir = osp.join(model_root, full_name)
    
    param_file = osp.join(model_dir, 'parameter.txt')
    
    if not osp.exists(param_file):
        raise FileNotFoundError(f"Parameter file not found: {param_file}")
    
    model_params = load_model_parameters(param_file)
    encoder_list = model_params.get('encoder', 'cnn,gru,fea').split(',')
    tokenizer = AutoTokenizer.from_pretrained(plm_model_path, do_lower_case=False, use_fast=False)
    Bert_encoder = None
    
    if 'plm' in encoder_list:
        print(f"Loading PLM model for feature extraction: {plm_model_path}")
        Bert_encoder = AutoModel.from_pretrained(plm_model_path, local_files_only=True)
        Bert_encoder.to(device)
        Bert_encoder.eval()
        
        for module in Bert_encoder.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0
    
    model_path = osp.join(model_dir, f'{full_name}_best_model_epoch.pt')
    model = PlantPTMNetSeq(
        Bert_encoder=Bert_encoder,
        vocab_size=tokenizer.vocab_size,
        encoder_list=encoder_list,
        win_size=window_size,
        embedding_dim=model_params.get('emd_dim', 32),
        fea_dim=41,
        hidden_dim=model_params.get('hidden_dim', 64),
        out_dim=model_params.get('out_dim', 32),
        kernel_size=model_params.get('kernel_size', 7),
        n_layers=model_params.get('gru_nlayer', 1),
        dropout=model_params.get('dropout', 0.5)
    ).to(device)
    
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0
    
    torch.set_grad_enabled(False)
    seq_records = list(SeqIO.parse(seq_path, "fasta"))
    all_predictions = []
    
    for record in seq_records:
        seq_id = record.id
        seq = str(record.seq)
        pssm_matrix = get_or_generate_pssm(seq_id, seq, pssm_folder=pssm_folder, uniref_db_path=uniref_db, psiblast_path=psiblast_path)
        full_bert_features = None
        
        if 'plm' in encoder_list and Bert_encoder is not None:
            print(f"Generating PLM feature matrix for protein {seq_id}")
            full_bert_features = generate_protein_bert_features(seq_id, seq, tokenizer, Bert_encoder, device)
        
        peplist = get_target_sites(seq, ptm_type, window_size=window_size)
        
        for desc, peptide, pos, residue in peplist:
            half = window_size // 2
            start = pos - 1 - half
            end = pos - 1 + half + 1
            pssm_window = []
            
            for i in range(start, end):
                if i < 0 or i >= pssm_matrix.shape[0]:
                    pssm_window.append([0.0]*20)
                else:
                    pssm_window.append(pssm_matrix[i])
            
            pssm_window = np.array(pssm_window, dtype=np.float32)
            fea = _get_encoding(peptide, pssm_matrix=pssm_window, feature=[PSSM_file, One_Hot])
            feature_tensor = torch.tensor(fea, dtype=torch.float32).unsqueeze(0).to(device)
            plm_feature_tensor = None
            
            if 'plm' in encoder_list and full_bert_features is not None:
                plm_window_feature = extract_plm_feature_slice(full_bert_features, pos, window_size//2)
                plm_feature_tensor = torch.tensor(plm_window_feature, dtype=torch.float32).unsqueeze(0).to(device)
            
            clean_peptide = [token for token in re.sub(r"[UZOB*]", "X", peptide.rstrip('*'))]
            encoded = tokenizer.encode_plus(' '.join(clean_peptide), add_special_tokens=True, 
                                           padding='max_length', max_length=window_size, 
                                           truncation=True, return_tensors='pt')
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            with torch.no_grad():
                pred = model(input_ids=input_ids, attention_mask=attention_mask, 
                           feature=feature_tensor, plm_feature=plm_feature_tensor)
                score = pred.squeeze().item()
            
            confidence = get_confidence_level(score, ptm_thresholds)
            all_predictions.append({
                'Protein': seq_id,
                'Position': pos,
                'Residue': residue,
                'Peptide': peptide,
                'PTM type': ptm_type,
                'Score': score,
                'Confidence': confidence
            })
    
    return pd.DataFrame(all_predictions)

def draw_ptm(df, savename='ptm_barplot.png'):
    plt.figure(figsize=(10, 6))
    palette = {
        'Extremely high': "#ffa2a6",
        'High': "#ffc4d3",
        'Medium': "#cbe6ff",
        'Low': "#d1f7eb",
        'Non-PTM': "#faecd3"
    }
    hue_order = ['Extremely high', 'High', 'Medium', 'Low', 'Non-PTM']
    ax = sns.barplot(x='Position', y='Score', data=df, hue='Confidence', palette=palette, hue_order=hue_order, dodge=False, edgecolor='black', linewidth=1.0)
    plt.axhline(y=0.027, color='#AD0404', linestyle='--', label='Threshold')
    ptm_type_key = df['PTM type'].iloc[0]
    _, short_name, _ = PTM_MAP[ptm_type_key]
    plt.title(f'{short_name} bar plot', fontsize=16)
    plt.xlabel('Position', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, title='Confidence', loc='upper right')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(savename, dpi=300)
    plt.close()

def draw_scatter_ptm(df, savename='ptm_scatter.png'):
    plt.figure(figsize=(10, 6))
    palette = {
        'Extremely high': "#ffa2a6",
        'High': "#ffc4d3",
        'Medium': "#cbe6ff",
        'Low': "#d1f7eb",
        'Non-PTM': "#faecd3"
    }
    hue_order = ['Extremely high', 'High', 'Medium', 'Low', 'Non-PTM']
    ax = sns.scatterplot(x='Position', y='Score', data=df, hue='Confidence', palette=palette, hue_order=hue_order, s=100, edgecolor='black', linewidth=1.0)
    plt.axhline(y=0.027, color="#AD0404", linestyle='--', label='Threshold')
    ptm_type_key = df['PTM type'].iloc[0]
    _, short_name, _ = PTM_MAP[ptm_type_key]
    plt.title(f'{short_name} scatter plot', fontsize=16)
    plt.xlabel('Position', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, title='Confidence', loc='upper right')
    plt.tight_layout()
    plt.savefig(savename, dpi=300)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Use PlantPTM for prediction.')
    parser.add_argument('--fasta', required=True,
                        help='Input protein file to be predicted in fasta format.')
    parser.add_argument('--model', default='models',
                        help='The folder where the trained model is stored.')
    parser.add_argument('--ptm', required=True, choices=['Ngly', 'Sacy', 'Khib', 'Kcro', 'Ksucc', 'Kmal', 'Kac', 'Kub', 'pho'],
                        help='The type of PTM that you want to predict.')
    parser.add_argument('--plm', default='/PlantPTM/prot_bert',
                        help='The folder containing the PLM model to be used.')
    parser.add_argument('--pssm_dir', default=None,
                        help='The folder where pre calculated PSSM files are stored. (Optional, priority use if provided)')
    parser.add_argument('--uniref', default='/PlantPTM/uniref50/uniref50',
                        help='Local uniref50 database or other databases path. (Required when PSSM folder is not provided)')
    parser.add_argument('--psiblast', default='/PlantPTM/ncbi-blast-2.16.0+/bin/psiblast',
                        help='The executable psiblast file path. (Required when PSSM folder is not provided)')
    parser.add_argument('--output', default='predictions.csv',
                        help='Output CSV file path for prediction results.')
    parser.add_argument('--window_size', type=int, default=51,
                        help='Window size for peptide extraction.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (-1 for CPU, recommended for consistency).')
    parser.add_argument('--threshold', default='Threshold.txt',
                        help='Path to threshold configuration file.')
    args = parser.parse_args()
    
    set_seed(2024)
    print("PROGRESS:Initializing Model...", flush=True)
    full_name, short_name, target_residues = PTM_MAP[args.ptm]
    print(f"PTM type: {full_name} ({short_name})")
    print(f"Target residues: {', '.join(target_residues)}")
    predictions_df = predict_engine(
        seq_path=args.fasta,
        model_root=args.model,
        ptm_type=args.ptm,
        plm_model_path=args.plm,
        window_size=args.window_size,
        gpu=args.gpu,
        threshold_file=args.threshold,
        pssm_folder=args.pssm_dir,
        uniref_db=args.uniref,
        psiblast_path=args.psiblast
    )
    predictions_df.to_csv(args.output, index=False)
    print(f"PROGRESS:Predictions saved to {args.output}.", flush=True)
    
    if not predictions_df.empty:
        print(f"\nPrediction statistics:")
        print(f"Total predicted sites: {len(predictions_df)}")
        print("Residue counts:")
        print(predictions_df['Residue'].value_counts())
        print("Confidence distribution:")
        print(predictions_df['Confidence'].value_counts())
    else:
        print("No predictions generated.")
