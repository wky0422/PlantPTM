#!/bin/python
# -*- coding:utf-8 -*-

import os
import re
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO
from tqdm import tqdm
from collections import defaultdict

def extract_protein_id(header):
    return header.split('|')[0]

def generate_plm_embeddings_by_protein(input_fasta, output_plm, plm_model_path, number=64, gpu=0, max_length=512):
    os.makedirs(output_plm, exist_ok=True)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(plm_model_path, do_lower_case=False, use_fast=False)
    model = AutoModel.from_pretrained(plm_model_path).to(device)
    model.eval()
    protein_data = defaultdict(list)
    records = list(SeqIO.parse(input_fasta, "fasta"))
    
    for record in records:
        protein_id = extract_protein_id(record.id)
        parts = record.id.split('|')
        
        if len(parts) >= 5:
            protein_length = int(parts[4])
        else:
            protein_length = len(str(record.seq))
        
        protein_data[protein_id].append({
            'sequence': str(record.seq),
            'length': protein_length,
            'header': record.id
        })
    
    for protein_id, sequences_info in tqdm(protein_data.items(), desc="Processing proteins"):
        output_path = os.path.join(output_plm, f"{protein_id}.npy")
        
        if os.path.exists(output_path):
            continue
        
        reference_seq_info = sequences_info[0]
        protein_length = reference_seq_info['length']
        full_sequence = reconstruct_full_sequence(sequences_info)
        
        if len(full_sequence) != protein_length:
            print(f"Warning: Reconstructed sequence length {len(full_sequence)} != expected length {protein_length} for {protein_id}.")
        
        if len(full_sequence) > max_length:
            embedding = process_long_sequence(full_sequence, tokenizer, model, device, max_length, number)
        else:
            embedding = process_single_sequence(full_sequence, tokenizer, model, device)
        
        np.save(output_path, embedding)
        print(f"Saved PLM feature file for protein {protein_id}: shape {embedding.shape}.")

def reconstruct_full_sequence(sequences_info):
    longest_seq = max(sequences_info, key=lambda x: len(x['sequence']))['sequence']
    clean_seq = longest_seq.replace('X', '')
    if len(sequences_info) > 1:
        best_seq = ""
        
        for seq_info in sequences_info:
            seq = seq_info['sequence']
            if 'X' not in seq and len(seq) > len(best_seq):
                best_seq = seq
        
        if best_seq:
            return best_seq
    
    return clean_seq if clean_seq else longest_seq

def process_single_sequence(sequence, tokenizer, model, device):
    spaced_sequence = ' '.join(list(sequence))
    inputs = tokenizer(spaced_sequence, return_tensors="pt", padding=True, truncation=True, add_special_tokens=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    embedding = outputs.last_hidden_state[0, 1:-1, :].cpu().numpy()
    return embedding

def process_long_sequence(sequence, tokenizer, model, device, max_length, number):
    overlap = 50
    step_size = max_length - overlap
    embeddings = []
    positions = []
    
    for start in range(0, len(sequence), step_size):
        end = min(start + max_length, len(sequence))
        chunk = sequence[start:end]
        chunk_embedding = process_single_sequence(chunk, tokenizer, model, device)
        chunk_start_in_full = start
        chunk_end_in_full = start + len(chunk)
        embeddings.append(chunk_embedding)
        positions.append((chunk_start_in_full, chunk_end_in_full))
    
    full_embedding = merge_overlapping_embeddings(embeddings, positions, len(sequence), overlap)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fasta", required=True,
                        help="Input protein file in fasta format.")
    parser.add_argument("--output_plm", required=True,
                        help="Output folder for PLM embeddings.")
    parser.add_argument("--plm_model", default="/PlantPTM/prot_bert",
                        help="The folder containing the PLM model to be used.")
    parser.add_argument("--number", type=int, default=64,
                        help="The number of proteins processed in parallel.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID to be used.")
    parser.add_argument("--max_length", type=int, default=512,
                        help="The maximum sequence length that PLM can input.")
    args = parser.parse_args()
    generate_plm_embeddings_by_protein(
        args.input_fasta,
        args.output_plm,
        args.plm_model,
        args.number,
        args.gpu,
        args.max_length
    )
    