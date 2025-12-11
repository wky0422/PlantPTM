#!/bin/python
# -*- coding:utf-8 -*- 

import torch
import torch.nn as nn
import torch.nn.functional as F

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv1d):
        nn.init.xavier_normal_(m.weight)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def detach(x):
    if x is None:
        return None
    return x.cpu().detach().numpy().squeeze()

class CNNEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, win_size, out_dim=64, kernel_size=9, strides=1, dropout=0.2):
        super(CNNEncoder, self).__init__()
        self.kernel_size = kernel_size
        self.strides = strides
        self.emd = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = torch.nn.Conv1d(in_channels=embed_dim, out_channels=hidden_dim, kernel_size=self.kernel_size, stride=self.strides)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=self.kernel_size, stride=self.strides)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.pool = nn.AvgPool1d(3, stride=strides)
        self.flat = nn.Flatten()
        self.lin1 = nn.Linear(hidden_dim * (win_size - (kernel_size - 1) * 2 - 2), out_dim)
        self.drop = nn.Dropout(p=dropout)
    
    def forward(self, x):
        x = self.emd(x.long())
        x = torch.permute(x, (0,2,1))
        x = F.relu(self.dropout1(self.bn1(self.conv1(x))))
        x = F.relu(self.dropout2(self.bn2(self.conv2(x))))
        x = self.pool(x)
        x = self.flat(x)
        x = self.drop(F.relu(self.lin1(x)))
        return x

class BiGRUEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, out_dim, n_layers, dropout, bidirectional=True):
        super(BiGRUEncoder, self).__init__()
        self.emd_layer = nn.Embedding(vocab_size, embedding_dim)
        self.n_layers = n_layers
        self.gru1 = nn.GRU(embedding_dim, hidden_dim*2, num_layers=n_layers, bidirectional=bidirectional, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        
        if bidirectional:
            self.gru2 = nn.GRU(hidden_dim * 4, out_dim, num_layers=n_layers, bidirectional=bidirectional, batch_first=True)
        else:
            self.gru2 = nn.GRU(hidden_dim * 2, out_dim, num_layers=n_layers, bidirectional=bidirectional, batch_first=True)
        
        self.dropout2 = nn.Dropout(dropout)
        self.attention = nn.Linear(out_dim * 2 if bidirectional else out_dim, 1)
    
    def forward(self, x):
        emd = self.emd_layer(x.long())
        self.raw_emd = emd
        output, _ = self.gru1(emd.float())
        output = self.dropout1(output)
        gruout2, _ = self.gru2(output)
        gruout2 = self.dropout2(gruout2)
        attention_weights = F.softmax(self.attention(gruout2), dim=1)
        attended_output = torch.sum(gruout2 * attention_weights, dim=1)
        attended_output = attended_output.unsqueeze(1)
        return attended_output

class FeatureEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, win_size, kernel_size=9, strides=1, dropout=0.2):
        super(FeatureEncoder, self).__init__()
        self.hidden_channels = hidden_dim
        
        if win_size < (kernel_size - 1) * 2:
            kernel_size = 7
        
        self.kernel_size = kernel_size
        self.strides = strides
        self.conv1 = torch.nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=self.kernel_size, stride=self.strides)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=self.kernel_size, stride=self.strides)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.pool = nn.AvgPool1d(3, stride=strides)
        self.flat = nn.Flatten()
        self.lin1 = nn.Linear(hidden_dim * (win_size - (kernel_size - 1) * 2 - 2), out_dim)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
        x = F.relu(self.dropout1(self.bn1(self.conv1(x.permute(0, 2, 1)))))
        x = F.relu(self.dropout2(self.bn2(self.conv2(x))))
        x = self.pool(x)
        x = self.flat(x)
        x = self.drop(F.relu(self.lin1(x)))
        return x

class PLMEncoder(nn.Module):
    def __init__(self, Bert_encoder, out_dim, PLM_dim=1024, kernel_size=3, dropout=0.2):
        super(PLMEncoder, self).__init__()
        self.bert = Bert_encoder
        self.conv1 = nn.Conv1d(PLM_dim, out_dim, kernel_size=kernel_size, stride=1, padding='same')
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.conv1.weight)
        
        if self.conv1.bias is not None:
            nn.init.constant_(self.conv1.bias, 0)
        
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
    
    def forward(self, input_ids=None, attention_mask=None, plm_feature=None):
        self.input_feature = plm_feature if plm_feature is not None else None
        
        if plm_feature is not None:
            pooled_output = plm_feature
        elif self.bert is not None:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state
        else:
            raise ValueError("Error: PLM feature file not provided, and no PLM model available.")
        
        imput = pooled_output.permute(0, 2, 1)
        conv1_output = F.relu(self.bn1(self.conv1(imput)))
        output = self.dropout(conv1_output)
        prot_out = torch.mean(output, axis=2, keepdim=True)
        prot_out = prot_out.permute(0, 2, 1)
        return prot_out

class MetaDecoder(nn.Module):
    def __init__(self, combined_dim, dropout=0.5):
        super(MetaDecoder, self).__init__()
        self.fc1 = nn.Linear(combined_dim, 32)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 5)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(5, 1)
        self.w_omega = nn.Parameter(torch.Tensor(combined_dim, combined_dim))
        self.u_omega = nn.Parameter(torch.Tensor(combined_dim, 1))
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)
    
    def attention_net(self, x):
        u = torch.tanh(torch.matmul(x, self.w_omega))
        att = torch.matmul(u, self.u_omega)
        att_score = F.softmax(att, dim=1)
        self.att_score = att_score
        scored_x = x * att_score
        context = torch.sum(scored_x, dim=1)
        return context
    
    def forward(self, fused_x):
        fusion_output = torch.cat(fused_x, axis=2)
        self.fusion_out = fusion_output
        attn_output = self.attention_net(fusion_output)
        self.attn_out = attn_output
        x = F.relu(self.dropout1(self.fc1(attn_output)))
        x = F.relu(self.dropout2(self.fc2(x)))
        self.final_out = x
        logit = self.fc(x)
        return logit

class PlantPTMNetSeq(nn.Module):
    def __init__(self, Bert_encoder, vocab_size, encoder_list=['cnn','gru','fea', 'plm'], win_size=51,
                 embedding_dim=32, fea_dim=41, hidden_dim=64, out_dim=32, PLM_dim=1024,
                 kernel_size=7, bidirectional=True, n_layers=1, dropout=0.2):
        super(PlantPTMNetSeq, self).__init__()
        dim_list = []
        self.encoder_list = encoder_list
        
        if 'cnn' in self.encoder_list:
            self.cnn_encoder = CNNEncoder(vocab_size, embed_dim=embedding_dim, hidden_dim=hidden_dim, win_size=win_size, out_dim=out_dim, kernel_size=kernel_size,dropout=dropout)
            dim_list.append(out_dim)
        
        if 'gru' in self.encoder_list:
            self.gru_encoder = BiGRUEncoder(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, out_dim=out_dim, n_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
            dim_list.append(out_dim*2)
        
        if 'fea' in self.encoder_list:
            self.fea_encoder = FeatureEncoder(input_dim=fea_dim, hidden_dim=hidden_dim, out_dim=out_dim, win_size=win_size, dropout=dropout)
            dim_list.append(out_dim)
        
        if 'plm' in self.encoder_list:
            self.plm_encoder = PLMEncoder(Bert_encoder=Bert_encoder, out_dim=out_dim, PLM_dim=PLM_dim, kernel_size=kernel_size, dropout=dropout)
            dim_list.append(out_dim)
        
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        combined_dim = sum(dim_list)
        self.decoder = MetaDecoder(combined_dim)
    
    def forward(self, input_ids, attention_mask, feature=None, plm_feature=None):
        fuse_x = []
        self.model_emd = {}
        
        if 'cnn' in self.encoder_list:
            cnn_out = self.cnn_encoder(input_ids).unsqueeze(1)
            fuse_x.append(cnn_out)
            self.model_emd['cnn_out'] = detach(cnn_out)
        
        if 'gru' in self.encoder_list:
            bi_gru_output = self.gru_encoder(input_ids)
            fuse_x.append(bi_gru_output)
            
            if hasattr(self.gru_encoder, 'raw_emd'):
                self.model_emd['raw'] = detach(self.gru_encoder.raw_emd)
            
            self.model_emd['gru_out'] = detach(bi_gru_output)
        
        if 'fea' in self.encoder_list and feature is not None:
            fea_out = self.fea_encoder(feature).unsqueeze(1)
            fuse_x.append(fea_out)
            self.model_emd['fea_in'] = detach(feature)
            self.model_emd['fea_out'] = detach(fea_out)
        
        if 'plm' in self.encoder_list:
            prot_out = self.plm_encoder(input_ids=input_ids, attention_mask=attention_mask, plm_feature=plm_feature)
            fuse_x.append(prot_out)
            
            if hasattr(self.plm_encoder, 'input_feature') and self.plm_encoder.input_feature is not None:
                self.model_emd['plm_input'] = detach(self.plm_encoder.input_feature)
            
            self.model_emd['plm_out'] = detach(prot_out)
        
        logit = self.decoder(fuse_x)
        self.model_emd['fusion_out'] = detach(self.decoder.fusion_out)
        self.model_emd['attn_out'] = detach(self.decoder.attn_out)
        self.model_emd['final_out'] = detach(self.decoder.final_out)
        return nn.Sigmoid()(logit)
    
    def _extract_embedding(self):
        print(f"Extract embedding from", list(self.model_emd.keys()))
        return self.model_emd
    