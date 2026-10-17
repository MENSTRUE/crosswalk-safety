from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

class TemporalConflictNet(nn.Module):
    def __init__(self,input_dim,conv_channels=64,hidden_size=96,dropout=.25):
        super().__init__()
        self.temporal=nn.Sequential(
            nn.Conv1d(input_dim,conv_channels,3,padding=1),
            nn.BatchNorm1d(conv_channels),nn.GELU(),nn.Dropout(dropout),
            nn.Conv1d(conv_channels,conv_channels,3,padding=1),
            nn.BatchNorm1d(conv_channels),nn.GELU(),
        )
        self.bilstm=nn.LSTM(conv_channels,hidden_size,batch_first=True,bidirectional=True)
        self.head=nn.Sequential(
            nn.LayerNorm(hidden_size*2),nn.Dropout(dropout),
            nn.Linear(hidden_size*2,64),nn.GELU(),nn.Dropout(dropout),nn.Linear(64,1)
        )

    def forward(self,x):
        x=x.transpose(1,2)
        x=self.temporal(x)
        x=x.transpose(1,2)
        x,_=self.bilstm(x)
        return self.head(x[:,-1,:]).squeeze(1)

class TemporalModelBundle:
    def __init__(self,checkpoint_path,mean_path,scale_path,device=None):
        self.device=torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint_path=Path(checkpoint_path); mean_path=Path(mean_path); scale_path=Path(scale_path)
        for p in [checkpoint_path,mean_path,scale_path]:
            if not p.exists(): raise FileNotFoundError(p)
        ckpt=torch.load(checkpoint_path,map_location=self.device,weights_only=False)
        self.feature_columns=list(ckpt["feature_columns"])
        self.sequence_length=int(ckpt["sequence_length"])
        self.threshold=float(ckpt["threshold"])
        self.sample_every_seconds=float(ckpt["sample_every_seconds"])
        self.predict_horizon_steps=int(ckpt["predict_horizon_steps"])
        self.mean=np.load(mean_path).astype(np.float32)
        self.scale=np.load(scale_path).astype(np.float32)
        self.scale=np.where(self.scale==0,1.0,self.scale).astype(np.float32)
        self.model=TemporalConflictNet(int(ckpt["input_dim"])).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def scale_sequence(self,sequence):
        sequence=np.asarray(sequence,dtype=np.float32)
        expected=(self.sequence_length,len(self.feature_columns))
        if sequence.shape!=expected:
            raise ValueError(f"Unexpected sequence shape {sequence.shape}, expected {expected}")
        return ((sequence-self.mean)/self.scale).astype(np.float32)

    @torch.no_grad()
    def predict_probability(self,sequence):
        x=torch.from_numpy(self.scale_sequence(sequence)[None,...]).to(self.device)
        return float(torch.sigmoid(self.model(x))[0].item())

    def classify(self,probability):
        return float(probability)>=self.threshold
