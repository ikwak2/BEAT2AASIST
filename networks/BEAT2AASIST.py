import torch
import torch.nn as nn
import torch.nn.functional as F

from .beats.BEATs import BEATsModel
from .AASIST import AASIST

### Feature Split method ###
class Feature_Split(nn.Module):
    def __init__(self, feature_split):
        super().__init__()
        self.split_strategy = feature_split

        if feature_split == 'Frequency':
            self.LL = nn.Linear(768, 128)
        elif feature_split == 'Channel':
            self.LL = nn.Linear(768, 256)
        else:
            self.LL = nn.Linear(768, 128)

    def forward(self, x):
        #x.shape = (B,top-k, 200, 768)
        x = self.LL(x) #(B, top-k, 200, d)
        if self.split_strategy == 'Frequency':
            x = x.transpose(2, 3) #(B, top-k, 128, 200)
            x_1 = x[:,:,:,:100]
            x_2 = x[:,:,:,100:]
            return x_1, x_2 #(B, top-k, 128, 100)
        elif self.split_strategy == 'Channel':
            x_1 = x[:,:,:, :128]
            x_2 = x[:,:,:, 128:]
            return x_1.transpose(2, 3), x_2.transpose(2, 3) #(B, top-k, 128, 200)
        
        return x.transpose(2, 3) #(B, top-k, 128, 200)
    

### Multi-layer Fusion method ###
class Multi_layer_Fusion(nn.Module):
    def __init__(self, multi_layer_fusion, top_k):
        super().__init__()
        if multi_layer_fusion not in [None,'None']: 
            if multi_layer_fusion == 'CNN-gate':
                self.fusion = CNN_Gate(top_k)
            elif multi_layer_fusion == 'SE-gate':
                self.fusion = SE_Gate(top_k)
            elif multi_layer_fusion == 'Concat':
                self.fusion = None
            else:
                self.fusion = None
        else:
            self.fusion = None

        self.top_k = max(int(top_k), 1)

    def forward(self, x, mel):
        x = torch.stack(x, dim=1)
        if  self.fusion != None:
            weights = self.fusion(x, mel)
            x = (x * weights).sum(dim=1, keepdim=True)  # [B, 1, L, C]

        return x

### CNN-gate for Multi-layer Fusion method ###
class CNN_Gate(nn.Module):
    """CNN-based gating network for mel-spectrogram input"""
    def __init__(self, top_k):
        super().__init__()
        
        # 2D CNN for mel-spectrogram [B, C, F, T] where C is usually 1
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # Global pooling -> [B, 128, 1, 1]
        )
        
        # FC to produce per-layer weights
        self.fc = nn.Linear(128, top_k)
    
    def forward(self, _, x):
        """
        x: [B, C, F, T] or [B, C, T] mel-spectrogram input
        returns: [B, n_layers, 1, 1] weights
        """
        
        # Handle different input shapes
        if x.dim() == 3:  # [B, C, T]
            x = x.unsqueeze(1)  # [B, 1, C, T]
        elif x.dim() == 2:  # [B, T]
            x = x.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, T]
        
        # Pass through CNN
        features = self.conv_layers(x)  # [B, 128, 1, 1]
        features = features.squeeze(-1).squeeze(-1)  # [B, 128]
        
        # Compute layer weights
        weights = self.fc(features)  # [B, top-k]
        weights = F.softmax(weights, dim=1)  # [B, top-k]
        
        return weights.unsqueeze(-1).unsqueeze(-1)  # [B, top-k, 1, 1]

### SE-gate for Multi-layer Fusion method ###
class SE_Gate(nn.Module):
    """Squeeze-and-Excitation style gating for layer aggregation"""
    def __init__(self, top_k):
        super().__init__()
        
        # Global pooling followed by FC layers
        self.squeeze = nn.AdaptiveAvgPool2d(1)  # [B, top_k, L, C] -> [B, top_k, 1, 1]
        
        # Bottleneck MLP
        self.excitation = nn.Sequential(
            nn.Linear(top_k * 768, (top_k * 768) // 4),
            nn.ReLU(inplace=True),
            nn.Linear((top_k * 768) // 4, top_k),
        )
    
    def forward(self, x, _):
        """
        x: [B, n_layers, L, C]
        returns: [B, n_layers, 1, 1] weights
        """
        B, top_k, L, C = x.shape
        
        # Squeeze: global average pooling per layer
        squeezed = self.squeeze(x.view(B, top_k, L, C))  # [B, top-k, 1, 1]
        squeezed = squeezed.view(B, -1)  # [B, top-k * 1 * 1] = [B, top-k]
        
        # Actually we want to pool over spatial dimensions only
        # Let's reshape properly
        x_reshaped = x.permute(0, 1, 3, 2)  # [B, top-k, C, L]
        pooled = F.adaptive_avg_pool1d(x_reshaped.reshape(B * top_k, C, L), 1)  # [B*top-k, C, 1]
        pooled = pooled.view(B, top_k * C)  # [B, top-k * C]
        
        # Excitation: learn layer importance
        weights = self.excitation(pooled)  # [B, top-k]
        weights = F.softmax(weights, dim=1)  # [B, top-k]
        
        return weights.unsqueeze(-1).unsqueeze(-1)  # [B, top-k, 1, 1]



### BEAT2AASIST Model ###
class BEAT2AASIST(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        ### BEATs ###
        self.BEATs = BEATsModel(args)

        ### Multi-layer Fusion ###
        if args.multi_layer_fusion not in [None,'None']:
            assert 1<args.top_k and args.top_k<=12, 'Select an appropriate value of top_k (1<k<13)!'
        if args.multi_layer_fusion in [None,'None']:
            assert args.top_k==1, 'Set top_k = 1 !'
        self.multi_layer_fusion = Multi_layer_Fusion(args.multi_layer_fusion, args.top_k)

        ### Feature Split ###
        self.feature_split = Feature_Split(args.feature_split)

        ### AASIST ###
        if args.feature_split in [None,'None']:
            self.AASIST = AASIST(args)
            self.out_layer = nn.Linear(160, 2)
        else:
            self.AASIST_1 = AASIST(args)
            self.AASIST_2 = AASIST(args)
            self.out_layer = nn.Linear(320, 2)   

    def forward(self, mel):
        # x.shape=(B, 402, 128)
        _, hidden_states = self.BEATs(mel, return_hidden=True, last_k=self.args.top_k)
        
        ### Multi-layer Fusion ###
        x_fusion = self.multi_layer_fusion(hidden_states, mel)

        ### Feature Split ###
        x_split = self.feature_split(x_fusion)
        if len(x_split)!=2:
            last_hidden = self.AASIST(x_split) #(B, 160)
        else:    
            x_1 = self.AASIST_1(x_split[0])
            x_2 = self.AASIST_2(x_split[1])
            last_hidden = torch.cat( [x_1, x_2], dim=1) #(B, 320)

        output = self.out_layer(last_hidden) # (B, 2)
        
        return output