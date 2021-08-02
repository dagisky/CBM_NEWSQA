import torch
import torch.nn as nn
import torch.nn.functional as F



class RelationalLayerBase(nn.Module):
    """
    The relational Base Layer 
    An RN is a neural network module with a structure primed for relational reasoning. The design
    philosophy behind RNs is to constrain the functional form of a neural network so that it captures the
    core common properties of relational reasoning.
    
    Child of nn.Module Class
    """
    def __init__(self, in_size, out_size, device, hyp):
        super().__init__()
        """
        Relational Base Layer Initializer
        Args:
            in_size (Integer): network input size
            out_size (Integer): network output size
            device (torch.device): Pytorch Device Object
            hyp (Dictionary): Hyperparmeters {g_layers:<Int>, f_fc1:<Int>, f_fc2:<Int>, dropout:<float>}
        Returns: None
        """
        self.device = device
        self.f_fc1 = nn.Linear(hyp["g_layers"][-1], hyp["f_fc1"]).to(device)
        self.f_fc2 = nn.Linear(hyp["f_fc1"], hyp["f_fc2"]).to(device)
        self.f_fc3 = nn.Linear(hyp["f_fc2"], out_size).to(device)
    
        self.dropout = nn.Dropout(p=hyp["dropout"])
        
        self.on_gpu = True
        self.hyp = hyp
        self.in_size = in_size
        self.out_size = out_size

    def cuda(self):
        """Load Model on CUDA"""
        self.on_gpu = True
        super().cuda()
    

class RelationalLayer(RelationalLayerBase):
    """
    Child Class of RelationalLayerBase
    Creates the g_layers for the given a set of “objects” O = {o1, o2, ..., on}

    Parameters
    ----------
        g_layers : list[nn.Linear]
            List of nn.Linear modules 
        edge_feature: nn.Linear Module
            Transform the output feature

    """
    def __init__(self, in_size, out_size, device, hyp):
        super().__init__(in_size, out_size, device, hyp)
        """
        Relational Layer Initializer
        Args:
            in_size (Integer): network input size
            out_size (Integer): network output size
            device (torch.device): Pytorch Device Object
            hyp (Dictionary): Hyperparmeters {g_layers:<Int>, f_fc1:<Int>, f_fc2:<Int>, dropout:<float>}
        Returns: None
        """
        self.in_size = in_size
        self.edge_feature = nn.Linear((in_size//2)*4, in_size).to(device)

        #create all g layers
        self.g_layers = []
        self.g_layers_size = hyp["g_layers"]

        for idx, g_layer_size in enumerate(hyp["g_layers"]):
            in_s = in_size if idx==0 else hyp["g_layers"][idx-1]
            out_s = g_layer_size
            l = nn.Linear(in_s, out_s).to(self.device)
            self.g_layers.append(l) 
        self.g_layers = nn.ModuleList(self.g_layers)
    
    def forward(self, x):
        """
        Implements the forward method of nn.Module Class
        Args:
            x(Tensor): batch_size x seqence_size x feature_size
        Returns:
            Tensor
        """
        b, d, k = x.size()   

        # cast all pairs against each other
        x_i = torch.unsqueeze(x, 1)                   # (B x 1 x d x 26)
        x_i = x_i.repeat(1, d, 1, 1)                    # (B x d x d x 26)

        x_j = torch.unsqueeze(x, 2)                   # (B x d x 1 x 26)
        x_j = x_j.repeat(1, 1, d, 1)                    # (B x d x d x 26)
        
        # concatenate all together
        x_full = torch.cat([x_i, x_j], 3)         # (B x d x d x 2*26)
        x_full = self.edge_feature(x_full)       # (B x d x d x 2*26)
        
        # reshape for passing through network
        x_ = x_full.view(b * d**2, self.in_size)

        for idx, (g_layer, g_layer_size) in enumerate(zip(self.g_layers, self.g_layers_size)):          
            x_ = g_layer(x_)
            x_ = F.relu(x_)

        # reshape again and sum
        x_g = x_.view(b, d**2, self.g_layers_size[-1])
        x_g = x_g.sum(1).squeeze(1)
        
        """f"""
        x_f = self.f_fc1(x_g)
        x_f = F.relu(x_f)
        x_f = self.f_fc2(x_f)
        x_f = self.dropout(x_f)
        x_f = F.relu(x_f)
        x_f = self.f_fc3(x_f)
        return x_f


