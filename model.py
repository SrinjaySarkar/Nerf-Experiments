import torch
import torch.nn.functional as F
import numpy as np

# class pos_encoding(torch.nn.Module):
#     def __init__(self,L,include_input,log_sampling):
#         super(pos_encoding,self).__init__()
#         self.L=L
#         self.include_input=include_input
#         self.log_sampling=log_sampling
#         self.enc_func=[torch.sin,torch.cos]
#         if self.log_sampling:
#             self.frequencies=2.0**torch.linspace(0.0,self.L-1,self.L)
#         else:
#             self.frequencies=torch.linspace(2.0**0.0,2.0**(self.L-1),self.L)
    
#     def forward(self,p):
#         if self.include_input:
#             op=[p]
#         else:
#             op=[]
#         for freq in self.frequencies:
#             for enc in self.enc_func:
#                 op.append(enc(p*freq))
#         op=torch.cat(op,dim=-1)
#         return (op)

# def get_embedding_function(L,include_input=True,log_sampling=True):
#     encoder_function=pos_encoding(L,include_input,log_sampling)
#     return (lambda x:encoder_function(x))



def positional_encoding(tensor,num_encoding_functions=6,include_input=True,log_sampling=True):
    encoding=[tensor] if include_input else []
    frequency_bands=None
    if log_sampling:
        frequency_bands=2.0**torch.linspace(0.0,num_encoding_functions-1,num_encoding_functions,dtype=tensor.dtype,device=tensor.device)
    else:
        frequency_bands = torch.linspace(2.0**0.0,2.0**(num_encoding_functions-1),num_encoding_functions,dtype=tensor.dtype,device=tensor.device)

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)


def get_embedding_function(num_encoding_functions=6, include_input=True, log_sampling=True):
    return lambda x:positional_encoding(x,num_encoding_functions,include_input,log_sampling)



class nerf(torch.nn.Module):
    def __init__(self,filter_size,L,directions=True):
        super(nerf,self).__init__()
        self.L=L
        self.xyz_encoding_dims=3+3*2*L
        if directions:
            self.dir_encoding_dims=3+3*2*L
        else:
            self.dir_encoding_dims=0
        self.layer1=torch.nn.Linear(self.xyz_encoding_dims+self.dir_encoding_dims,filter_size)
        self.layer2=torch.nn.Linear(filter_size,filter_size)
        self.layer3=torch.nn.Linear(filter_size,4)
    
    def forward(self,p):
        p=F.relu(self.layer1(p))
        p=F.relu(self.layer2(p))
        p=self.layer3(p)
        return (p)


class nerf2(torch.nn.Module):
    def __init__(self,num_layers=4,hidden_size=128,skip_connect_every=4,L_xyz=6,L_dir=4,include_xyz=True,include_dir=True,use_viewdirs=True):
        
        super(nerf2,self).__init__()
        include_input_xyz=3 if include_xyz else 0
        include_input_dir=3 if include_xyz else 0
        self.xyz_dim=include_input_xyz+2*3*L_xyz
        self.dir_dim=include_input_dir+2*3*L_dir
        self.skip_connect_every=skip_connect_every
        if not use_viewdirs:
            self.dir_dim=0
        self.layer1=torch.nn.Linear(self.xyz_dim,hidden_size)
        self.layers_xyz=torch.nn.ModuleList()
        for i in range(num_layers-1):
            if i%self.skip_connect_every == 0 and i>0 and i!=num_layers-1:
                self.layers_xyz.append(torch.nn.Linear(self.xyz_dim+hidden_size,hidden_size))
            else:
                self.layers_xyz.append(torch.nn.Linear(hidden_size,hidden_size))
        self.use_viewdirs=use_viewdirs
        if self.use_viewdirs:
            self.layers_dir=torch.nn.ModuleList()
            self.layers_dir.append(torch.nn.Linear(self.dir_dim+hidden_size,hidden_size//2))
            self.fc_alpha=torch.nn.Linear(hidden_size,1)
            self.fc_rgb=torch.nn.Linear(hidden_size//2,3)
            self.fc_feat=torch.nn.Linear(hidden_size,hidden_size)
        else:
            self.fc_out=torch.nn.Linear(hidden_size,4)

    def forward(self,p):
        if self.use_viewdirs:
            xyz=p[...,:self.xyz_dim]
            view=p[...,self.xyz_dim:]
        else:
            xyz=p[...,:self.xyz_dim]
        p=self.layer1(xyz)
        for i in range(len(self.layers_xyz)):
            if (i%self.skip_connect_every==0 and i>0 and i!=len(self.linear_layers)-1):
                p=torch.cat((p,xyz),dim=-1)
            p=F.relu(self.layers_xyz[i](p))
        if self.use_viewdirs:
            feat=F.relu(self.fc_feat(p))
            alpha=self.fc_alpha(p)
            p=torch.cat((feat,view),dim=-1)
            for l in self.layers_dir:
                p=F.relu(l(p))
            rgb=self.fc_rgb(p)
            return (torch.cat((rgb, alpha),dim=-1))
        else:
            return (self.fc_out(p))



class FlexibleNeRFModel(torch.nn.Module):
    def __init__(
        self,
        num_layers=4,
        hidden_size=128,
        skip_connect_every=4,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
        use_viewdirs=True,
    ):
        super(FlexibleNeRFModel, self).__init__()

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
        self.skip_connect_every = skip_connect_every
        if not use_viewdirs:
            self.dim_dir = 0

        self.layer1 = torch.nn.Linear(self.dim_xyz, hidden_size)
        self.layers_xyz = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            if i % self.skip_connect_every == 0 and i > 0 and i != num_layers - 1:
                self.layers_xyz.append(
                    torch.nn.Linear(self.dim_xyz + hidden_size, hidden_size)
                )
            else:
                self.layers_xyz.append(torch.nn.Linear(hidden_size, hidden_size))

        self.use_viewdirs = use_viewdirs
        if self.use_viewdirs:
            self.layers_dir = torch.nn.ModuleList()
            # This deviates from the original paper, and follows the code release instead.
            self.layers_dir.append(
                torch.nn.Linear(self.dim_dir + hidden_size, hidden_size // 2)
            )

            self.fc_alpha = torch.nn.Linear(hidden_size, 1)
            self.fc_rgb = torch.nn.Linear(hidden_size // 2, 3)
            self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)
        else:
            self.fc_out = torch.nn.Linear(hidden_size, 4)

        self.relu = torch.nn.functional.relu

    def forward(self, x):
        if self.use_viewdirs:
            xyz, view = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
        else:
            xyz = x[..., : self.dim_xyz]
        x = self.layer1(xyz)
        for i in range(len(self.layers_xyz)):
            if (
                i % self.skip_connect_every == 0
                and i > 0
                and i != len(self.linear_layers) - 1
            ):
                x = torch.cat((x, xyz), dim=-1)
            x = self.relu(self.layers_xyz[i](x))
        if self.use_viewdirs:
            feat = self.relu(self.fc_feat(x))
            alpha = self.fc_alpha(x)
            x = torch.cat((feat, view), dim=-1)
            for l in self.layers_dir:
                x = self.relu(l(x))
            rgb = self.fc_rgb(x)
            return torch.cat((rgb, alpha), dim=-1)
        else:
            return self.fc_out(x)