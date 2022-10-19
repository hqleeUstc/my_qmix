import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# the same structure as the code in icnn, only consider z = f(y), ignore x
# FC层的输入 输出 size可能要调整，比如调整为[200, 200]
class ICNN(nn.Module):
    def __init__(self, args):
        super(ICNN, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim

        hidden_layer_sizes=[32, 32, 32]
        rank = 1
        dropout=0.03
        activation='celu'
        strong_convexity = 1e-6


        self.strong_convexity = strong_convexity
        self.hidden_layer_sizes = hidden_layer_sizes
        self.droput = dropout
        self.activation = activation
        self.rank = rank   
        self.nLayers = len(hidden_layer_sizes) - 1

        self.quadratic_layers = nn.ModuleList([
            nn.Sequential(
                ConvexQuadratic(in_dim, out_features, rank=rank, bias=True),
                nn.Dropout(dropout)
            )
            for out_features in hidden_layer_sizes
        ])
        
        sizes = zip(hidden_layer_sizes[:-1], hidden_layer_sizes[1:])
        self.convex_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, out_features, bias=False),
                nn.Dropout(dropout)
            )
            for (in_features, out_features) in sizes
        ])
        
        self.final_layer = nn.Linear(hidden_layer_sizes[-1], 1, bias=False)


    def forward(self, agent_qs, states):
        input = agent_qs
        output = self.quadratic_layers[0](input)
        i = 0
        prevZ = agent_qs
        for i in range(self.nLayers + 1):
            sz = self.hidden_layer_sizes[i] if i < self.nLayers else 1
            # z 与y的映射
            linear_zy = nn.Linear(self.hidden_layer_sizes[i - 1], sz, bias = False)
            # z = F.relu(linear_zy(y))
            
            if i > 0:
            #z与z的映射
                linear_zz = nn.Linear(self.hidden_layer_sizes[i - 1], sz, bias = False)
                z_z = linear_zz(prevZ)
                z = z + z_z

            if i < self.nLayers:
                z = F.relu(z)
            prevZ = z


        z = prevZ.reshape(len(prevZ), -1)
        return z


    def push(self, input):
        output = autograd.grad(
            outputs=self.forward(input), inputs=input,
            create_graph=True, retain_graph=True,
            only_inputs=True,
            grad_outputs=torch.ones((input.size()[0], 1)).cuda().float()
        )[0]
        return output    
    
    def convexify(self):
        for layer in self.convex_layers:
            for sublayer in layer:
                if (isinstance(sublayer, nn.Linear)):
                    sublayer.weight.data.clamp_(0)
        self.final_layer.weight.data.clamp_(0)
