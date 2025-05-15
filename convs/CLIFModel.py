import torch
import torch.nn as nn
from convs.layers import GIN,FDModel
import math


# class MLP(torch.nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.feature_extract = torch.nn.Sequential(
#             torch.nn.Linear(2880, 512),
#             torch.nn.ReLU(),
#             torch.nn.Linear(512, 512),
#             torch.nn.ReLU(),
#             torch.nn.Linear(512, 64),
#         )
#     def forward(self, x):
#         x = self.feature_extract(x)
#         return x

class CLIFNet_new(nn.Module):
    def __init__(self,args):
        super(CLIFNet_new, self).__init__()
        self.rand_seed = 0

        # Label semantic encoding module
        n_class = args['total_class']
        self.label_embedding = nn.Parameter(torch.eye(n_class),
                                            requires_grad=False)

        # self.GIN_encoder = GIN(2, n_class, 256,
        #                        [math.ceil(256 / 2)])
        # self.FD_model = FDModel(in_features_x=args["input_size"], in_features_y=256, hidden_features=64, out_features=64, in_layers1=3,
        #                         out_layers=1, batchNorm=False, nonlinearity='relu', negative_slope=0.1)
        self.GIN_encoder = GIN(1, n_class, 256,
                               [math.ceil(256 / 2)])
        self.FD_model = FDModel(in_features_x=args["input_size"], in_features_y=256, hidden_features=64, out_features=64, in_layers1=2,
                                out_layers=1, batchNorm=False, nonlinearity='relu', negative_slope=0.1)

        self.reset_parameters()

    def reset_parameters(self):
        # Init_random_seed(self.rand_seed)
        nn.init.normal_(self.label_embedding)
        self.GIN_encoder.reset_parameters()
        self.FD_model.reset_parameters()
        # self.cls_conv.reset_parameters()
    def get_config_optim(self):
        return [{'params': self.GIN_encoder.parameters()},
                {'params': self.FD_model.parameters()}]

    def forward(self,input,label_adj):
        # Generating semantic label embeddings via label semantic encoding module
        label_embedding = self.GIN_encoder(self.label_embedding[:label_adj.shape[0]], label_adj)
        feature,X = self.FD_model(input, label_embedding)
        # output = self.cls_conv(X).squeeze(2)
        return{
            'uni_features':feature,
            'dis_features':X,
            'label_embedding':label_embedding
        }






