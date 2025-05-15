import torch

class gcn(torch.nn.Module):
    def __init__(self, w_in, w_out):
        super(gcn, self).__init__()
        self.w_in = w_in
        self.w_out = w_out
        # self.weight = nn.Parameter(torch.Tensor(w_in, w_out))
        # self.bias = nn.Parameter(torch.Tensor(w_out))
        self.linear = torch.nn.Linear(64, 1)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def norm(self, H, add=False):
        if add == False:
            H = H * ((torch.eye(H.shape[0]) == 0).type(torch.FloatTensor).cuda())
        else:
            H = H * ((torch.eye(H.shape[0]) == 0).type(torch.FloatTensor).cuda()) + torch.eye(H.shape[0]).type(
                torch.FloatTensor).cuda()
        deg = torch.sum(H, dim=1)
        deg[deg<0] = 0
        deg_inv = deg.pow(-0.5)
        deg_inv[deg_inv == float('inf')] = 0
        deg_inv = deg_inv * torch.eye(H.shape[0]).type(torch.FloatTensor).cuda()
        H = torch.mm(deg_inv, H)
        H = torch.mm(H, deg_inv)
        return H
    def forward(self, X_b, H):
        batch_size = X_b.shape[0]
        output = torch.tensor([]).cuda()
        for i in range(batch_size):
            X = X_b[i,:,:].squeeze(0)
            X = self.linear(X)
            H = self.norm(H, add=True)
            Out = torch.mm(H, X).view(-1).unsqueeze(0)
            output = torch.cat((output, Out), dim=0)
        return output
class AGCNNet(torch.nn.Module):
    def __init__(self,args):
        super(AGCNNet, self).__init__()
        input_size = args['input_size']
        output_size = args['total_class']
        self.classifier1 = torch.nn.Linear(64,output_size)
        self.gcn_layer = gcn(64,1)
        self.alpha = torch.nn.Parameter(torch.tensor(0.0))
        self.feature_extract = torch.nn.Sequential(
            torch.nn.Linear(input_size,512),
            torch.nn.ReLU(),
            torch.nn.Linear(512,64),
        )
    def forward(self, x):
        x = self.feature_extract(x)
        return{'features':x}


# class AGCNNet(torch.nn.Module):
#     def __init__(self,args):
#         super(AGCNNet, self).__init__()
#         input_size = args['input_size']
#         output_size = args['total_class']
#         self.classifier1 = torch.nn.Linear(64,output_size)
#         self.gcn_layer = gcn(64,1)
#         self.alpha = torch.nn.Parameter(torch.tensor(0.0))
#         self.feature_extract = torch.nn.Sequential(
#             torch.nn.Linear(input_size,512),
#             torch.nn.ReLU(),
#             torch.nn.Linear(512,64),
#         )
#     def forward(self, x,H):
#         x = self.feature_extract(x)
#         ya = self.classifier1(x)
#         label_embedding = x.unsqueeze(2).repeat(1, 1, H.shape[0]).permute(0, 2, 1)
#         label_embedding = label_embedding * self.classifier1.weight
#         yb = self.gcn_layer(label_embedding,H)
#         y =  ya + yb
#         return y