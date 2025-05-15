import torch
class MLP(torch.nn.Module):
    def __init__(self,args):
        super(MLP, self).__init__()
        self.feature_extract = torch.nn.Sequential(
            torch.nn.Linear(args['input_size'], 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, args['feature_dim']),
        )
    def forward(self, x):
        x = self.feature_extract(x)
        return {
            'features': x
        }
