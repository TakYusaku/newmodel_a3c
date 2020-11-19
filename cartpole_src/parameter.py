import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from torch.optim import Optimizer, RMSprop

def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)

class Policy(nn.Module):
    def __init__(self, dim_obs, num_actions, out_dim=128):
        super(Policy, self).__init__()
        self.head_a = nn.Linear(dim_obs, out_dim)
        self.head_v = nn.Linear(dim_obs, out_dim)
        self.p = nn.Linear(out_dim, num_actions)
        self.v = nn.Linear(out_dim, 1)
        set_init([self.head_a,self.head_v, self.p, self.v])
        self.distribution = torch.distributions.Categorical
        self.train()

    def forward(self, x):
        oa = torch.tanh(self.head_a(x))
        policy = self.p(oa)
        ov = torch.tanh(self.head_v(x))
        value = self.v(ov)
        return policy, value

    def sync(self, global_module):
        for p, gp in zip(self.parameters(), global_module.parameters()):
            p.data = gp.data.clone()

    def get_action(self, s):
        self.eval()
        logits, value = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0], prob, value

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)
        
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss

    def loss_func_etp(self, args, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = args.c_loss_coeff * td.pow(2)
    
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v

        entropy = (probs * (probs + args.eps).log()).sum(dim=1)
        entropy = args.entropy_beta * torch.t(entropy.unsqueeze(0))
        total_loss = torch.add(entropy,c_loss + a_loss).mean()

        return total_loss, c_loss.sum().data.item(), entropy.sum().data.item()
    '''    
    def loss_func_etp(self, args, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)
    
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v

        entropy = -(probs * (probs + args.eps).log()).sum(dim=1)
        entropy = torch.t(entropy.unsqueeze(0))
        total_loss = torch.add(entropy,c_loss + a_loss).mean()

        return total_loss, c_loss.sum().data.item(), entropy.sum().data.item()
    '''