import torch
import torch.nn.functional as F

# DCGAN loss
def loss_dcgan_dis(dis_fake, dis_real):
  L1 = torch.mean(F.softplus(-dis_real))
  L2 = torch.mean(F.softplus(dis_fake))
  return L1, L2


def loss_dcgan_gen(dis_fake):
  loss = torch.mean(F.softplus(-dis_fake))
  return loss


# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real):
  loss_real = torch.mean(F.relu(1. - dis_real))
  loss_fake = torch.mean(F.relu(1. + dis_fake))
  return loss_real, loss_fake
# def loss_hinge_dis(dis_fake, dis_real): # This version returns a single loss
  # loss = torch.mean(F.relu(1. - dis_real))
  # loss += torch.mean(F.relu(1. + dis_fake))
  # return loss


def loss_hinge_gen(dis_fake):
  loss = -torch.mean(dis_fake)
  return loss

# Default to hinge loss
generator_loss = loss_hinge_gen
discriminator_loss = loss_hinge_dis

def crammer_singer_criterion(X, Ylabel):
  num_real_classes = X.shape[1] - 1
  mask = torch.ones_like(X).bool()
  mask.scatter_(1, Ylabel.unsqueeze(-1), 0)
  wrongs = torch.masked_select(X,mask.bool()).reshape(X.shape[0], num_real_classes)
  max_wrong, _ = wrongs.max(1)
  max_wrong = max_wrong.unsqueeze(-1)
  target = X.gather(1,Ylabel.unsqueeze(-1))
  return torch.mean(F.relu(1 + max_wrong - target))
    
def crammer_singer_complement_criterion(X, Ylabel):
  num_real_classes = X.shape[1] - 1
  mask = torch.ones_like(X).bool()
  mask.scatter_(1, Ylabel.unsqueeze(-1), 0)
  wrongs = torch.masked_select(X,mask.bool()).reshape(X.shape[0], num_real_classes)
  max_wrong, _ = wrongs.max(1)
  max_wrong = max_wrong.unsqueeze(-1)
  target = X.gather(1,Ylabel.unsqueeze(-1))
  return torch.mean(F.relu(1 - max_wrong + target))
  
def not_fake_criterion(X, Ylabel):
  wrong = X[:,-1]
  target = X.gather(1,Ylabel.unsqueeze(-1))
  return torch.mean(F.relu(1 + wrong - target))
  
def linear_wrong(X, Ylabel):
  wrong = X[:,-1]
  return torch.mean( wrong )

def linear_right(X, Ylabel):
  right = X[:,:-1]
  return -torch.mean( right )
  
def linear_target(X, Ylabel):
  target = X.gather(1,Ylabel.unsqueeze(-1))
  return torch.mean( -target )
  
def weston_watkins_criterion(X, Ylabel):
  num_real_classes = X.shape[1] - 1
  mask = torch.ones_like(X)
  mask.scatter_(1, Ylabel.unsqueeze(-1), 0)
  wrongs = torch.masked_select(X,mask.byte()).reshape(X.shape[0], num_real_classes)
  target = X.gather(1,Ylabel.unsqueeze(-1))
  return torch.mean(F.relu(1 + wrongs - target))
  #return torch.mean(torch.sum(F.relu(1 + wrongs - target), 1))
  
mh_loss = crammer_singer_criterion
