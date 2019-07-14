from torch.nn.init import normal_, xavier_normal_, xavier_uniform_, kaiming_normal_, kaiming_uniform_, uniform_, orthogonal_

def weigths_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        normal_(m.weight.data)

def weigths_init_xavier_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        xavier_normal_(m.weight.data)

def weigths_init_xavier_uniform(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        xavier_uniform_(m.weight.data)

def weigths_init_kaiming_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        kaiming_normal_(m.weight.data)

def weigths_init_kaiming_uniform(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        kaiming_uniform_(m.weight.data)

def weigths_init_uniform(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        uniform_(m.weight.data)

def weigths_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        orthogonal_(m.weight.data)
