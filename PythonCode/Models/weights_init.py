from torch.nn import init

###############################################################################
# Choose a Function to Make the Weights' Initialization
###############################################################################

def weights_init(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


###############################################################################
# Functions
###############################################################################
        
def weights_init_normal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight, 1.0, 0.02)
        init.constant_(m.bias, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight, gain=1)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight, 1.0, 0.02)
        init.constant_(m.bias, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')  # nonlinearity='leaky_relu'
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')  # nonlinearity='leaky_relu'
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight, 1.0, 0.02)
        init.constant_(m.bias, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight, 1.0, 0.02)
        init.constant_(m.bias, 0.0)
