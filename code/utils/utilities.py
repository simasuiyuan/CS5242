import math

def display_num_param(net):
    nb_param = 0
    for param in net.parameters():
        nb_param += param.numel()
    print('There are {} ({:.2f} million) parameters in this neural network'.format(nb_param, nb_param/1e6))

def get_accuracy(scores, labels):
    num_data = scores.size(0)
    predicted_labels = scores.argmax(dim=1)
    indicator = (predicted_labels == labels)
    num_matches = indicator.sum()
    return 100*num_matches.float()/num_data

def normalize_gradient(net):
    grad_norm_sq=0
    for p in net.parameters():
        grad_norm_sq += p.grad.data.norm()**2
    grad_norm=math.sqrt(grad_norm_sq)
    if grad_norm<1e-4:
        net.zero_grad()
        print('grad norm close to zero')
    else:    
        for p in net.parameters():
             p.grad.data.div_(grad_norm)
    return grad_norm