import torch.optim as optim


def build_optimizer(config, model):
    """ Initialize an optimizer for the parameters of the model,
    set weight decay of normalization to 0 by default. """

    optimizer_name = config.optim.name.lower()
    if optimizer_name == 'adam':
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(parameters, config.optim.lr)
    elif optimizer_name == 'adamw':
        parameters = set_weight_decay(model)
        optimizer = optim.AdamW(parameters, eps=config.optim.eps, betas=config.optim.betas,
                                lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    elif optimizer_name == 'sgd':
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.SGD(parameters, config.optim.lr, momentum=config.optim.momentum,
                              weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(f'Unsupported optimizer: {optimizer_name}')

    return optimizer


def set_weight_decay(model):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]
