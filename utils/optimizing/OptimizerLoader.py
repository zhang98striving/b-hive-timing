from torch.optim import Adam, AdamW, RAdam

def OptimizerLoader(
    optimizer_name,
    learning_rate,
    params,
    weight_decay=0,
    eps=1e-8,
    amsgrad=False,
    fused=False,
):
    optimizer_dict = {
        'RAdam': RAdam,
        'Adam': Adam,
        'AdamW': AdamW
    }
    
    if optimizer_name not in optimizer_dict:
        raise NotImplementedError(f"{optimizer_name} is not implemented")
    
    optimizer_class = optimizer_dict[optimizer_name]
    betas = (0.9, 0.999) if optimizer_name != 'AdamW' else (0.95, 0.999)
    
    optimizer = optimizer_class(
        params,
        lr=learning_rate,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        amsgrad=amsgrad,
        fused=fused
    )
    
    return optimizer
