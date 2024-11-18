from torch import optim
from inspect import signature

def OptimizerLoader(
    optimizer_name: str,
    learning_rate: float,
    params,
    **kwargs,
) -> object:
    """
    Loads the optimizer based on the specified name and parameters.

    Args:
        optimizer_name (str): The name of the optimizer to use.
        params: Parameters of the model to optimize.
        learning_rate (float, optional): Learning rate for the optimizer. Default depends on the optimizer.
        **kwargs: Additional keyword arguments for the optimizer.

    Returns:
        torch.optim.Optimizer: An instance of the specified optimizer.

    Raises:
        NotImplementedError: If the optimizer name is not recognized.
    """
    optimizer_dict = {
        'Adam':      optim.Adam,
        'AdamW':     optim.AdamW,
        'Adamax':    optim.Adamax,
        'NAdam':     optim.NAdam,
        'RMSprop':   optim.RMSprop,
        'Adadelta':  optim.Adadelta,
        'Adafactor': optim.Adafactor, 
        'Adagrad':   optim.Adagrad,
        'ASGD':      optim.ASGD,
        'LBFGS':     optim.LBFGS,
        'RAdam':     optim.RAdam,
        'Rprop':     optim.Rprop,
        'SGD':       optim.SGD,
    }

    optimizer_class = optimizer_dict.get(optimizer_name)
    if optimizer_class is None:
        raise NotImplementedError(f"{optimizer_name} is not implemented. Supported optimizers: {list(optimizer_dict.keys())}")

    # Get the signature of the optimizer class to validate kwargs
    optimizer_signature = signature(optimizer_class)
    valid_params = set(optimizer_signature.parameters.keys())
    
    # Filter out invalid kwargs
    valid_kwargs = [k for k in kwargs if k in valid_params]
    
    # Initialize the optimizer with the specified parameters
    return optimizer_class(params, lr=learning_rate, **kwargs)