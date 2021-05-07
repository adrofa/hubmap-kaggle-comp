import torch


def get_optimizer(version, model_parameters):

    if version == "adam_v1":
        # Karpathy Score
        return torch.optim.Adam(params=model_parameters, lr=3e-4)

    elif version == "adam_v2":
        return torch.optim.Adam(params=model_parameters, lr=3e-5)

    elif version == "adam_v3":
        return torch.optim.Adam(params=model_parameters, lr=3e-6)

    elif version == "adam_v4":
        # for model_v13; lr found via lr_finder
        return torch.optim.Adam(params=model_parameters, lr=3.45E-16)

    elif version == "sgd_v0":
        return torch.optim.SGD(params=model_parameters, lr=0.1,
                               momentum=0, dampening=0, weight_decay=0, nesterov=False)

    else:
        # Unknown model version
        raise Exception(f"Optimizer version [{version}] is UNKNOWN!")
