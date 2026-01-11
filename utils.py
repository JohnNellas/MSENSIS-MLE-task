from argparse import ArgumentTypeError
from torchvision import models
import torch.nn as nn

def get_pretrained_model(model_name: str,
                          num_classes: int, 
                          pretrained: bool = True):
    """Factory to return pre-trained models with an appended classification head.

    Args:
        model_name (str): The name of the model
        num_classes (int): The number of classes
        pretrained (bool, optional): If pretrained model is desired. Defaults to True.

    Raises:
        ValueError: if the desired model is not supported

    Returns:
        nn.Module: the desired model
    """
    
    model_name = model_name.lower()
    weights = "DEFAULT" if pretrained else None
    
    if model_name.startswith('resnet'):
        # Works for resnet18, resnet34, resnet50, resnet101
        model = getattr(models, model_name)(weights=weights)
        
        #for param in model.parameters():
        #    param.requires_grad = False
        
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    elif model_name.startswith('vgg'):
        model = getattr(models, model_name)(weights=weights)
        
        #for param in model.parameters():
        #    param.requires_grad = False
        
        # VGG head is a Sequential block; last layer is at index 6
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)

    elif model_name.startswith('mobilenet'):
        model = getattr(models, model_name)(weights=weights)
        
        #for param in model.parameters():
        #    param.requires_grad = False
        
        last_layer_idx = 3 if 'v3' in model_name else 1
        in_features = model.classifier[last_layer_idx].in_features
        model.classifier[last_layer_idx] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported yet.")

    return model



def float_0_1_range(value):
    """
    A function for checking if an input value is between 0 and 1.
    
    Args:
        value (float): the input value.
        
    Returns:
        float between 0 and 1, otherwise raises an exception.
    """

    try:
        # try to convert input value to float
        value = float(value)

        # if conversion is successful check if the float is not in the range [0,1]
        # if this is true then raise an error
        if value < 0 or value > 1:
            raise ArgumentTypeError(f"{value} is not a non-negative float")
    except ValueError:

        # if conversion to float fails then the input is not a float
        raise ArgumentTypeError(f"{value} is not a float.")

    return value

def non_negative_int_input(value):
    """
    A function for checking if an input value is a non-negative integer.
    
    Args:
        value (int): the input value.
    
    Returns:
        int: the non-negative integer value if this holds, otherwise raises an exception.
    """

    try:
        # try to convert input value to integer
        value = int(value)

        # if conversion is successful check if the integer is non-negative
        if value < 0:
            # raise an exception if the integer is not a non-negative integer
            raise ArgumentTypeError(f"{value} is not a non-negative integer")
    except ValueError:

        # if conversion to integer fails then the input is not an integer
        raise ArgumentTypeError(f"{value} is not an integer.")

    # return the non-negative integer value if every process is successfully completed
    return value

def non_negative_float_input(value):
    """
    A function for checking if an input value is a non-negative float.
    
    Args:
        value (float): the input value.
        
    Returns:
        float the non-negative float if this holds, otherwise raises an exception.
    """

    try:
        # try to convert input value to float
        value = float(value)

        # if conversion is successful check if the float is non-negative
        if value < 0:
            # raise an exception if the float is not a non-negative float
            raise ArgumentTypeError(f"{value} is not a non-negative float")
    except ValueError:

        # if conversion to float fails then the input is not a float
        raise ArgumentTypeError(f"{value} is not a float.")

    # return the non-negative float value if every process is successfully completed
    return value

def relation_wrapper(relation:str):
    """get a less/greater relationship lambda function

    Args:
        relation (str): the desired relationship

    Returns:
        lambda: the less or greater relationship function
    """
    if relation == "less":
        return lambda arg1, arg2: arg1<arg2
    else:
        return lambda arg1, arg2: arg1>arg2