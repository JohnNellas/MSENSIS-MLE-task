import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from os import mkdir
from os.path import join, isdir
import math
from argparse import ArgumentParser
from utils import get_pretrained_model, non_negative_float_input, non_negative_int_input, relation_wrapper

def arg_parse():
    """
    A function for parsing the provided command line arguments

    Returns:
        ArgumentParser: The argument parser object
    """
    
    parser = ArgumentParser(
        description="Fine tune a custom model")
    
    parser.add_argument("--data_path", type=str,
                required=True,
                action="store", metavar="PATH",
                help="The path to the dataset splits.")
    
    parser.add_argument("--checkpoints_path", type=str,
                required=False, default=join(".", "checkpoints"),
                action="store", metavar="PATH",
                help="The path to the dataset splits.")
    
    parser.add_argument("--model_name", type=str,
                required=True,
                action="store", metavar="MODEL_NAME",
                help="The name of the model. Supported pretrained models are resnet, vgg, mobilenet family from pytorch official site ")
    
    parser.add_argument("--monitor_metric_name", type=str,
                required=False, choices=["accuracy", "loss"], default="accuracy",
                action="store", metavar="METRIC_NAME",
                help="The name of the metric to be monitored for best metric values for checkpoints.")
    

    parser.add_argument("--lr", type=non_negative_float_input,
                            required=False, action="store", metavar="LEARNING_RATE",
                            default=0.001, help="The learning rate.")
    
    parser.add_argument("--nclasses", type=non_negative_int_input,
                            required=False, action="store", metavar="NCLASSES",
                            default=2, help="The number of classes.")
    
    parser.add_argument("--batch_size", type=non_negative_int_input,
                            required=False, action="store", metavar="BATCH_SIZE",
                            default=32, help="The batch size.")
    
    parser.add_argument("--epochs", type=non_negative_int_input,
                            required=False, action="store", metavar="N_EPOCHS",
                            default=3, help="The number of epochs.")
    
    parser.add_argument("--device", type=str,
                required=False, default="mps",
                action="store", metavar="DEVICE",
                help="The utilized device.")
    
    args = parser.parse_args()
    
    return args

def train_epoch(dataloader: DataLoader,
                model: nn.Module,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                device: str) -> dict:
    """A training step

    Args:
        dataloader (DataLoader): the training dataloader
        model (nn.Module): the model to be trained
        criterion (nn.Module): the loss function
        optimizer (torch.optim.Optimizer): the optimizer
        device (str): the utilized device for training

    Returns:
        dict: the values of each metric for the epoch
    """
    
    
    total_loss = 0
    nbatches = len(dataloader)
    
    model.train()
    
    for images, labels in dataloader:
        
        # send batch to device
        images, labels = images.to(device), labels.to(device)
        
        # calculate outputs
        outputs = model(images)
        
        # calculate loss
        loss = criterion(outputs, labels)
        
        # calculate gradients and perform weight updates
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # average over batches
    total_loss = total_loss/nbatches
    
    return {"loss": total_loss}



def evaluate_model(dataloader: DataLoader,
             model: nn.Module,
             criterion: nn.Module,
             device: str):
    """Evaluate a model on a dataloader

    Args:
        dataloader (DataLoader): the utilized dataloader for evaluation
        model (nn.Module): the model
        criterion (nn.Module): the loss function
        device (str): _description_

    Returns:
        _type_: _description_
    """
    total_loss = 0
    correct = 0
    nbatches = len(dataloader)
    size = len(dataloader.dataset)
    
    # set model to evaluation mode
    model.eval()
    
    for images, labels in dataloader:
        
        # send batch to device
        images, labels = images.to(device), labels.to(device)
        
        # get model outputs
        outputs = model(images)
        
        # calculate losses for batch
        loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()
    
    total_loss = total_loss/nbatches
    accuracy = correct/size
    
    
    return {"loss": total_loss,
            "accuracy": accuracy}

def train_model(args: ArgumentParser):
    """
    The function that trains a pretrained model

    Args:
        args (ArgumentParser): The provided command line argument values
    """
    # create the checkpoint folder
    if not isdir(args.checkpoints_path):
        mkdir(args.checkpoints_path)
    
    # Pre-processing transforms 
    dtransform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # set up the training and validation datasets along with their
    # corresponding dataloaders 
    train_dataset = ImageFolder(root=join(args.data_path, "train"),
                                transform=dtransform
                                )
    
    validation_dataset = ImageFolder(root=join(args.data_path, "val"),
                                     transform=dtransform
                                     )
    
    train_dataloader = DataLoader(train_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            pin_memory=True)
    
    validation_dataloader = DataLoader(validation_dataset,
                                       batch_size=args.batch_size,
                                       shuffle=False)

    # get the specified model
    model = get_pretrained_model(args.model_name,
                                  num_classes=args.nclasses,
                                  pretrained=True
                                  )
    model = model.to(args.device)
    
    # set up training settings
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr)

    # Training Loop
    print("Starting training...")
    current_best = 0 if args.monitor_metric_name == "accuracy" else math.inf
    comparison = relation_wrapper("greater") if args.monitor_metric_name == "accuracy" else relation_wrapper("less")
    
    
    for epoch in range(args.epochs): # Keep it short for the task demo
        print(f"Epoch {epoch+1}")
        
        # train for an epoch
        metrics = train_epoch(
            dataloader=train_dataloader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=args.device
        )
        
        print("Train")
        print(metrics)
        
        # validate model
        eval_metrics = evaluate_model(
            dataloader=validation_dataloader,
            model=model,
            criterion=criterion,
            device=args.device
        )
        print("Validation")
        print(eval_metrics)
        
        # Save a model checkpoint if the performance achieves better metric value
        if comparison(eval_metrics[args.monitor_metric_name], current_best):
            
            print(f"new best found based on {args.monitor_metric_name}, from {current_best} -> {eval_metrics[args.monitor_metric_name]}")
            print("Saving...")
            torch.save(model.state_dict(),
               join(args.checkpoints_path, f"{args.model_name}_cat_dog_model.pth"))
            current_best = eval_metrics[args.monitor_metric_name]
        
        print("---------------")
            

if __name__ == "__main__":
    args = arg_parse()
    train_model(args)