from numpy import ndarray, unique, where
from pandas import read_csv
from os.path import join, isdir
from os import mkdir
from shutil import copy2
from sklearn.model_selection import train_test_split
from PIL import Image
from argparse import ArgumentParser
from utils import float_0_1_range


def arg_parse():
    """
    A function for parsing the provided command line arguments

    Returns:
        ArgumentParser: The argument parser object
    """
    
    parser = ArgumentParser(
        description="Data Preparation Script")
    
    parser.add_argument("--image_path", type=str,
                required=True,
                action="store", metavar="PATH",
                help="The path to the images.")
    
    parser.add_argument("--dst_path", type=str,
                required=True,
                action="store", metavar="PATH",
                help="The destination path of the structured dataset.")
    
    parser.add_argument("--label_path", type=str,
                required=True,
                action="store", metavar="PATH",
                help="The path to the csv file containing the label information.")

    parser.add_argument("--test_size", type=float_0_1_range,
                            required=False, action="store", metavar="TEST_SIZE",
                            default=0.3, help="The size of the test set.")
    
    parser.add_argument("--validation_size", type=float_0_1_range,
                            required=False, action="store", metavar="VAL_SIZE",
                            default=0.1, help="The percentage of the train set utilized for validation.")
    
    args = parser.parse_args()
    
    return args

def structure_dataset(image_path: str,
                      label_path: str,
                      out_path: str = ".",
                      test_size: float = 0.3,
                      validation_size: float = 0.1):
    
    """This is a function that receives a folder of images (```image_path```) and a csv file mapping each filename to a label (```label_path```) 
    and structures the dataset for training validation and testing. Each split has subdirectories containing the appropriate images
    for each class indicated by unique labels in the csv file and splited to a percentage indicated by ```test_size``` and ```validation_size``` parameters.
    The directory structure is created at the ```out_path``` destination.

    Args:
        image_path (str): the path to the folder containing the images.
        label_path (str): the path to the csv file.
        out_path (str, optional): the directory that will contain the directory structure. Defaults to ".".
        test_size (float, optional): The size of the test set. Defaults to 0.3.
        validation_size (float, optional): The percentage of the train set utilized for validation. Defaults to 0.3.
    """
    
    # create destination folder
    if not isdir(out_path):
        mkdir(out_path)
    
    # read the label csv
    label_info = read_csv(label_path)

    # there are NaN values in the image name list. This originates from the fact that
    # some image paths do not correspond to actual images thus we remove these rows 
    # because there is no correspondence to actual data
    label_info = label_info.dropna(axis=0)
    
    # create train validation and test splits
    x_train, x_test, y_train, y_test = train_test_split(
        label_info["image_name"].values,
        label_info["label"].values,
        test_size=test_size,
        stratify=label_info["label"].values,
        random_state=42
    )
    
    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=validation_size,
        stratify=y_train,
        random_state=42
    )
    
    # create the subdirectories
    create_data_subset(subset_filenames=x_train,
                       subset_labels=y_train,
                       src_path=image_path,
                       out_path=out_path,
                       split_name="train")
    
    create_data_subset(subset_filenames=x_val,
                       subset_labels=y_val,
                       src_path=image_path,
                       out_path=out_path,
                       split_name="val")
    
    create_data_subset(subset_filenames=x_test,
                       subset_labels=y_test,
                       src_path=image_path,
                       out_path=out_path,
                       split_name="test")
    
def create_data_subset(subset_filenames: ndarray,
                       subset_labels: ndarray,
                       src_path: str = ".",
                       out_path: str = ".",
                       split_name: str = "train"):
    """Create a directory contain a subset of labeled images

    Args:
        subset_filenames (ndarray): image filenames of the subset.
        subset_labels (ndarray): image labels of the subset
        src_path (str, optional): source path containing the images. Defaults to ".".
        out_path (str, optional): destination path of the created subset. Defaults to ".".
        split_path (str, optional): the name of the split. Defaults to "train".
    """
    # if the split directory does not exist create it
    dst_path = join(out_path, split_name)
    if not isdir(dst_path):
        mkdir(dst_path)
    
    # for each class create a directory inside the split directory 
    # with class name and the corresponding images of the class as contents
    for class_name in unique(subset_labels):

        # create a path of the form outPath/splitName/class_name
        class_path = join(dst_path, class_name) 
        mkdir(class_path)
        
        # find the rows containing the class name
        class_locations = where(subset_labels == class_name)[0]
        
        print(f"{class_name} entries {class_locations.shape[0]}")
        ncorruptions = 0
        for _, filename in enumerate(subset_filenames[class_locations]):
            
            # check for corruption
            isCorrupted = False
            try:
                with Image.open(join(src_path, filename)) as img:
                    img.verify()
            except:
                print(f"Image {filename} is corrupted: ")
                ncorruptions += 1
                isCorrupted = True
            
            # if they are not corrupted move them to the corresponding directory
            if not isCorrupted:
                copy2(join(src_path, filename), join(class_path, filename))
        
        print(f"{class_name} corruptions {ncorruptions}")
        

if __name__ == "__main__":
    args = arg_parse()
    
    structure_dataset(image_path=args.image_path, 
                      label_path=args.label_path,
                      out_path=args.dst_path,
                      test_size=args.test_size,
                      validation_size=args.validation_size
                      )