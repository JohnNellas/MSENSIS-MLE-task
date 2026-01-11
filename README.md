# MSENSIS-MLE-task

## Overview
This is the repository for the Machine Learning Engineering Task at MSensis for a dog/cat classifier. Specifically this is a full-stack ML application that utilized either a pre-trained Vision Transformer (ViT) from Hugging Face or a fine tuned model (Only Mobilenet_v3_small is currently supported).

## Tech Stack
* **Model:** Hugging Face Transformers, PyTorch
* **Data:** Pandas, NumPy
* **Backend:** FastAPI
* **Frontend:** Streamlit

## Setup Instructions
1.  Install dependencies:

    `pip install -r requirements.txt`

2. Data Preparation using `data_prep.py`:
    - **Example** - Prepare the dataset with images located at path/to/images, csv file containing the labels located at path/to/csvFile, output_directory set to path/to/output with test size equal to 30% of the original dataset and percentage of the train set utilized for validation equal to 0.1:

        `python3 -m data_prep --image_path path/to/images --dst_path path/to/csvFile --label_path path/to/output`

    - see usage:

        ```
        usage: data_prep.py [-h] --image_path PATH --dst_path PATH --label_path PATH [--test_size TEST_SIZE] [--validation_size VAL_SIZE]

        Data Preparation Script

        options:
        -h, --help            show this help message and exit
        --image_path PATH     The path to the images.
        --dst_path PATH       The destination path of the structured dataset.
        --label_path PATH     The path to the csv file containing the label information.
        --test_size TEST_SIZE
                                The size of the test set.
        --validation_size VAL_SIZE
                                The percentage of the train set utilized for validation.
        ```

   
3. (Optional) Fine tune a pretrained model: 

    - **Example** - Fine-tune a Mobilenet_v3_small model on the prepared dataset which is at the location `path/to/data` and model name set to `mobilenet_v3_small`:

        `python3 -m app.train --data_path path/to/data --model_name mobilenet_v3_small`

    - see usage:
    ```
    usage: train.py [-h] --data_path PATH [--checkpoints_path PATH] --model_name MODEL_NAME [--monitor_metric_name METRIC_NAME] [--lr LEARNING_RATE]
                [--nclasses NCLASSES] [--batch_size BATCH_SIZE] [--epochs N_EPOCHS] [--device DEVICE]

    Fine tune a custom model

    options:
    -h, --help            show this help message and exit
    --data_path PATH      The path to the dataset splits.
    --checkpoints_path PATH
                            The path to the dataset splits.
    --model_name MODEL_NAME
                            The name of the model. Supported pretrained models are resnet, vgg, mobilenet family from pytorch official site
    --monitor_metric_name METRIC_NAME
                            The name of the metric to be monitored for best metric values for checkpoints.
    --lr LEARNING_RATE    The learning rate.
    --nclasses NCLASSES   The number of classes.
    --batch_size BATCH_SIZE
                            The batch size.
    --epochs N_EPOCHS     The number of epochs.
    --device DEVICE       The utilized device.
    ```

4.  Start FastAPI Backend: 

    `uvicorn app.main:app --reload`

5.  Start Streamlit Frontend: 

    `streamlit run ui.py`

## Architecture Decisions

* **ViT:** [Vision Transformers](https://arxiv.org/pdf/2010.11929/1000) capture global context in images better than CNNs, when they are pretrained on large amounts of data offering excellent performance for pre-trained tasks.
* **MobileNet v3 small**: is utilized because it offers efficiency (only 2.5M parameters with 0.06GFLOPS) along with adequate performance for its size (accuracy@1 of 67.6% on ImagNet), enabling efficient fine-tuning and inference with adequate performance.
* **Streamlit:** utilized because it is a powerful open-source Python framework delivering interactive data apps.
* **FastAPI:** FastAPI is a high-performance web framework for building APIs with Python.


## Observations

### Dataset Observations
- The label csv file for the dataset contained NaN values in the label column, which corresponded to image filenames that don't exist in the dataset. These filename entries were removed from the dataset.
- Also, a really small number of images (2 images in total) were corrupted, thus these images were removed.

### Model Observations
- For the image classification task, a pretrained Vision Transformer (ViT) from Hugging face was used, because this model when it is pretrained on wide corpus of image data it achieves excellent performance. This fact makes the model a suitable choice for pretrained tasks.
- For fine tuning, a small and efficient model is employed entitled Mobilenet_v3_small. This allows for efficient fine tuning and inference while achieving adaquete performance (validation set accuracy for 3 epoch fine-tuning is equal to 96.8%).
