# xG-CNN
This repository allows you to fully reproduce the results of our research we submitted to the 10th Workshop on Machine Learning and Data Mining for Sports Analytics (Turin, 2023).

You can train the model yourself either re-creating the input from scratch, or using a provided input.

## Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/mttmtt31/xg_cnn.git

2. Install the packages
   ```shell
   cd xg_cnn
   pip install -r requirements.txt

## (Optional) Individuate shots
1. To fully reproduce our results and retrieve the shots' json file already located in the `data/shots.json`, you can run the `create_json.py` script located in the `src/utils` directory. This command simply creates a json file which is looked up when representing the shot frame without invoking the API each time.
  
   ```shell
   python src/utils/create_json.py
   ```

## (Optional) Shot frames generation
Each shot can be represented in two ways: as a tensor or as a picture in `.png` format. Either way, the input can also be downloaded.
1. Run the `create_pictures.py` script located in the `src/utils` directory. 
   ```shell
   python src/utils/create_pictures.py --s
   ```

    --**s**: Specifies the size of the scatter plots. In our implementation, `s=1`.

2. Run the `create_tensor.py` script located in the `src/utils` directory. 
   ```shell
   python src/utils/create_tensor.py
   ``` 
If you created the input files, you can skip the following section and go directly to the **Train the model** section. 

## Download data
To download data, run the following bash script.
```
$ bash download.sh
```

## Train the model
1. Run the `main.py` script:
   ```shell
   python main.py --device --batch-size --learning-rate --dropout  --epochs --version --augmentation --picture --optim --weight-decay --n-runs --wandb
   ```

    **--device**: Specifies the device for training

    **--batch-size**: Specifies the batch size for training.

    **--learning-rate**: Specifies the learning rate. 

   **--dropout**: Specifies the dropout in the fully connected layer. 

   **--epochs**: Specifies the number of epochs for training.

   **--version**: Architecture to use. At the moment, only possible choice is 'v1', *i.e.*, the only implemented architecture.

   **--augmentation**: If specified, performs augmentation as described in the paper.

   **--picture**: If specified, pictures are used (instead of tensors).

   **--optim** : Specifies the optimiser to use.

   **--weight-decay** : Specifies the weight decay.

   **--n-runs** : Specifies how many times each training run is repeated (in order to obtain more reliable results).

   **-wandb**: If specified, logs results in Weights & Biases (wandb). When logging in wandb, please make sure wandb is currectly configured on your devie.
