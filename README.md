# xG-CNN


## Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/your-username/project.git

2. Install the packages
   ```shell
   cd xg-cnn
   pip install -r requirements.txt

## (Optional) Individuate shots
1. To fully reproduce our results and retrieve the shots' json file already located in the `data/shots.json`, you can run the `create_json.py` script located in the `src/utils` directory. 
   ```shell
   python src/utils/create_json.py
   ```

## Shot frames generation
1. Run the `create_pictures.py` script located in the `src/utils` directory. 
   ```shell
   python src/utils/create_pictures.py --s <size> [--voronoi] [--cones]
   ```

    --**s**: Specify the size of the scatter plots. In our implementation, `s=2` when `--cones`, otherwise always `s=5`.  

    --**voronoi**: (Optional) Generate scatter plots with Voronoi diagrams as background. Will generate two types of pictures, one with the Voronoi diagram defined on the whole pitch and one with the Voronoi diagram defined only in the visible frame. The pictures will be store in the subfolders `images/all` and `images/visible`, respectively.
    
    --**cones**: (Optional) Generate scatter plots with cones around each player. The cones represent the shadow of a player with respect to the ball. If specified, the images will be store in the subfolder `images/cones`. 

    If neither --**voronoi** nor --**cones** is specified, the scatter plots will be plotted on a white background with no additional feature in the `images/white/` directory.

    A preview of the pictures can be found in the `sample_images` folder. 

## Train the model
1. Run the `main.py` script:
   ```shell
   python main.py --device <device> --batch-size <batch_size> --learning-rate <lr> --epochs <epochs> --picture_type <picture_type> [--log-wandb]
   ```

    **--device**: Specify the device for training

    **--batch-size**: Specify the batch size for training.

    **--learning-rate**: Specify the learning rate. 

    **--epochs**: Specify the number of epochs for training.

    **--picture-type**: Specify the type of pictures to use (all, cones, visible, white).

    **-wandb**: (Optional) Log results in Weights & Biases (wandb). When logging in wandb, please make sure wandb is currectly configured on your devie.
