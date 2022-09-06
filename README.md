# U-Net semantic segmentation model to detect ships
The model is created to label ship's pixels on images.
The dataset used: https://www.kaggle.com/datasets/mikaelstrauhs/airbus-ship-detection-train-set-70
## Main Files:
train.py -- Python file for model training  
model.py -- Model realization  
predict.py -- Python file to predict mask for a given image
## How to use:
#### Training
To train model with your set use
```
python train.py images_folder_path mask_table_path
```
For example right now model is trained with
```
python train.py data/Images/ data/test_ship_segmentations_v3.csv
```
It will train model and save it as trained_model. Then predict.py loads the trained_model and you can use it for predictions
#### Predicting
To predict a mask for an image use(it will save mask to predictions/ folder)
```
python predict.py image_path
```
It will save an image with side by side original image and predicted mask. If you only want mask as an output use:
```
python predict.py image_path -compare_flag false
```
Example:
```
python predict.py data/Images/0a1a7f395.jpg -compare_flag false
```
### Model
For building model I used U-net with 3 encoding blocks and 2 decoding followed by convolutional layer and another (1x1) convolutional layer as the output layer. Encoding block consists of 2 convolutional layer and the maxpolling layer to decrease by 2 times height and width while increasing amount of filters twice. And decoding block consists of transposed convolution layers to increase height and width while decreasing amount of filters and also 2 convolutional layers. Encoding and Decoding blocks are connected with skip connection, so that after transposed convolution decoding layer is concatenated with corresponding encoding layer. Also to reduce overfitting I used dropout technique.

### Project Structure

```
.
├── data
│   └── test_ship_segmentations_v3.csv      #csv file with masks
│   └── Images      # A folder with train images
├── Analysis.ipynb      # Ipython file with analysis
├── logs        # Folder with loss and iou plots
│   ├── model_iou_no_dropout.png    
│   ├── model_iou.png
│   ├── model_loss_no_dropout.png
│   └── model_loss.png
├── model.py    
├── predictions
├── predict.py
├── README.md
├── requirements.txt
├── trained_model   # Saved model to use in predict.py
│   ├── assets
│   ├── keras_metadata.pb
│   ├── saved_model.pb
│   └── variables
│       ├── variables.data-00000-of-00001
│       └── variables.index
├── trained_model_weights.h5    # Saved weights to continue training if needed
└── train.py


```