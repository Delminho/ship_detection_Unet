# U-Net semantic segmentation model to detect ships
The model is created to label ship's pixels on images.
The data is taken from kaggle competition: https://www.kaggle.com/competitions/airbus-ship-detection
## Main Files:
train.py -- Python file for model training  
model.py -- Model realization  
predict.py -- Python file to predict mask for a given image
## How to use:
#### Training
Install required packages
```
pip install -r requirements.txt
```
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
The model is a U-net architecture trained on 256x256 images.  
The loss function is a Dice score.  
Optimizer chosen is an `Adam` with lr=0.0007.  
Since it is Fully Convolutional, it also works with different sizes.  
My U-net model has 4 encoding and 3 decoding blocks, followed by a CONV layer and another (1x1) CONV as an output layer.  
Also, `BatchNormalize` layers were added.  
It is trained on a dataset resized to 256x256 from 768x768 and with downsampled shipless images.

### Project Structure

```
.
├── Analysis_Modeling.ipynb      # Ipython file with analysis and modeling
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
└── train.py
```
### Results
I got a following results (on 256x256 images):
- Training set: 0.78 Dice and 0.8 IoU
- Validation set: 0.77 Dice and 0.8 IoU
- Test set: 0.76 Dice and 0.795 IoU

### Ways to improve
1. The model can be improved by adding a classifier for images `has_ship` is 0 or 1. Without it, the U-net model makes a lot of false positives on shipless images. And shipless images are the majority. And this would allow training U-net only on images with ships, making the model indentify ships better.
2. Data augmentation can be added.
3. Instead of resizing full images to 256x256 in order to train, we could split original image to 9 256x256 images and train on them. That would increase the dataset's size. Another way is sampling 256x256 rectangles from original images to train.  
4. Make more filters. Due to lack of time and computational resources, I only used from 4 to 32 filters for Convolutional layers. Original paper had from 64 to 1024.