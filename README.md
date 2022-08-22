# U-Net semantic segmentation model to detect ships
The model is created to label ship's pixels on images.
The dataset used: https://www.kaggle.com/datasets/mikaelstrauhs/airbus-ship-detection-train-set-70
## Main Files:
train.py -- Python file for model training
model.py -- Model realization  
predict.py -- Get predicted by model image (model is not properly trained yet so doesn't work for now)

### Model
For building model I used U-net with 4 encoding blocks and 3 decoding followed by convolutional layer and another (1x1) convolutional layer as the output layer. Encoding block consists of 2 convolutional layer and the maxpolling layer to decrease by 2 times height and width while increasing amount of filters twice. And decoding block consists of transposed convolution layers to increase height and width while decreasing amount of filters and also 2 convolutional layers. Encoding and Decoding blocks are connected with skip connection, so that after transposed convolution decoding layer is concatenated with corresponding encoding layer.
