# DL_Project1
Ke Zhou, Chenning Li, Di Wu

In this project, we designed a 20 layers Residual Network model and studied the influences of 6 parameters and several regularization strategies on the accuracy of the model. By fine-tuning the the depth (N), the width (C), the block number in each layer(B), the convolutional kernel size (F), the skip connection kernel size (K) and average pooling kernel size (P) in the residual layers we achieved an accuracy of 90% with a model size below 5M parameters.

Through the tuning process, we finally obtained our model to perform an accuracy at 90.5%. The model has an architecture with 3 residual layers, and each layer has 3 residual blocks. The convolution layer behind the input layer with 2x2 kernel size (F1). Each residual block with two 3x3 convolutions. As mentioned above, a large number of channels is preferred, since we observed that channels are the most significant factor for improving accuracy. Then in our model C1 was set to 68. The detailed model parameters are shown in the table below. 

|Name|Value|
|-------------|--------------|
|Residual Blocks in each Residual Layer | [3,3,3]|
|Ci in each Residual Layer | [68,136,272]|
|F1 in Cov| 2|
|K|1|
|P|7|
|Dropout|0|
|Batch Size|64|
|Number of Parameters|4,883,906|
|Accuracy|0.905|

In our final model, we have selected data normalization and converting image to tensor as data augmentation strategies to reduce generalization errors. 

The figure below shows the curve for our final model.

![Final Model Curve](https://github.com/dw2761/DL_Project1/blob/main/img_1.png)


To load the model file of this project, please use the following code:
```
use model.load_state_dict(torch.load(PATH))
model.eval()
```
