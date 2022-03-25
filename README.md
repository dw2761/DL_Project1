# DL_Project1
In this project, we studied the influences of 6 parameters and several regularization strategies on the accuracy of the model. By fine-tuning the the depth (N), the width (C), the block number in each layer(B), the convolutional kernel size (F), the skip connection kernel size (K) and average pooling kernel size (P) in the residual layers we achieved an accuracy of 90% with a model size below 5M parameters.

use model.load_state_dict(torch.load(PATH))
model.eval()

Through the tuning process, we finally obtained our model to perform an accuracy at 90.5%. The model has an architecture with 3 residual layers, and each layer has 3 residual blocks. The convolution layer behind the input layer with 2x2 kernel size (F1). Each residual block with two 3x3 convolutions. As mentioned above, a large number of channels is preferred, since we observed that channels are the most significant factor for improving accuracy. Then in our model C1 was set to 68. The detailed model parameters are shown in the table below. 
In our final model, we have selected data normalization and converting image to tensor as data augmentation strategies to reduce generalization errors. 

<img width="288" alt="image" src="https://user-images.githubusercontent.com/95495325/160050951-0966cc2d-ee00-46c5-adbe-1682b1526093.png">

Curve for Final Model
