https://colab.research.google.com/drive/1Z5CEk6Hi81rqDXJeW_RACJDc87xzEWkN?usp=sharing

This code contains several components that tackle different tasks. 

Plotting Function:
python
Copy code
import matplotlib.pyplot as plt
def plot(x,title=None):
    ...
    plt.show()
This is a helper function to plot an image tensor. It processes the tensor to ensure it's in the correct format and dimension for displaying, and then it plots the image using matplotlib.

Downloading the Dataset:
python
Copy code
!wget ...
!wget ...
!unzip 'flower_data.zip'
This section downloads the Oxford-102 Flower dataset and unzips it to be used later in the code.

Data Loading and Transformation:
python
Copy code
import torch
from torchvision import datasets, transforms
...
print(f"Images tensor shape: {images.shape}")
print(f"Labels tensor shape: {labels.shape}")
This section sets up the necessary transformations for image preprocessing and then loads the training set of the flower dataset using ImageFolder. It also loads the entire dataset into memory (using a batch size equal to the length of the dataset) for demonstration purposes and then extracts a batch of images and labels. This approach is not memory-efficient and would not be suitable for large datasets.

Display an Image:
python
Copy code
i = 50
plot(images[i], dataset_labels[i]);
This part displays the 50th image from the loaded dataset using the earlier defined plot function.

Model Setup and Image Classification with AlexNet:
python
Copy code
import torch
from torchvision import models, transforms
...
print('Predicted class:', labels[class_idx.item()])
This segment sets up a pretrained AlexNet model, defines image preprocessing steps, and uses the model to predict the class of the aforementioned 50th image. It then prints out the predicted class.

Extract Weights of AlexNet:
python
Copy code
w0 = alexnet.features[0].weight.data
...
img_t.shape, w0.shape
This section extracts the weights from various layers of the AlexNet model. It also checks the shape of the image tensor and the first set of weights (w0).

Visualizing the Image Tensor:
python
Copy code
def scale(img):
    ...
def tensor_plot(img_t, index=0):
    ...
tensor_plot(img_t)
This section provides two helper functions: scale normalizes a numpy array to [0, 1], and tensor_plot displays an image tensor. It then visualizes the earlier processed image tensor.

Feature Map Visualization:
python
Copy code
f0 = F.conv2d(img_t, w0, stride=4, padding=2)
...
plot_feature_maps_with_filters(f0, w0)
This part involves computing the feature maps of the image by applying a convolution operation using the weights of the first layer of the AlexNet model (w0). The resultant feature maps are then visualized alongside their corresponding filters using the plot_feature_maps_with_filters function.

Overall, this code is an exploration of the Oxford-102 Flower dataset using PyTorch. It focuses on data preprocessing, visualization, and employing a pretrained AlexNet model for image classification. The latter sections delve into a more detailed inspection of the model by visualizing its filters and the resultant feature maps when applied to an image.




