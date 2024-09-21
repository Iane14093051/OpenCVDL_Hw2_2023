
# Course Assignments

[Assignment Requirements Slide](https://github.com/hsylin/OpenCVDL_Hw2/raw/refs/heads/main/OpenCv_Hw2_Q_20231205_V1B3.pptx)

## Environment

- OS: Windows 10
- Python Version: 3.8

## Setup Instructions

1. Clone the repository:
   ```bash
   $ git clone https://github.com/Iane14093051/OpenCVDL_Hw2_2023.git
   ```
2. Install the required dependencies:
   ```bash
   $ pip install -r requirements.txt
   ```
3. Train the ResNet50:
   ```bash
   $ python model_ResNet50_lee.py
   $ python model_ResNet50_no_lee.py
   ```
4. Generate the compare result of ResNet50:
   ```bash
   $ python model_ResNet50_compare_lee.py
   ```    
5. Run the application:
   ```bash
   $ python hw2.0.py
   ```

   
## Application Features

Once the application (`hw2.0.py`) is running, the UI is divided into five sections: **Hough Circle Transformation**, **Histogram Equalization**, **Morphology Operation**, **VGG19 MNIST Classifier**, and **ResNet50 Cat-Dog Classifier**.

### 1. Hough Circle Transformation
Offers two features:
- **Draw Contour**: Detect and draw the contours of circles in the loaded image.
- **Count Coins**: Detect the circle center points using the HoughCircles function and display the total number of coins present in the image.

### 2. Histogram Equalization
Provides one feature:
- **Apply Histogram Equalization**: Perform two types of histogram equalizations on the loaded image:
  1. OpenCV `cv2.equalizeHist()` method.
  2. A manual method using PDF and CDF.

### 3. Morphology Operation
Offers two features:
- **Closing Operation**: Manually implement the closing operation by applying dilation followed by erosion to fill small holes in the image.
- **Opening Operation**: Manually implement the opening operation by applying erosion followed by dilation to remove small objects or noise from the image.

### 4. VGG19 MNIST Classifier with Batch Normalization
Deep learning features using the VGG19 model:
- **Show Model Structure**: Display the architecture of VGG19 with Batch Normalization.
- **Show Training/Validation Accuracy and Loss**: Show a plot of training and validation accuracy/loss.
- **Inference**: Draw a digit on the canvas and predict its class using the trained model. Also, display the probability distribution of the prediction.
- **Reset Canvas**: Clear the canvas for a new drawing.

### 5. ResNet50 Cat-Dog Classifier
Offers four features for classifying images of cats and dogs:
- **Show Random Images**: Display random images from the `inference_dataset/Cat` and `inference_dataset/Dog` directories.
- **Show ResNet50 Model Structure**: Display the architecture of the ResNet50 model.
- **Random Erasing**: Compare classification accuracy with and without random erasing as a data augmentation technique.
- **Inference**: Load an image, classify it using the trained model, and display the predicted class label.
