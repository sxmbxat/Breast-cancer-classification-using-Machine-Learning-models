# Breast cancer classification using Machine Learning models
This paper seeks to create models to classify breast tissues as either cancerous or non-cancerous. Two models were developed for this project. 

1.0 Introduction

Breast cancer is the most frequent malignant disease in women (Harbeck, et al., 2019), affecting so many worldwide that it is now the most common cancer in women (Key, et al., 2001). When it is discovered early, breast cancer is curable in 70 to 80% of women with early-stage, non-metastatic versions of the disease. On the other hand, advanced breast cancer is considered incurable with the current available treatments.
Childbearing reduces the risk of cancer in women, and breastfeeding can offer some level of protection. 

On the other hand, oral contraceptives, hormonal therapy, and alcohol reduces risk. In more developed countries, the incidence rates are higher, but fast increasing in less developed countries where rates were previously low (Key, et al., 2001).
Breast cancer is preventable where it is quickly detected through simple scans to enable healthcare practitioners to speedily diagnose the disease. If a model to accurately differentiate malignant from benign cancers is developed, then the world will be well on their way to defeating the scourge that is breast cancer.

Machine learning has spread far and wide, and is being used in a number of research areas including text mining, spam detection, video recommendation, image classification, and multimedia concept retrieval. Deep learning, which is a subset of machine learning, is inspired by human thinking patterns (Alzubaidi, et al., 2021).

This paper will develop two deep learning models for image classification, differentiating malignant cancers from benign growths, comparing the two models, and presenting the better of the two models. It will also use the machine learning knowledge and procedures to attempt to correctly solve this problem.

2.0 Breast cancer classification

Image classification is the process of predicting the class membership of an unknown image based on the class membership of the images that are known. This is a binary classification problem. Two deep learning models: the neural network architecture, CNN, as well the pretrained model, Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG-19), will be used for this problem. The accuracy for the two models is somewhat low, but efforts were made to finetune the models for greater accuracy.
The field of machine learning has evolved quite a bit in recent times with the rise of Artificial Neural Networks (ANN). These neural networks are classed under the deep
learning category and are able to produce better models and greater accuracy than regular machine learning models. ANN consists of three layers including the input layer, middle layers and the output layer (Mohseni-Dargah, et al., 2022).

Deep learning models are considered to be one of the most powerful tools and is widely used because of their ability to handle a large amount of data. To build a solution for this problem, two models will be created, explored, preprocessed, built, evaluated, finetuned, and visualized.
The ROC and AUC curve will be used to analyze the models because the two classes are equally distributed. The ROC-AUC curve is used to measure the performance of a deep learning model. The ROC curve explains the probability rate and is plotted by comparing the true positive and false positive rates. On the other hand, the AUC value explains how easy it is for the model to distinguish between classes—the higher the AUC value, the better.

2.1 CNN

Convolutional Neural Networks (CNN) are the most famous and commonly used neural networks because of its most important feature: their ability to identify relevant features with minimal human supervision. The major benefits of CNN are equivalent representations, sparse interactions, and parameter sharing. A CNN architecture for image classification is illustrated below.



Figure 1: CNN architecture (Alzubaidi, et al., 2021)

There are four layers in the CNN model: the convolution layer, ReLU layer, pooling layer, and the fully connected layer. CNN works to improve generalization and avoid overfitting. It also concurrently learns the feature extraction and classification layers, which helps the model to be highly organized.

2.1.1 Image classification with CNN

Prior to the model building, the data was loaded into the Jupyter environment with numpy. The shape was found to be (5547, 50, 50, 3) for the training file and 5547 for the testing data since it contains no target values. The image was loaded as an array since that’s the form that it was originally in. The data was then converted to image form for ease of processing.
The data is in two forms: IDC positive and negative. IDC stands for invasive ductal carcinoma. The images are classified as positive where IDC is present and negative when IDC is absent. There are 5547 data points in the dataset and they are evenly distributed with 2759 negative images and 2788 positive images. Following that, a plot of both classes shows that there are differences between the two: deeper colors in the positive class. The positive class also appears to be denser than the negative class. The histograms below further prove this information to be true.
After, one-hot encoding is done to make the data more useful and scalable. One-hot encoding means converting integer data to binary values: 1 and 0. Afterwards, preprocessing is complete and the model is built.

The CNN model used for this project has 8 layers: two Conv2D layers, one MaxPooling2D layer, two Dense layers, two Dropout layers, and one Flatten layer. The model is then fit and Early Stopping is applied to improve the accuracy of the model.
After building the model, a confusion matrix is plotted to analyze the model. Since it was previously established that the dataset is evenly distributed, an ROC curve is used to analyze the results and check for the AUC. An ROC (receiver operating characteristic) curve is a graph showing the performance of a classification model while the AUC is the area under the curve.
The CNN performed relatively well with an accuracy of 65% with a precision, recall, and F1 score of 66% across the board for the negative IDC class. For the positive IDC, the precision, recall, and F1 score was 65%, 64%, and 65% respectively.


Figure 2: Accuracy of the CNN model

The AUC under the ROC for both classes is 0.72. This indicates that the model performs relatively well, but there is room for improvement.

Figures 3 & 4: ROC curve for classes in CNN 2.2 VGG-19

The Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG-19) is one of the most popular pre-trained models for image classification. It was developed at the Visual Geometry Group at the University of Oxford and was trained on over a million images from the imageNet database. The architecture of VGG-19 is illustrated in the figure below.

Figure 5: VGG-19 architecture

The model has 16 convolution layers, 3 Fully connected layers, 5 maxpool layers and 1 SoftMax layer. The model uses a lot of filters and is sequential in nature. Despite how powerful the VGG-19 model is, it is a very slow model and is larger to train than others. Nonetheless, there are several models similar to the VGG-19 model, like the VGG-19 model. The VGG-19 model was selected because of its high accuracy and faster speed.

2.2.1 Image classification with VGG-19

To solve this image classification problem, the VGG-19 was used with more impressive results this time. For this model, the dimension is adjusted and the number of images is reduced to 4437 while the image size is increased to 64, 64, 3. After, the data is normalized and reshaped to make it suitable for the pre-trained model.
Afterwards, the dataset is one-hot encoded, resized, and passed into the model. The VGG-19 model generated over 20 million parameters which contributed to the dismal speed at which the model processed the epochs. The VGG-19 model returned an accuracy of 68% with a precision, recall, and F1 score of 72%, 60%, and 66% respectively for the negative IDC. For the positive IDC, the precision, recall, and F1 score were 64%, 75%, and 69% respectively.

Figure 6: Accuracy for VGG-19 model

The model did very well on recall, but not so much with precision. Afterwards, learning curves were used to interpret the loss and accuracy, as well as an ROC-AUC curve.

Figure 7: ROC curve for classes in VGG-19

The graphs for the learning curves are illustrated in figure 8.

Figure 8: learning curves showing loss and accuracy for VGG-19

One observation is that the AUC value for the classes in this dataset is very high at 0.75 and 0.75. This means that the VGG-19 model is particularly good at distinguishing between the two classes, especially for positive IDC, and in relation with the CNN model.

2.3 Model comparison

How do the two models compare against each other? At a glance, VGG-19 appears to perform better with a slightly higher accuracy of 68% and higher precision and recall rates. Additionally, the ROC curves for both models are similar, indicating that the two models do very well with differentiating between different classes.
One major difference between the two models is the amount of time that their execution takes. This can probably be attributed to their parameters: CNN has 4.3 million parameters while VGG-19 has over 20 million parameters. Of these, there are 279,042 trainable parameters which the model must adjust as part of the gradient. Because of this, VGG-19 takes over 2 minutes each to cycle through each epoch where the CNN model takes less than a minute to process each epoch.

3.0 Conclusion

Breast cancer image classification is an important problem with real world applications. In this paper, deep learning models that classifies the images based on whether they contain malignant or benign breast tissues were built. The accuracy obtained from both models in this case was quite low at 63% and 67% percent with CNN and VGG-19 models, suggesting that other approaches should be made when working with the models. Some approaches that can be employed include:

• Using other deep learning models

• Finetuning the models further in order to improve the accuracy.
