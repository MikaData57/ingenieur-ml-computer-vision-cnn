![baner_dogs](http://www.mf-data-science.fr/images/projects/dogs.jpg)
# Dog breed detection from images with convolutional neural networks and transfert learning

## Table of contents
* [General information](#general-info)
* [Data](#data)
* [Technologies](#technologies)
* [Setup](#setup)
* [API](#API)

## <span id="general-info">General information</span>
The objective of this Notebook is to detail the implementation of a **dog breed detection algorithm on a photo**, in order to speed up the work of indexing in a database.

### The constraints imposed :
- **Pre-process the images** with specific techniques *(e.g. Whitening, equalization, possibly modification of the size of the images)*.
- Perform **data augmentation** *(mirroring, cropping ...)*.
- Implementation of 2 approaches to the use of CNNs :
    - Create a CNN neural network from scratch by optimizing the parameters.     
    - Use the transfer learning and thus use an already trained network.
    - Fine-tuning of the pre-trained model

## <span id="data">Data</span>
The [Stanford Dogs dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) contains images of 120 breeds of dogs from around the world. This dataset has been built using images and annotation from ImageNet for the task of fine-grained image categorization. It was originally collected for fine-grain image categorization, a challenging problem as certain dog breeds have near identical features or differ in colour and age.
- Number of categories: 120
- Number of images: 20,580
- Annotations: Class labels, Bounding boxes

Aditya Khosla, Nityananda Jayadevaprakash, Bangpeng Yao and Li Fei-Fei. Novel dataset for Fine-Grained Image Categorization. First Workshop on Fine-Grained Visual Categorization (FGVC), IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2011. [pdf] [poster] [BibTex]

Secondary:
J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li and L. Fei-Fei, ImageNet: A Large-Scale Hierarchical Image Database. IEEE Computer Vision and Pattern Recognition (CVPR), 2009. [pdf] [BibTex]
	
## <span id="technologies">Technologies</span>
Project is created with:
* [Kaggle](https://www.kaggle.com/michaelfumery) Notebook
* Python 3.8
* OpenCV, PIL, KERAS ...

	
## <span id="setup">Setup</span>
Download the Notebook and import it preferably into Google Colaboratoty, Kaggle or Jupyter via Anaconda.      
Then just perform a *Run all* to run the project.

An online application is available through Heroku at the following address for testing your images on the 15 preset dog breeds in the template: https://dogs-breeds-detection-cnn.herokuapp.com/ 