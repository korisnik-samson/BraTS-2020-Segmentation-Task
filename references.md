# Reference Papers and Books
- [1] [Multimodal CNN networks for brain tumor segmentation in MRI: a BraTS 2022 challenge solution](https://link.springer.com/chapter/10.1007/978-3-031-33842-7_11) Zeineldin, Ramy A; Karar, Mohamed E; Burgert, Oliver; Mathis-Ullrich, Franziska;
- [2] [The rsna-asnr-miccai brats 2021 benchmark on brain tumor segmentation and radiogenomic classification](https://arxiv.org/abs/2107.02314) by Baid, Ujjwal; Ghodasara, Satyam; Mohan, Suyash; Bilello, Michel; Calabrese, Evan; Colak, Errol; Farahani, Keyvan; Kalpathy-Cramer, Jayashree; Kitamura, Felipe C; Pati, Sarthak; 
- [3] [nnU-Net for brain tumor segmentation](https://link.springer.com/chapter/10.1007/978-3-030-72087-2_11) Isensee, Fabian; Jäger, Paul F; Full, Peter M; Vollmuth, Philipp; Maier-Hein, Klaus H;
- [4] [Brain tumor segmentation with self-ensembled, deeply-supervised 3D U-net neural networks: a BraTS 2020 challenge solution](https://link.springer.com/chapter/10.1007/978-3-030-72084-1_30) Henry, Theophraste; Carré, Alexandre; Lerousseau, Marvin; Estienne, Théo; Robert, Charlotte; Paragios, Nikos; Deutsch, Eric; Henry, Theophraste; Carré, Alexandre; Lerousseau, Marvin; Estienne, Théo; Robert, Charlotte; Paragios, Nikos; Deutsch, Eric; 

# Improvement Notes
* Try to use a different optimizer like `AdamW` or `RMSprop`.
* Try to use a different loss function like **Dice Loss** or **Focal Loss**.
* Try using `VisionTransformer` or `SwinTransformer` from PyTorch models.
* Try using a different backbone like `ResNet`, `DenseNet`, or `EfficientNet`.
* Try without Data Augmentation.

# Reference Pages
- [Reference Paper A](C:\Users\sammi\Desktop\projects\BraTS-2020\papers\brainlesion-glioma-multiple-sclerosis-stroke-and-traumatic-brain-2021.pdf)
  -  [ ] p136: **nnU-Net for Brain Tumor Segmentation**
  - [ ] p122: **Glioma Segmentation with 3D U-Net Backed  with Energy-Based Post-Processing**

- [Reference Paper B](C:\Users\sammi\Desktop\projects\BraTS-2020\FULLTEXT01.pdf)
    - [ ] This will be the Main context and template of your paper.
    - [ ] [Medium Reference](https://medium.com/@sumitgulati59/brain-tumor-segmentation-b97de6619e04)
    - [ ] [Medium Reference](https://musstafa0804.medium.com/optimizers-in-deep-learning-7bf81fed78a0)

# INSIGHTS NO. 1
* AdamW optimizer is a variant of Adam that uses weight decay as a regularization method, which yielded better results in steep ranged epochs.
* Augmentation is a technique to artificially increase the size of the training dataset by applying transformations to the original images. However, it wasn't needed as 
  accuracy wasn't affected, but improved by 5%.

# INSIGHTS NO. 2
* The Dice Loss function is a loss function that is used to measure the similarity between two samples. It is used in segmentation tasks, where the goal is to segment an 
  image into different regions. The Dice Loss function is defined as the intersection of the two samples divided by the sum of the two samples. It is used in 
  conjunction with the Dice Coefficient, which is a measure of the similarity between two samples. The Dice Coefficient is defined as the intersection of the two 
  samples divided by the sum of the two samples. The Dice Loss function is used in conjunction with the Dice Coefficient to measure the similarity between two samples.

# INSIGHTS NO. 3
* Vision Transformers are a type of neural network architecture that is used for image classification tasks. They are based on the Transformer architecture, which was 
  originally developed for natural language processing tasks. Vision Transformers have been shown to achieve state-of-the-art performance on a wide range of image 
  classification tasks. They are able to capture long-range dependencies in images and are able to model complex patterns in images. Vision Transformers are able to 
  achieve this by using self-attention mechanisms, which allow the network to focus on different parts of the image at different scales. Vision Transformers have been 
  shown to outperform traditional convolutional neural networks on a wide range of image classification tasks.

# INSIGHTS NO. 4
* While training, it's crucial as well to take note of the optimizer in use and the learning rates. The learning rate is a hyperparameter that controls how much we are 
  adjusting the weights of our network with respect to the loss gradient. The optimizer is the algorithm that is used to update the weights of the network in order to 
  minimize the loss function. The learning rate and the optimizer are two of the most important hyperparameters that need to be tuned in order to achieve good 
  performance on a deep learning model.
  * **AdamW with a learning rate of 0.0005** was used in the training of the model, which yielded better results in steep ranged epochs (100), but will need early stopping 
    algorithms to prevent overfitting.

# ADAM OPTIMIZER VARIANTS:
{
"Adafactor",
"Adadelta",
"Adagrad",
"Adam",
"Adamax",
"AdamW",
"ASGD",
"LBFGS",
"lr_scheduler",
"NAdam",
"Optimizer",
"RAdam",
"RMSprop",
"Rprop",
"SGD",
"SparseAdam",
"swa_utils"
}

# MODEL UPLOADS: https://drive.google.com/drive/folders/1qsKw3Jdfizlf3SwmDYPowFuYPNB2LNJ6?usp=sharing