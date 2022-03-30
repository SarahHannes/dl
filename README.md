# Deep Learning

- [Feed Forward Neural Network](https://github.com/SarahHannes/dl#feed-forward-neural-network)
  * [Categorial](https://github.com/SarahHannes/dl#categorical)
    + [1.0 Titanic Dataset](https://github.com/SarahHannes/dl#10-titanic-dataset-code)
    + [2.0 Women Chess Dataset](https://github.com/SarahHannes/dl#20-women-chess-dataset-code)
    + [3.0 German Credit Dataset: Overfitted Model](https://github.com/SarahHannes/dl#30-german-credit-dataset-overfitted-model-code)
    + [4.0 German Credit Dataset: Generalized Model](https://github.com/SarahHannes/dl#40-german-credit-dataset-generalized-model-code)
- [Convolutional Neural Network](https://github.com/SarahHannes/dl#convolutional-neural-network)
  * [1.0 CIFAR10](https://github.com/SarahHannes/dl#10-cifar10-code)
  * [2.0 Bread](https://github.com/SarahHannes/dl#20-bread-code)

<!-- toc -->
Feed Forward Neural Network
------------
### Categorical
#### 1.0 Titanic Dataset <a href="feedforward/10_categorical_Titanic.py">[Code]</a>
<img src="feedforward/plots/10_loss.png" width="350"> <img src="feedforward/plots/10_accuracy.png" width="345">

#### 2.0 Women Chess Dataset <a href="feedforward/20_categorical_WomenChess.py">[Code]</a>
<img src="feedforward/plots/20_loss.png" width="345"> <img src="feedforward/plots/20_accuracy.png" width="345">

#### 3.0 German Credit Dataset: Overfitted Model <a href="feedforward/30_categorical_GermanCredit.py">[Code]</a>
> Learning goal: 
> 1. Prepare data and NN model is such a way that it will prone to overfit.
> 	 - One hot encoding all categorical columns to increase the number of feature columns.
> 	 - Increase model complexity (more dense layer, more nodes in each layers).
> 	 - Train with high epochs.
```
Model: "german_credit_model_overfit"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 61)]              0         
                                                                 
 dense (Dense)               (None, 64)                3968      
                                                                 
 dense_1 (Dense)             (None, 32)                2080      
                                                                 
 dense_2 (Dense)             (None, 16)                528       
                                                                 
 dense_3 (Dense)             (None, 8)                 136       
                                                                 
 dense_4 (Dense)             (None, 2)                 18        
                                                                 
=================================================================
Total params: 6,730
Trainable params: 6,730
Non-trainable params: 0
_________________________________________________________________
```
<img src="feedforward/plots/30_loss.png" width="340"> <img src="feedforward/plots/30_accuracy.png" width="345">
<!-- 
#### 4.0 German Credit Dataset: Generalized Model <a href="feedforward/40_functional_categorical_GermanCredit.py">[Code]</a>
> Learning goal:
> 
> 2.  Reduce overfitting in the previous model.
>     -   Reduce model complexity.
```
```
<img src="feedforward/plots/40_loss.png" width="345"> <img src="feedforward/plots/40_accuracy.png" width="345">
 -->
 Convolutional Neural Network
------------
 #### 1.0 CIFAR10 <a href="cnn/10_cifar10.ipynb"> [Code]</a>
```
Model: "Model4"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_12 (Conv2D)          (None, 32, 32, 64)        1792      
                                                                 
 conv2d_13 (Conv2D)          (None, 32, 32, 64)        36928     
                                                                 
 max_pooling2d_7 (MaxPooling  (None, 16, 16, 64)       0         
 2D)                                                             
                                                                 
 conv2d_14 (Conv2D)          (None, 16, 16, 32)        18464     
                                                                 
 conv2d_15 (Conv2D)          (None, 16, 16, 32)        9248      
                                                                 
 max_pooling2d_8 (MaxPooling  (None, 8, 8, 32)         0         
 2D)                                                             
                                                                 
 conv2d_16 (Conv2D)          (None, 8, 8, 16)          4624      
                                                                 
 conv2d_17 (Conv2D)          (None, 8, 8, 16)          2320      
                                                                 
 max_pooling2d_9 (MaxPooling  (None, 4, 4, 16)         0         
 2D)                                                             
                                                                 
 flatten_3 (Flatten)         (None, 256)               0         
                                                                 
 dense_9 (Dense)             (None, 32)                8224      
                                                                 
 dropout (Dropout)           (None, 32)                0         
                                                                 
 dense_10 (Dense)            (None, 10)                330       
                                                                 
=================================================================
Total params: 81,930
Trainable params: 81,930
Non-trainable params: 0
_________________________________________________________________
```

 #### 2.0 Bread <a href="cnn/20_bread.ipynb"> [Code]</a>
> Learning goal:
> 
> Create, load dataset and build CNN model for "good" and "moldy" bread image classification.

<img src="https://user-images.githubusercontent.com/78901374/160868266-3855f094-db24-4f05-92f2-abf434ef1ab8.png" width="350" height="350">
