# Deep Learning

- [Feed Forward Neural Network](https://github.com/SarahHannes/dl/blob/main/README.md#feed-forward-neural-network)
  * [Categorial](https://github.com/SarahHannes/dl/blob/main/README.md#categorical)
    + [1.0 Titanic Dataset](https://github.com/SarahHannes/dl/blob/main/README.md#10-titanic-dataset-code)
    + [2.0 Women Chess Dataset](https://github.com/SarahHannes/dl/blob/main/README.md#20-women-chess-dataset-code)
    + [3.0 German Credit Dataset: Overfitted Model](https://github.com/SarahHannes/dl/blob/main/README.md#30-german-credit-dataset-overfitted-model-code)
    + [4.0 German Credit Dataset: Generalized Model](https://github.com/SarahHannes/dl/blob/main/README.md#30-german-credit-dataset-generalized-model-code)

<!-- toc -->
Feed Forward Neural Network
------------
### Categorical
#### 1.0 Titanic Dataset <a href="feedforward/10_functional_categorical_output_Titanic.py">[Code]</a>
<img src="feedforward/plots/10_loss.png" width="350"> <img src="feedforward/plots/10_accuracy.png" width="345">

#### 2.0 Women Chess Dataset <a href="feedforward/20_functional_categorical_output_WomenChess.py">[Code]</a>
<img src="feedforward/plots/20_loss.png" width="345"> <img src="feedforward/plots/20_accuracy.png" width="345">

#### 3.0 German Credit Dataset: Overfitted Model <a href="feedforward/30_functional_categorical_output_GermanCredit.py">[Code]</a>
```
Learning goal:
    (1) Prepare data and NN model is such a way that it will prone to overfit.
        - One hot encoding all categorical columns to increase the number of feature columns.
        - Increase model complexity (more dense layer, more nodes in each layers)
        - Train with high epochs
```
<img src="feedforward/plots/30_loss.png" width="340"> <img src="feedforward/plots/30_accuracy.png" width="345">

#### 4.0 German Credit Dataset: Generalized Model <a href="feedforward/40_functional_categorical_output_GermanCredit.py">[Code]</a>
```
Learning goal:
    (2) Reduce overfitting in the previous model.
```
<img src="feedforward/plots/40_loss.png" width="345"> <img src="feedforward/plots/40_accuracy.png" width="345">
