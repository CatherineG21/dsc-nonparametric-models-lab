# Nonparametric ML Models - Cumulative Lab

## Introduction

In this cumulative lab, you will apply two nonparametric models you have just learned — k-nearest neighbors and decision trees — to the forest cover dataset.

## Objectives

* Practice identifying and applying appropriate preprocessing steps
* Perform an iterative modeling process, starting from a baseline model
* Explore multiple model algorithms, and tune their hyperparameters
* Practice choosing a final model across multiple model algorithms and evaluating its performance

## Your Task: Complete an End-to-End ML Process with Nonparametric Models on the Forest Cover Dataset

![line of pine trees](https://curriculum-content.s3.amazonaws.com/data-science/images/trees.jpg)

Photo by <a href="https://unsplash.com/@michaelbenz?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Michael Benz</a> on <a href="/s/photos/forest?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>

### Business and Data Understanding

To repeat the previous description:

> Here we will be using an adapted version of the forest cover dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/covertype). Each record represents a 30 x 30 meter cell of land within Roosevelt National Forest in northern Colorado, which has been labeled as `Cover_Type` 1 for "Cottonwood/Willow" and `Cover_Type` 0 for "Ponderosa Pine". (The original dataset contained 7 cover types but we have simplified it.)

The task is to predict the `Cover_Type` based on the available cartographic variables:


```python
# Run this cell without changes
import pandas as pd

df = pd.read_csv('data/forest_cover.csv')
df
```

> As you can see, we have over 38,000 rows, each with 52 feature columns and 1 target column:

> * `Elevation`: Elevation in meters
> * `Aspect`: Aspect in degrees azimuth
> * `Slope`: Slope in degrees
> * `Horizontal_Distance_To_Hydrology`: Horizontal dist to nearest surface water features in meters
> * `Vertical_Distance_To_Hydrology`: Vertical dist to nearest surface water features in meters
> * `Horizontal_Distance_To_Roadways`: Horizontal dist to nearest roadway in meters
> * `Hillshade_9am`: Hillshade index at 9am, summer solstice
> * `Hillshade_Noon`: Hillshade index at noon, summer solstice
> * `Hillshade_3pm`: Hillshade index at 3pm, summer solstice
> * `Horizontal_Distance_To_Fire_Points`: Horizontal dist to nearest wildfire ignition points, meters
> * `Wilderness_Area_x`: Wilderness area designation (3 columns)
> * `Soil_Type_x`: Soil Type designation (39 columns)
> * `Cover_Type`: 1 for cottonwood/willow, 0 for ponderosa pine

This is also an imbalanced dataset, since cottonwood/willow trees are relatively rare in this forest:


```python
# Run this cell without changes
print("Raw Counts")
print(df["Cover_Type"].value_counts())
print()
print("Percentages")
print(df["Cover_Type"].value_counts(normalize=True))
```

Thus, a baseline model that always chose the majority class would have an accuracy of over 92%. Therefore we will want to report additional metrics at the end.

### Previous Best Model

In a previous lab, we used SMOTE to create additional synthetic data, then tuned the hyperparameters of a logistic regression model to get the following final model metrics:

* **Log loss:** 0.13031294393913376
* **Accuracy:** 0.9456679825472678
* **Precision:** 0.6659919028340081
* **Recall:** 0.47889374090247455

In this lab, you will try to beat those scores using more-complex, nonparametric models.

### Modeling

Although you may be aware of some additional model algorithms available from scikit-learn, for this lab you will be focusing on two of them: k-nearest neighbors and decision trees. Here are some reminders about these models:

#### kNN - [documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

This algorithm — unlike linear models or tree-based models — does not emphasize learning the relationship between the features and the target. Instead, for a given test record, it finds the most similar records in the training set and returns an average of their target values.

* **Training speed:** Fast. In theory it's just saving the training data for later, although the scikit-learn implementation has some additional logic "under the hood" to make prediction faster.
* **Prediction speed:** Very slow. The model has to look at every record in the training set to find the k closest to the new record.
* **Requires scaling:** Yes. The algorithm to find the nearest records is distance-based, so it matters that distances are all on the same scale.
* **Key hyperparameters:** `n_neighbors` (how many nearest neighbors to find; too few neighbors leads to overfitting, too many leads to underfitting), `p` and `metric` (what kind of distance to use in defining "nearest" neighbors)

#### Decision Trees - [documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)

Similar to linear models (and unlike kNN), this algorithm emphasizes learning the relationship between the features and the target. However, unlike a linear model that tries to find linear relationships between each of the features and the target, decision trees look for ways to split the data based on features to decrease the entropy of the target in each split.

* **Training speed:** Slow. The model is considering splits based on as many as all of the available features, and it can split on the same feature multiple times. This requires exponential computational time that increases based on the number of columns as well as the number of rows.
* **Prediction speed:** Medium fast. Producing a prediction with a decision tree means applying several conditional statements, which is slower than something like logistic regression but faster than kNN.
* **Requires scaling:** No. This model is not distance-based. You also can use a `LabelEncoder` rather than `OneHotEncoder` for categorical data, since this algorithm doesn't necessarily assume that the distance between `1` and `2` is the same as the distance between `2` and `3`.
* **Key hyperparameters:** Many features relating to "pruning" the tree. By default they are set so the tree can overfit, and by setting them higher or lower (depending on the hyperparameter) you can reduce overfitting, but too much will lead to underfitting. These are: `max_depth`, `min_samples_split`, `min_samples_leaf`, `min_weight_fraction_leaf`, `max_features`, `max_leaf_nodes`, and `min_impurity_decrease`. You can also try changing the `criterion` to "entropy" or the `splitter` to "random" if you want to change the splitting logic.

### Requirements

#### 1. Prepare the Data for Modeling

#### 2. Build a Baseline kNN Model

#### 3. Build Iterative Models to Find the Best kNN Model

#### 4. Build a Baseline Decision Tree Model

#### 5. Build Iterative Models to Find the Best Decision Tree Model

#### 6. Choose and Evaluate an Overall Best Model

## 1. Prepare the Data for Modeling

The target is `Cover_Type`. In the cell below, split `df` into `X` and `y`, then perform a train-test split with `random_state=42` and `stratify=y` to create variables with the standard `X_train`, `X_test`, `y_train`, `y_test` names.

Include the relevant imports as you go.


```python
# Import the relevant function
from sklearn.model_selection import train_test_split

# Split df into X and y
X = df.drop("Cover_Type", axis=1)
y = df["Cover_Type"]

# Perform train-test split with random_state=42 and stratify=y
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
```

Now, instantiate a `StandardScaler`, fit it on `X_train`, and create new variables `X_train_scaled` and `X_test_scaled` containing values transformed with the scaler.


```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)```

The following code checks that everything is set up correctly:


```python
# Run this cell without changes

# Checking that df was separated into correct X and y
assert type(X) == pd.DataFrame and X.shape == (38501, 52)
assert type(y) == pd.Series and y.shape == (38501,)

# Checking the train-test split
assert type(X_train) == pd.DataFrame and X_train.shape == (28875, 52)
assert type(X_test) == pd.DataFrame and X_test.shape == (9626, 52)
assert type(y_train) == pd.Series and y_train.shape == (28875,)
assert type(y_test) == pd.Series and y_test.shape == (9626,)

# Checking the scaling
assert X_train_scaled.shape == X_train.shape
assert round(X_train_scaled[0][0], 3) == -0.636
assert X_test_scaled.shape == X_test.shape
assert round(X_test_scaled[0][0], 3) == -1.370
```

## 2. Build a Baseline kNN Model

Build a scikit-learn kNN model with default hyperparameters. Then use `cross_val_score` with `scoring="neg_log_loss"` to find the mean log loss for this model (passing in `X_train_scaled` and `y_train` to `cross_val_score`). You'll need to find the mean of the cross-validated scores, and negate the value (either put a `-` at the beginning or multiply by `-1`) so that your answer is a log loss rather than a negative log loss.

Call the resulting score `knn_baseline_log_loss`.

Your code might take a minute or more to run.


```python
# Replace None with appropriate code

# Relevant imports
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# Creating the model
knn_baseline_model = KNeighborsClassifier()

# Perform cross-validation
knn_baseline_log_loss = -cross_val_score(knn_baseline_model, X_train_scaled, y_train, scoring="neg_log_loss").mean()
knn_baseline_log_loss
```

Our best logistic regression model had a log loss of 0.13031294393913376

Is this model better? Compare it in terms of metrics and speed.


```python
# Replace None with appropriate text
"""
•kNN achieved a slightly better log loss than the tuned logistic regression model.
•kNN was much slower, while logistic regression ran faster and scaled better.
•Choose logistic regression for speed; choose kNN if accuracy is the top priority.
"""
```

## 3. Build Iterative Models to Find the Best kNN Model

Build and evaluate at least two more kNN models to find the best one. Explain why you are changing the hyperparameters you are changing as you go. These models will be *slow* to run, so be thinking about what you might try next as you run them.


```python
The default setting of 5 neighbors may be causing overfitting given the large dataset size. To address this, I will increase the number of neighbors tenfold and check whether performance improves.    
```


```python
"""
It looks good. I'll keep the same number of neighbors and change the distance metric from Euclidean to Manhattan to see how it performs.
"""

knn_third_model = KNeighborsClassifier(n_neighbors=50, metric="manhattan")

knn_third_log_loss = -cross_val_score(knn_third_model, X_train_scaled, y_train, scoring="neg_log_loss").mean()
knn_third_log_loss
```


```python
"""
That's a small improvement, but the gain is much smaller this time. I’ll try increasing the number of neighbors again to see if it improves performance further.
"""

knn_fourth_model = KNeighborsClassifier(n_neighbors=75, metric="manhattan")

knn_fourth_log_loss = -cross_val_score(knn_fourth_model, X_train_scaled, y_train, scoring="neg_log_loss").mean()
knn_fourth_log_loss
```
```python
"""
Although this result is still better than the default of 5 neighbors, it performs worse than when I used 50 neighbors. If I continued experimenting, I would explore values between 5 and 50 to find the optimal setting. For now, I'll stop here and conclude that knn_third_model is the best-performing model, with a log loss of approximately 0.0859.
"""
```
## 4. Build a Baseline Decision Tree Model

Now that you have chosen your best kNN model, start investigating decision tree models. First, build and evaluate a baseline decision tree model, using default hyperparameters (with the exception of `random_state=42` for reproducibility).

(Use cross-validated log loss, just like with the previous models.)


```python
from sklearn.tree import DecisionTreeClassifier

dtree_baseline_model = DecisionTreeClassifier(random_state=42)

dtree_baseline_log_loss = -cross_val_score(dtree_baseline_model, X_train, y_train, scoring="neg_log_loss").mean()
dtree_baseline_log_loss```

Interpret this score. How does this compare to the log loss from our best logistic regression and best kNN models? Any guesses about why?


```python
# Replace None with appropriate text
"""
This performance is worse than both the logistic regression and kNN models. It is likely that the decision tree is heavily overfitting, since it has not been pruned or regularized in any way. Decision trees can easily memorize the training data and make overly confident predictions, especially when no limits are placed on tree depth or complexity. As a result, they may perform well on the training set but generalize poorly to unseen data, which leads to a much higher log loss in cross-validation.
"""
```

## 5. Build Iterative Models to Find the Best Decision Tree Model

Build and evaluate at least two more decision tree models to find the best one. Explain why you are changing the hyperparameters you are changing as you go.


```python
To reduce overfitting in the decision tree, I will adjust key hyperparameters that control model complexity. I will begin by increasing min_samples_leaf, which ensures each leaf has more samples before making a split. This restricts the tree from growing too deep and helps improve generalization. Conceptually, this is similar to increasing
the number of neighbors in kNN, since both approaches require more evidence before making a prediction, although they use different mechanisms to determine similarity.
"""

dtree_second_model = DecisionTreeClassifier(random_state=42, min_samples_leaf=10)

dtree_second_log_loss = -cross_val_score(dtree_second_model, X_train, y_train, scoring="neg_log_loss").mean()
dtree_second_log_loss
```

```python
"""
Great, meaning, raising min_samples_leaf from 1 to 10 improved generalization. Let me try pushing this value even higher to see if performance improves further.
"""

dtree_third_model = DecisionTreeClassifier(random_state=42, min_samples_leaf=100)

dtree_third_log_loss = -cross_val_score(dtree_third_model, X_train, y_train, scoring="neg_log_loss").mean()
dtree_third_log_loss```


```python
"""
Now my scores are in the same range as the best logistic regression model and the baseline kNN model. However, I noticed that this dataset is very imbalanced, and I haven’t accounted for that yet. So next, I will keep the same min_samples_leaf value but also apply class weighting to handle the imbalance.
"""

dtree_fourth_model = DecisionTreeClassifier(random_state=42, min_samples_leaf=100, class_weight="balanced")

dtree_fourth_log_loss = -cross_val_score(dtree_fourth_model, X_train, y_train, scoring="neg_log_loss").mean()
dtree_fourth_log_loss
```
```python
"""
That didn't work as expected — it looks like the model may have over-adjusted when I tried to balance the classes, 
so I’ll skip the class_weight parameter for now. I also notice that this dataset has many features, so next I’ll try limiting the number of features the tree can consider at each split, while keeping min_samples_leaf the same.
"""

dtree_fifth_model = DecisionTreeClassifier(random_state=42, min_samples_leaf=100, max_features="sqrt")

dtree_fifth_log_loss = -cross_val_score(dtree_fifth_model, X_train, y_train, scoring="neg_log_loss").mean()
dtree_fifth_log_loss
```
```python
"""
This still isn’t performing better than my dtree_third_model. I’ll test one more value for min_samples_leaf to see if it improves the results.
"""

dtree_sixth_model = DecisionTreeClassifier(random_state=42, min_samples_leaf=75)

dtree_sixth_log_loss = -cross_val_score(dtree_sixth_model, X_train, y_train, scoring="neg_log_loss").mean()
dtree_sixth_log_loss
```
```python
"""
That result looks promising. In a more detailed workflow I might run a proper grid search, but for now, I’ll treat the sixth model as the best-performing decision tree model.
"""
```
## 6. Choose and Evaluate an Overall Best Model

Which model had the best performance? What type of model was it?

Instantiate a variable `final_model` using your best model with the best hyperparameters.

```python
"""
This model runs significantly slower than the best logistic regression and decision tree models, but in this case the performance gain seems worth it. However, depending on the business objectives, the extra time may not always be justified.
"""
```

```python
final_model = KNeighborsClassifier(n_neighbors=50, metric="manhattan")

# Fit the model on the full training data
# (scaled or unscaled depending on the model)
final_model.fit(X_train_scaled, y_train)
```

Now, evaluate the log loss, accuracy, precision, and recall. This code is mostly filled in for you, but you need to replace `None` with either `X_test` or `X_test_scaled` depending on the model you chose.


```python
# Replace None with appropriate code
from sklearn.metrics import accuracy_score, precision_score, recall_score

preds = final_model.predict(X_test_scaled)
probs = final_model.predict_proba(X_test_scaled)

print("log loss: ", log_loss(y_test, probs))
print("accuracy: ", accuracy_score(y_test, preds))
print("precision:", precision_score(y_test, preds))
print("recall:   ", recall_score(y_test, preds))
```

Interpret your model performance. How would it perform on different kinds of tasks? How much better is it than a "dummy" model that always chooses the majority class, or the logistic regression described at the start of the lab?


```python
# Replace None with appropriate text
"""
This model achieves 97% accuracy, outperforming a simple baseline model (~92%). Its precision for class 1 is about 89%, noticeably higher than logistic regression (~67%). 
Recall also improves to around 69% versus 48% for logistic regression, though it's still not ideal. If avoiding false negatives (missing real class 1 areas) is a priority, adjusting the decision threshold may be necessary.

"""
```

## Conclusion

In this lab, you practiced the end-to-end machine learning process with multiple model algorithms, including tuning the hyperparameters for those different algorithms. You saw how nonparametric models can be more flexible than linear models, potentially leading to overfitting but also potentially reducing underfitting by being able to learn non-linear relationships between variables. You also likely saw how there can be a tradeoff between speed and performance, with good metrics correlating with slow speeds.
