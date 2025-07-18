# Machine-Learning-Concepts
Here's a Machine Learning Topics Tree organized in a hierarchical structure so you can clearly see which topic belongs where and we gonna cover them all.

```bash
Machine Learning
├── 1. Types of Learning
│   ├── 1.1 Supervised Learning
│   │   ├── Regression
│   │   │   ├── Linear Regression 
│   │   │   ├── Polynomial Regression 
│   │   │   ├── Ridge / Lasso Regression
│   │   ├── Classification
│   │   │   ├── Logistic Regression 
│   │   │   ├── K-Nearest Neighbors (KNN) 
│   │   │   ├── Support Vector Machine (SVM) 
│   │   │   ├── Decision Trees 
│   │   │   ├── Naive Bayes 
│
│   ├── 1.2 Unsupervised Learning 
│   │   ├── Clustering
│   │   │   ├── K-Means 
│   │   │   ├── Hierarchical Clustering 
│   │   ├── Dimensionality Reduction
│   │   │   ├── PCA (Principal Component Analysis) 
│   │   │   ├── t-SNE 
│
│   ├── 1.3 Reinforcement Learning 
│       ├── Q-Learning
│       ├── Deep Q-Networks (DQN)

# Math Prerequisite

├── 2. Math Foundations
│   ├── Linear Algebra 🔜
│   ├── Calculus (Gradient Descent) 
│   ├── Probability & Statistics 
│   ├── Optimization
│       ├── Cost Functions
│       │   ├── MSE 
│       │   ├── Cross-Entropy Loss 
│       ├── Gradient Descent 

# Training and Preprocessing

├── 3. Model Development
│   ├── Data Splitting
│   │   ├── Train/Test Split 
│   │   ├── Cross-Validation 
│   ├── Data Preprocessing
│   │   ├── Scaling / Normalization 
│   │   ├── Encoding Categorical Data 
│   │   ├── Handling Missing Values 

# Model Evaluation

├── 4. Model Evaluation
│   ├── Regression Metrics
│   │   ├── Mean Squared Error (MSE) 
│   │   ├── R² Score 
│   ├── Classification Metrics
│   │   ├── Accuracy 
│   │   ├── Precision 
│   │   ├── Recall 
│   │   ├── F1 Score 
│   │   ├── Confusion Matrix

```
While learning about all this machine learning concepts and models, I find myself hard to remember that how i imported certain model last time, so here is the tree to remember the sklearn models. Here's a well-organized Tree View of sklearn's models and where they live inside the library, so you can see which models belong to which category and module (e.g., classification, regression, clustering, etc.).

```bash
sklearn
├── model_selection
│   ├── train_test_split 
│   ├── cross_val_score
│   ├── GridSearchCV
│   ├── StratifiedKFold
│
├── linear_model
│   ├── LinearRegression 
│   ├── LogisticRegression 
│   ├── Ridge
│   ├── Lasso 
│   ├── ElasticNet 
│   ├── SGDClassifier 
│   ├── SGDRegressor
│
├── ensemble
│   ├── RandomForestClassifier
│   ├── RandomForestRegressor
│   ├── GradientBoostingClassifier
│   ├── GradientBoostingRegressor
│   ├── AdaBoostClassifier
│   ├── AdaBoostRegressor
│   ├── BaggingClassifier
│   ├── VotingClassifier
│
├── tree
│   ├── DecisionTreeClassifier
│   ├── DecisionTreeRegressor
│
├── neighbors
│   ├── KNeighborsClassifier
│   ├── KNeighborsRegressor
│
├── svm
│   ├── SVC (Support Vector Classifier)
│   ├── SVR (Support Vector Regressor)
│   ├── LinearSVC
│   ├── NuSVC
│
├── naive_bayes
│   ├── GaussianNB
│   ├── MultinomialNB
│   ├── BernoulliNB
│
├── cluster
│   ├── KMeans
│   ├── DBSCAN 
│   ├── AgglomerativeClustering
│
├── decomposition
│   ├── PCA (Principal Component Analysis)
│   ├── TruncatedSVD
│
│
├── metrics
│   ├── accuracy_score
│   ├── mean_squared_error
│   ├── r2_score
│   ├── confusion_matrix
│   ├── precision_score
│   ├── recall_score
│   ├── f1_score
```

## What is Machine Learning?
- **Machine Learning (ML)** is a branch of **Artificial Intelligence (AI)** that focuses on building systems that can automatically learn and improve from experience without being **explicitly programmed** for every task.
- For Example have a module which classcify email to **SPAM** and **HAM**. So Instead of explicitly programming spam rules, we let the machine learn patterns from real-world email data. It then applies that knowledge to classify new emails—even if the spammer slightly changes their tricks.
- “**Machine learning** is the field of study that gives computers the ability to learn without being explicitly programmed.” — **Arthur Samuel, 1959**.

  <img width="350" height="346" alt="image" src="https://github.com/user-attachments/assets/ab777ec1-2b68-4660-850b-d63e752ab309" />


### Types of Machine Learning:
1. Supervised Learning
2. Unsupervised Learning
3. Reinforcement Learning

### Supervised Learning
- **Supervised Learning** is like learning with a teacher. The teacher not only tells you what each thing is, but also corrects you when you're wrong. In this analogy, the teacher acts as the **supervisor**.
- So, in **supervised learning**, the machine is trained on labeled data—which means each input comes with the correct output (label). The model learns by comparing its predictions to the actual answers, just like a student learns by getting feedback from the teacher.
- Supervised Learning can be broadly divided into two main types :
  1. **Regression**
  2. **Classification**
 
### Unsupervised Learning
- **Unsupervised learning** is like learning without a teacher. You're given a pile of information, but no labels or instructions about what it is or what it means. The system has to find patterns and relationships in the data all by itself.
- In this type of learning, the machine is not told what to predict, but rather it tries to make sense of the data—by grouping similar things together or reducing complexity.
- Common Types of Unsupervised Learning :
  1. **Clustering**
  2. **Dimensionality Reduction**
 
### Reinforcement Learning
- **Reinforcement Learning** is like learning through **experience and feedback—just** like training a dog or learning to play a video game.
- There’s no teacher telling you the exact answer, but you get **rewards or penalties** based on what you do.
- The machine interacts with an environment, takes actions, and learns by trial and error to **maximize the total reward over time**.
- **Examples** : AI playing chess, Go, or video games like Atari or Dota.

## Machine Learning Topics:
lets learn some important topics of machine learning, we might not cover all the topics, but will try to cover those you must know.

### Linear Regression & Polynomial Regression** :
[**Use this repo**](https://github.com/YogendraShastri/Must-Learn-Regressions-Before-Deep-Learning)

### Ridge / Lasso Regression OR (L1 & L2 regularization)
- Regularization prevents from overfitting and underfitting problem.
- L1 and L2 regularization are methods to avoid overfitting in machine learning models like linear regression.
- They add a penalty to the model to stop it from relying too much on any one feature (by keeping the weights small).

[**Under Fit Vs Best Fit Vs Over Fit**](https://medium.com/greyatom/what-is-underfitting-and-overfitting-in-machine-learning-and-how-to-deal-with-it-6803a989c76)

<img width="606" height="382" alt="image" src="https://github.com/user-attachments/assets/136b26dc-d903-4b44-ab38-cd464a4ccf23" />

#### L1 Regularization (Lasso)
- Adds absolute values of the coefficients to the loss:
- Here Original Loss is MSE (Mean Squared Error) for Regression & cross entropy for Classification.
- L1 Regularization, also known as Lasso Regularization.

<img width="250" height="86" alt="image" src="https://github.com/user-attachments/assets/a25e204b-6cc0-49e5-ad6d-e9651a135a5b" />


#### L2 Regularization (Ridge)
- Adds squared values of the coefficients to the loss.
- L2 Relularization is also known as Ridge.

<img width="250" height="84" alt="image" src="https://github.com/user-attachments/assets/bd13b63b-3709-413d-989a-b0b941f34a4b" />

**Notebook** : []()
