import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------------
# Step 1: Load the Pima Indians Diabetes dataset from a URL
# ---------------------------------------------------------------------------------

# URL to the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

# Column names based on dataset documentation
columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

# Read CSV data from the URL and assign column names
data = pd.read_csv(url, header=None, names=columns)

# Separate the features (X) and the target variable (y)
X = data.drop("Outcome", axis=1)
y = data["Outcome"]


# ---------------------------------------------------------------------------------
# Step 2: Define the objective function for Optuna optimization
# ---------------------------------------------------------------------------------

def objective(trial):
    """
    Objective function for Optuna to maximize accuracy.
    This function dynamically selects a classifier and its corresponding
    hyperparameters based on the trial suggestion.
    """
    # Dynamically choose the classifier using a categorical suggestion
    classifier_name = trial.suggest_categorical("classifier", ["xgb",
                                                                # "svm",
                                                                  "nn"
                                                                  ])
    
    if classifier_name == "xgb":
        # ---------------------------------------------------------------------
        # XGBoost Classifier: Define model-specific hyperparameter search space
        # ---------------------------------------------------------------------
        n_estimators = trial.suggest_int("xgb_n_estimators", 50, 300)
        max_depth = trial.suggest_int("xgb_max_depth", 3, 10)
        learning_rate = trial.suggest_float("xgb_learning_rate", 1e-3, 1.0, log=True)
        subsample = trial.suggest_float("xgb_subsample", 0.5, 1.0)
        classifier = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            # max_iter=1000,
            # tol=1e-4,
            # use_label_encoder=False,  # Avoid warnings regarding label encoding
            eval_metric="logloss"     # Evaluation metric for binary classification
        )
    # elif classifier_name == "svm":
    #     # ---------------------------------------------------------------------
    #     # SVM Classifier: Define model-specific hyperparameter search space
    #     # ---------------------------------------------------------------------
    #     c = trial.suggest_float("svm_C", 1e-3, 1e3, log=True)
    #     kernel = trial.suggest_categorical("svm_kernel", [
    #         # "linear",
    #           "rbf"
    #           ])
    #     if kernel == "rbf":
    #         # For RBF kernel, include gamma in the search space
    #         gamma = trial.suggest_float("svm_gamma", 1e-4, 1e-1, log=True)
    #         classifier = SVC(C=c, kernel=kernel, gamma=gamma, probability=True)
    #     else:
    #         classifier = SVC(C=c, kernel=kernel, probability=True)
    elif classifier_name == "nn":
        # ---------------------------------------------------------------------
        # Neural Network (MLPClassifier): Define model-specific hyperparameter search space
        # ---------------------------------------------------------------------
        hidden_layer_sizes = trial.suggest_categorical("nn_hidden_layer_sizes", [(50), (100), (150)])
        activation = trial.suggest_categorical("nn_activation", ["relu", "tanh"])
        alpha = trial.suggest_float("nn_alpha", 1e-5, 1e-1, log=True)
        learning_rate_init = trial.suggest_float("nn_learning_rate_init", 1e-4, 1e-1, log=True)
        classifier = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=500  # Set maximum iterations to ensure convergence
        )
    
    # ---------------------------------------------------------------------------------
    # Step 3: Evaluate the classifier using 5-fold cross-validation
    # ---------------------------------------------------------------------------------
    # This leverages the dynamic search by result feature of Optuna via TPE sampler,
    # where past trial outcomes influence future parameter suggestions.
    score = cross_val_score(classifier, X, y, cv=5, scoring="accuracy").mean()
    
    return score


# ---------------------------------------------------------------------------------
# Step 4: Create an Optuna study and optimize the objective function
# ---------------------------------------------------------------------------------

# Create a study object to maximize the accuracy score
study = optuna.create_study(direction="maximize")

# Optimize the study with a specified number of trials
study.optimize(objective, n_trials=30)

# ---------------------------------------------------------------------------------
# Step 5: Display the best parameters and corresponding accuracy
# ---------------------------------------------------------------------------------

print(f"Best parameters: {study.best_params}")
print(f"Best accuracy: {study.best_value}")
print(study.trials_dataframe()['params_classifier'].value_counts())
print(study.trials_dataframe().groupby('params_classifier')['value'].mean())


from optuna.visualization import plot_contour

plot_contour(study).show()