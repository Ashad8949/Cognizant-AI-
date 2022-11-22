# ------- BEFORE STARTING - SOME BASIC TIPS
# You can add a comment within a Python file by using a hashtag '#'
# Anything that comes after the hashtag on the same line, will be considered
# a comment and won't be executed as code by the Python interpreter.

# --- 1) IMPORTING PACKAGES
# The first thing you should always do in a Python file is to import any
# packages that you will need within the file. This should always go at the top
# of the file
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# --- 2) DEFINE GLOBAL CONSTANTS
# Constants are variables that should remain the same througout the entire running
# of the module. You should define these after the imports at the top of the file.
# You should give global constants a name and ensure that they are in all upper
# case, such as: UPPER_CASE
def load_data(path: str = "/path/to/csv/"):
    """This function takes one argument path read and loads
     it into Pandas dataframe
     
     :param             path(optional): str, relative path of CSV file
     :returns           df: DataFrame
     """
    df = pd.read_csv(f"{path}")
    df.drop(columns = ["Unnamed: 0"], inplace=True , error="ignore")
    return df

def global_constants(k: int=10, fold: float= 0.25):
    """This function takes two argumengts k and fold and returns K and FOLD
    param:      k:  int, no of iteration cross-validation makes
    param:      fold: float, test data split size
    returns:    K:    int
                FOLD:    float
    """  
    K = k 
    FOLD = fold
    return K,FOLD

# --- 3) ALGORITHM CODE
# Next, we should write our code that will be executed when a model needs to be 
# trained. There are many ways to structure this code and it is your choice 
# how you wish to do this. The code in the 'module_helper.py' file will break
# the code down into independent functions, which is 1 option. 
# Include your algorithm code in this section below:
def create_target_and_predictors(
    data: pd.DataFrame=None,
    target: str="estimated_stock_pct"):
    """ This functions takes two arguments data and target and split the 
    columns into target column and set of preditor variables i.e. X and y
    
    param:      data: pd.Dataframe, dataframe containing data for model
    param:      target: str(optional), target column

    return:     X : pd.DataFrame
                y : pd.Series
    """
    if target not in data.columns:
        raise Exception(f"Target: {target} is not present in data")
    X = data.drop(target)
    y = data[target]
    return X, y



 
# --- 4) MAIN FUNCTION
# Your algorithm code should contain modular code that can be run independently.
# You may want to include a final function that ties everything together, to allow
# the entire pipeline of loading the data and training the algorithm to be run all
# at 
def train_algorithm_with_cross_validation(
    X : pd.DataFrame=None,
    y : pd.Series=None
):
    """This function takes two argument X features and target column y 
    and returns the value of mean absolute error in each fold of cross 
    validation and also avg of mean absolute error
    
    param:      X: pd.DataFrame
    param:      y: pd.Series
    :return
    """

    accuracy = []

    for fold in range(0, K):
        # split the data into tain and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = SPLIT, random_state = 42)

        # initaiate the algorithms
        model = RandomForestRegressor()
        scaler = StandardScaler()

        # Standard scale the train data
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # train the model 
        trained_model = model.fit(X_train,y_train)

        # predict
        y_pred = trained_model.predict(X_test)

        # mean_absolute_error 
        mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
        print(f"Fold{fold+1}, MAE = {mae:.3f}")

        accuracy.append(mae)
    print(f"Average value of MAE is {sum(accuracy)/len(accuracy):.2f}")








