"""
Welcome to the Utilities function libary for the Energy Data Analytics Lab Bass Connections Project. 
This library of functions will be useful for extracting and manipulating Indian village data.
This file, named utils.py, should be in the same directory as feature_extraction.ipynb, as this notebook imports and utlizes some of these functions. 
Thorough descriptions and docstrings for each function are provided in the definitions. 
For additional clarification, feel free to reach out to a member of the 2017-2018 team. 


Import this into a jupyter notebook or other python file in the same directory with: 
from utils import *
confirm_utils() # confirmation message

"""
# Imports 

# General 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
import math
from skimage import io
import re

# Data Preprocessing 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

# Data Visualization
import seaborn as sns
from matplotlib import style
from seaborn import heatmap

# Classifiers & Algorithms 
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

# Evaluation Metrics
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score



def confirm_utils():
    """Confirmation message to ensure utils.py imported correctly."""
    print ('Utilities library successfully loaded.') 
    return 

def create_csv(feature_labels, VIIRS_IMAGE_PATH = "./indian_village_dataset/imagery_res30_48bands/", 
    MASK_IMAGE_PATH = "./indian_village_dataset/masks_res30/", csv_name = 'output.csv', debug = False): 
    """
    Output a CSV file with rows of villages and columns of image features.

    Parameters
    feature_labels (List of strings): List of desired feature labels
    VIIRS_IMAGE_PATH (str): path of image directory
    MASK_IMAGE_PATH (str): path of masks directory 
    csv_name (str): desired output CSV filename 
    debug (boolean): If True, does a test run with 100 villages instead of on all data. 
    """
    files = [file for file in os.listdir(MASK_IMAGE_PATH) if file.endswith('.tif')] 

    if (debug): 
        files = np.random.choice(files, 100) 
        print ("Debug Mode: Testing on 100 villages.") 
    else: 
        print ("Full Mode: Running for all {} villages.".format(len(files)))

    features, village_ids = np.array([]), np.array([])
    counter_invalid, id_invalid, error_image = 0, 0, 0

    print('Initialized file reading.')
    for i, file in enumerate(files):
        
        try: 
            im = io.imread(VIIRS_IMAGE_PATH + file) # read image
            regexp = re.search('([0-9])\w+', file) # check for numbers in filename 
            # check for invalids  
            if im.shape[2] < 47: 
                counter_invalid += 1
            elif type(regexp) == type(None): 
                id_invalid += 1
            else:
                lights = im[:, :, 47]  # get lights at night band
                mask = io.imread(MASK_IMAGE_PATH + file)
                valid_lights = lights[mask>0]

                stats = np.percentile(valid_lights, [0,10,50,90,100], interpolation='nearest') # min, 10p, med, 90p, max 
                mean = np.mean(valid_lights) 
                variance_lights = np.std(valid_lights) # standard dev 
                sum_lights = np.sum(valid_lights) # total_lan (sum)
                area = len(lights) # area 
                stats = np.append(stats, [mean, variance_lights, sum_lights, area])

                village_id = str(file).split('-')[-1][:-4]
                village_ids = np.append(village_ids, village_id)
                for x in range(0, 47): # for everything except lan, add all features to a list, then add list to features. 
                    layer = im[:, :, x]
                    if len(layer) == 0:
                        counter_invalid += 1 
                    else: 
                        valid_layer = layer[mask>0]
                        more_features = [np.amax(valid_layer), np.mean(valid_layer), np.std(valid_layer), np.median(valid_layer),
                                   np.sum(valid_layer), np.percentile(valid_layer, 10, interpolation='nearest'),
                                   np.percentile(valid_layer, 90, interpolation='nearest')] # add here 
                        stats = np.append(stats, more_features, axis=0)
             #   assertEqual(len(stats), len(feature_labels), msg='Number of feature labels != Number of features extracted!')
                if len(feature_labels) == len(stats): 
                    features = np.append(features, stats, axis=0)
            #features.append(stats)
            # print (len(feature_labels), len(stats))
            # print (features.shape)
            if i % (len(files)//10) == 0: # print message every ~10,000 files, just to know it's working
                print ('{} of {} image files read.'.format(i, len(files)))
        except: 
            counter_invalid += 1
    
    print ('Number of invalid images: {}, number of invalid IDs: {}'.format(counter_invalid, id_invalid))
    features = features.reshape((-1, len(feature_labels)))
    data = pd.DataFrame(data = features, index = village_ids, columns = feature_labels)  
    data.to_csv(csv_name)
    return ("Finished writing CSV file {}.".format(csv_name))

def preprocess_garv(garv_data_path, dropNaNs = False): 
    """
    DataFrame creation, preprocessing and setup for GARV data. Things like column names, etc. can be modified as needed easily. 

    Parameters
    garv_data_path (str): Path of CSV file of the GARV dataset
    dropNaNs (boolean): If True, drops rows with NaNs

    Returns dataframe with a new percentage electrified column, cleaned of nan values, and dropped duplicates. 
    Replaces -9 (missing data) values with np.NaN.
    """
    df = pd.read_csv(garv_data_path)
    df = df.replace(-9, np.nan)
    df['Census 2011 ID'] = df['Census 2011 ID'].astype(str).str[:-2] # Drop ".0" from census ID 
    df['Percentage Electrified'] = (df['Number of Electrified Households']/df['Number of Households'])*100
    if dropNans: 
        df = df.dropna(axis=0, how='any') # drop rows that have NaN values 
    df[~df.index.duplicated(keep=False)]
    print ('Preprocessing of GARV dataframe complete.')
    return df 

def preprocess_features(csv_name: str): 
    """
    Preprocesses the feature dataframe to have numerical values when applicable, drop full NaN rows, etc. 

    Parameters 
    csv_name: Name of feature dataframe to read (created from the create_csv function above, ideally in the same directory.)

    Returns the modified dataframe.
    """
    df = pd.read_csv(csv_name, skip_blank_lines=True).dropna(axis=0, how='all')
    df = df.rename(index=str, columns={"Unnamed: 0": "Census 2011 ID"})
    df['Census 2011 ID'] = df['Census 2011 ID'].astype(str)
    df = df[~df.index.duplicated(keep=False)]
    df = df.apply(pd.to_numeric, errors="ignore")
    print ('Preprocessing of feature dataframe complete.')
    return df 

def clean_read_numerical(path: str): 
    """
    Read in a CSV file without the "Unnamed" columns, and with all values numerical. 

    Argument: path- the file location of the CSV file to read as a string 
    Returns: DataFrame with 'Unnamed' columns removed. 

    """
    df = pd.read_csv(path)
    for col_name in df.columns:
        if str(col_name[:7]) == 'Unnamed':
            del df[col_name]

    for col in df:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def interpolate(df: pd.DataFrame): 
    """
    Returns the DataFrame with missing values filled in using column (axis = 1) interpolation. 
    For example, if we're missing data for the month of March, the surrounding months of January and February will be averaged and imputed for March. 
    Edge cases are considered negligible and are ignored. 
    """
    df = df.interpolate(axis=1)
    print ('Interpolation successful.')
    return df 

def make_train_test_split(df: pd.DataFrame, ratio: float): 
    """
    Make train and test split for the DataFrame with the desired ratio.

    Arguments: 
    df: DataFrame containing all data 
    ratio: Proportion of the data (0 < x < 1) desired for the testing set. 

    Returns: A tuple (a, b) of two DataFrames, the train and test DataFrames, respectively: (Test, Train)
    """
    df, test_data_df = train_test_split(df, test_size=ratio)
    print ('Train Shape', df.shape, 'Test Shape', test_data_df.shape)

def upsample(df: pd.DataFrame):
    """
    Upsample the minority class in the training data for better classification

    Arguments: 
    df: DataFrame containing all training DataFrame 

    Returns: A DataFrame with upsampled minority class (unelectrified villages) and downsampled majority class (electrified)
    """
    df0 = df[df.electric_category == 0]
    df1 = df[df.electric_category == 1]

    print ('Sizes before sampling: ', df1.shape, df0.shape)

    df0 = resample(df0, 
                    replace=True,     # sample with replacement
                    n_samples=12000,  # to match majority class, Was 10000 
                    random_state=123) # reproducible results. 

    df1 = resample(df1, n_samples = 12000)
    df = pd.concat([df1, df0])

    print ('Train Shape', df.shape, 'Test Shape', test_data_df.shape)
    print ('New Train Classes')
    print (df.electric_category.value_counts())
    print ('New Train Classes')
    print (test_data_df.electric_category.value_counts())
    return df 

def check_train_test_valid(train_df: pd.DataFrame, test_df: pd.DataFrame): 
    """
    Simple sanity check that ensures that the Training and Testing datasets have no rows in common

    Arguments: 
    train_df: Training DataFrame
    test_df: Testing DataFrame

    Returns: Boolean, if True, then training and testing data are different as they should be. 
    """
    a = [x for x in train_df['Census 2011 ID'].values if x in test_df['Census 2011 ID'].values]
    if len(a) == 0: 
        print ('Train and Test dataframes are valid and contain no same rows.')
        return True 
    else: 
        print ('Warning: Train and Test dataframes have some rows in common-')
        print (a)
        return False 

def scale(df: pd.DataFrame, test_data_df: pd.DataFrame, minmaxscaling = True): 
    """
    Scale the training and testing data in the proper way (with attention to transform and fit transform)

    Arguments: 
    df: The training dataframe
    test_data_df: The testing dataframe
    minmax: (Boolean) If true, MinMax(0,1) scaling will be used over the Standard Scaler, which is used otherwise. 
    MinMaxScaler is used by default. 

    Returns: (X_train, Y_train, X_test, Y_test) DataFrames properly scaled. 
    """
    if (minmaxscaling): 
        scaler = MinMaxScaler(feature_range=(0, 1))
    else: 
        scaler = StandardScaler()
    # Training 
    X_training = df.drop(['electric_category'], axis = 1)
    X_training_scaled = pd.DataFrame(scaler.fit_transform(X_training), columns = X_training.columns)
    Y_training = df[['electric_category']].values

    # Testing
    X_testing = test_data_df.drop(['electric_category'], axis = 1)
    X_testing_scaled = pd.DataFrame(scaler.transform(X_testing), columns = X_testing.columns)
    Y_testing = test_data_df[['electric_category']].values

    # Simpler names for usage in models 
    X_train = X_training_scaled
    Y_train = Y_training.ravel()
    X_test = X_testing_scaled
    Y_test = Y_testing
    
    return (X_train, Y_train, X_test, Y_test)


def basic_rf_model(X_train, Y_train, X_test, Y_test): 
    """
    Trains a basic, non-hyperparametrized Random Forest classifier on the data to guage results. 

    Arguments: 
    X_train, Y_train, X_test, Y_test dataframes output from the scale step above. 

    Returns: List of predictions for each row in X_test. 

    """
    forest_model = RandomForestClassifier()
    forest_model.fit(X_train, Y_train) 
    predictions = forest_model.predict(X_test)
    print ('Basic Accuracy Score (warning- very preliminary metric):')
    print ('The accuracy is {}, with {} villages correctly classified.'.format(accuracy_score(Y_test, predictions), 
            accuracy_score(Y_test, predictions, normalize=False)))
    return predictions

def basic_evaluation(Y_test, predictions): 
    """
    Basic model evaluation functions. 

    Arguments: 
    Y_test: Actual labels of the rows in X_test
    predictions: Classifier predictions of the rows in X_test 

    Prints out basic model evaluation tools- confusion matrix, classification report. 

    """
    print("Confusion Matrix:")
    print(confusion_matrix(Y_test, predictions))
    print()
    print("Classification Report")
    print(classification_report(Y_test, predictions))


def plot_confusion_matrix(title: str, Y_test, predictions): 
    """
    Plot a confusion matrix of the classification. 

    Arguments: 
    title (str): Title of the confusion matrix 
    Y_test: Actual labels of the rows in X_test 
    predictions: Classifier predictions of the rows in X_test 
    """
    conf_matrix = confusion_matrix(Y_test, predictions)
    heatmap(conf_matrix, annot=True, fmt="d", annot_kws={"size":17})
    plt.xlabel('Prediction', fontsize=18)
    plt.ylabel('Truth', fontsize=18)
    plt.title(title, fontsize=22)
    plt.show()
    
def advanced_metrics_rf(X_test, Y_test, predictions): 
    """Various Metrics"""
    print("F1 Score:", f1_score(Y_test, predictions))
    print("Precision:", precision_score(Y_test, predictions))
    print("Recall:", recall_score(Y_test, predictions))

    y_scores = random_forest.predict_proba(X_test)
    y_scores = y_scores[:,1]
    print("ROC-AUC-Score:", roc_auc_score(Y_test, y_scores))

"""
Note: Check models_test.ipynb for code on Cross Validation, feature importances, etc. 
"""

def plot_precision_vs_recall(Y_test, X_test):
    """Plots a Precision-Recall curve"""
    y_scores = random_forest.predict_proba(X_test)
    y_scores = y_scores[:,1]
    precision, recall, threshold = precision_recall_curve(Y_test, y_scores)

    plt.plot(recall, precision, "g--", linewidth=2.5)
    plt.ylabel("recall", fontsize=19)
    plt.xlabel("precision", fontsize=19)
    plt.axis([0, 1.5, 0, 1.5])
    # plt.figure(figsize=(14, 7)) # Adjust size if needed
    plt.show()


def plot_roc_curve(Y_test, label=None):
    """Plots an ROC curve"""
    y_scores = random_forest.predict_proba(X_test)
    y_scores = y_scores[:,1]
    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, y_scores)
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'r', linewidth=4)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)
    plt.show()
    # plt.savefig('name.png') # Use to save image 
