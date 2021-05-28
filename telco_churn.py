import pandas as pd
# import subprocess
# import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

DEBUG = True
MAX_COLUMN_NULL_RATE = 0.2
TESTING_SIZE_RATE = 1
REDUCE_FEATURES_BEST_MODEL = False

###########################################################
# General Routine to Print a specific data from all columns in the dataset
###########################################################
def listFeaturesData(X,feature_data,name_data):
    data_columns = X.columns
    features_table = pd.DataFrame(data=zip(data_columns,feature_data),columns=['Feature',name_data])
    print(features_table)

##################################################################
# General routine to reduce Features for any Classification Model
##################################################################
# Comment : Features the are statistically significant :
# It means that at least in 95% of the times we do a prediction, these feature influences the result
# On the other hand, statiscally non significant features influences the resuly by chance
# The P-Value of a feature statistically significant should be less (< 0.05), since the Hypotesis is that the variable is NOT statistically significant
def reduceFeaturesByModel(model, model_name, X):
    print("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Features reduction using model {}\n".format(model_name))

    # List the variables that are and not are statistically significant
    # False = insignificant feature; True = significant fearture
    columns_significance = model.get_support()

    if DEBUG:
        listFeaturesData(X,columns_significance,"Statistically Significant")
        # print(columns_significance)

    data_columns = X.columns
    non_significant_columns = []
    significant_columns = []
    for i in range(len(columns_significance)):
        if columns_significance[i] == False:
            non_significant_columns.append(data_columns[i])
        else:
            significant_columns.append(data_columns[i])

    print("\nThe following columns have been eliminated using {}".format(model_name))
    print(non_significant_columns)

    columns_kept = len(columns_significance)-len(non_significant_columns)
    print("\nNumber of Columnns KEPT after eliminating features using {} : {}".format(model_name,columns_kept))
    print("Number of Columnns REMOVED using {} : {}".format(model_name,len(non_significant_columns)))

    # Get the transformed ARRAY from X after eliminating variables that are not statistically significant
    X_transformed_Array = model.transform(X)
    print("\nDataset shape after Features reduction :")
    print(X_transformed_Array.shape)
    X_transformed=pd.DataFrame(X_transformed_Array, columns=significant_columns)
    return X_transformed, columns_kept

def reduceFeaturesbyLogisticaBinomial(X_original, y_original, just_data=True):
    import statsmodels.api as sm

    model_logistic_binomial = sm.GLM(y_original, X_original, family=sm.families.Binomial())
    result=model_logistic_binomial.fit(fit_intercept=True)
    summary = result.summary2()
    print(result.summary2())
    # Comment : Features the are statistically significant :
    # It means that at least in 95% of the times we do a prediction, these feature influences the result
    # On the other hand, statiscally non significant features influences the resuly by chance
    # The P-Value of a feature statistically significant should be less (< 0.05), since the Hypotesis is that the variable is NOT statistically significant
    # By Using the Summary, all variables with P-Value larger than 0.05 should be eliminate

    # List the features / columns that are not statistically significant
    data_columns = X_original.columns
    non_significant_columns = []
    for i in range(len(data_columns)):
        if result.pvalues[i] > 0.05:
            non_significant_columns.append(data_columns[i])

    model_name = "Logistic Binomial Model"
    print("\nThe following columns have been eliminated using {}".format(model_name))
    print(non_significant_columns)

    columns_kept = len(data_columns)-len(non_significant_columns)
    print("\nNumber of Columnns KEPT after eliminating features using {} : {}".format(model_name,columns_kept))
    print("Number of Columnns REMOVED using {} : {}".format(model_name,len(non_significant_columns)))

    # Get the transformed ARRAY from X after eliminating variables that are not statistically significant
    X_GLM = X_original.drop(columns = non_significant_columns, axis=1)
    print("\nDataset shape after Features reduction :")
    print(X_GLM.shape)

    if just_data:
        return X_GLM
    else:
        return X_GLM, model_name, columns_kept

def evaluateLinearRegressionDataset(X_dataset, y_dataset):
    from sklearn.linear_model import LogisticRegression

    X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, test_size=0.3, random_state=42)
    logreg = LogisticRegression(max_iter=10000)
    logreg.fit(X_train, y_train)
    # y_predicted = logreg.predict(X_test)
    accuracy = logreg.score(X_test, y_test)
    return accuracy #, y_test, y_predicted

def reduceFeatures(X_original,y_original):
    from sklearn.feature_selection import SelectFromModel
    from sklearn.linear_model import LogisticRegression

    datasets_models = pd.DataFrame(columns = ['Model_Name',
                                                # 'Model',
                                                'Model_Dataset','Columns_Kept','Model_Dataset_Accuracy'])

    #### Performing Feature Selection
    #### OPTION 1 - Using Logistic Regression

    model_logistic_regression = SelectFromModel(estimator=LogisticRegression(max_iter=5000)).fit(X_original, y_original)

    X_logistic_regression, columns_kept = reduceFeaturesByModel(model_logistic_regression, "Logistic Regression", X_original)

    datasets_models = datasets_models.append({'Model_Name' : "Logistic Regression",
                                            'Model_Dataset' : X_logistic_regression, 'Columns_Kept' : columns_kept },
                                                ignore_index=True)

    # List the abs(coefficient) value for feature significance
    # The fatures above the abs(thersholds) are not statistically significant
    # It means that it cannot be guaranteed that this variables will impact the prediction in at least 95% of the cases
    print("\nLogistic Threshold used to remove statistically significant feature :")
    # Cutoff abs(coefficient) value for feature significance
    print(model_logistic_regression.threshold_)

    # List the coeficients of each one of the variables used to predict
    print("\nLogistic Coeficient for each feature:")
    listFeaturesData(X_original,model_logistic_regression.estimator_.coef_[0], "Feature Coeficient")

    # Logistic Coeficient Interpretation
    # 1) Sign of the Coeficient
    # SeniorCitizen            0.150507
    # A positive sign means that, all else being equal, senior citizens were more likely to have churned than non-senior citizens.
    #  tenure           -0.042463
    # A negative sign means that, all else being equal, more time as customer is less likely to churn than new customer
    # 2) Magnitude
    # InternetService_Yes            0.251419
    # DeviceProtection_Yes            0.091653
    # If everything is a very similar magnitude, a larger pos/neg coefficient means larger effect, all things being equal.
    # So InternetService_Yes has more than twice the effect than DeviceProtection_Yes on the prediction result
    # However, if your data isn't normalized, the magnitude of the coefficients don't mean anything (without context).
    # For instance you could get different coefficients by changing the units of measure to be larger or smaller.
    # So keep in mind that logistic regression coefficients are in fact odds ratios,
    # and you need to transform them to probabilities to get something more directly interpretable.
    # By setting the Coeficient Threshold as 0.38, it means that the Features with impact less than 0.38 will be elimiated by the model

    #### Performing Feature Selection
    #### OPTION 2 - Using Linear Support Vector Classification

    from sklearn.svm import LinearSVC
    from sklearn.feature_selection import SelectFromModel

    lsvc = LinearSVC(C=0.02, penalty="l1", dual=False).fit(X_original, y_original)
    model_LSVC = SelectFromModel(lsvc, prefit=True)
    model_name = "LinearSVC"
    X_LSVC, columns_kept = reduceFeaturesByModel(model_LSVC, "model_name", X_original)

    datasets_models = datasets_models.append({'Model_Name' : model_name,
                                'Model_Dataset' : X_LSVC, 'Columns_Kept' : columns_kept },ignore_index=True)

    #### Performing Feature Selection
    #### OPTION 3 - Using SelectKBest and Mutual Info Classification
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.feature_selection import SelectKBest

    # As parameter for KBest we will use half of the fatures
    model_KBest_mutual = SelectKBest(score_func=mutual_info_classif,k=int(len(X_original.columns)/2))

    results = model_KBest_mutual.fit(X_original,y_original)

    model_name = "SelectKBest / Mutual Info Classification"
    X_KBest_mutual, columns_kept = reduceFeaturesByModel(model_KBest_mutual, model_name , X_original)
    datasets_models = datasets_models.append({'Model_Name' : model_name,
                                'Model_Dataset' : X_KBest_mutual, 'Columns_Kept' : columns_kept },ignore_index=True)

    print("\nKBest scores for each feature:")
    listFeaturesData(X_original,results.scores_, "Feature Score")
    # score_func=mutual_info_classif Interpretation
    # The mutual information (MI) between two random variables or random vectors
    # measures the “amount of information”, i.e. the “loss of uncertainty” that one can
    # bring to the knowledge of the other, and vice versa.

    #### Performing Feature Selection
    #### OPTION 4 - RUN THE MODEL WITH ALL FEATURES AND SELECT ONLY SIGNIFICANT ONES

    X_GLM, model_name, columns_kept  = reduceFeaturesbyLogisticaBinomial(X_original, y_original, just_data=False)

    datasets_models = datasets_models.append({'Model_Name' : model_name,
                                'Model_Dataset' : X_GLM, 'Columns_Kept' : columns_kept },ignore_index=True)

    #############################################################
    # Evaluate the best Logistic Regression performance dataset
    #############################################################
    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics
    from sklearn.model_selection import train_test_split

    for i, datasets_models_row in datasets_models.iterrows():
        accuracy = evaluateLinearRegressionDataset(datasets_models_row['Model_Dataset'], y_original)
        datasets_models.loc[i,'Model_Dataset_Accuracy'] = accuracy


    print("Accuracy of Logistic Regression classifier on test set reduced from model :")
    datasets_models = datasets_models.sort_values(by=['Model_Dataset_Accuracy'], ascending=False).reset_index()
    print(datasets_models[['Model_Name', 'Columns_Kept','Model_Dataset_Accuracy']])

    # Selecting the best dataset, from the best model used to reduce features
    X_final = datasets_models.iloc[0]['Model_Dataset']

    if DEBUG:
        print("Head of the best dataset")
        print(X_final.head())

    return X_final

###########################################################
# Implementing Logistic Regression Analysis over a dataset
############################################################
def LogisticRegressionAnalysis(X_dataset, y_dataset):
    # Get Summary
    logit_model = sm.GLM(list(y_dataset), X_dataset, family=sm.families.Binomial())
    result=logit_model.fit(fit_intercept=True)
    print(result.summary2())

    # odds ratios and 95% CI
    params = result.params
    conf = result.conf_int()
    conf['OR'] = params
    conf.columns = ['2.5%', '97.5%', 'OR']
    print(np.exp(conf))

#######################################################
# Show the Confusion Matrix HeatMap
#######################################################
def showConfusionMatrixGraph(confusion_matrix, y_test):
    # Graph Confusion Matrix
    import seaborn as sns

    df_cm = pd.DataFrame(confusion_matrix, columns=np.unique(y_test), index = np.unique(y_test))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize = (5,5))
    sns.set(font_scale=1.2)
    sns.heatmap(df_cm, annot=True,annot_kws={"size": 12}, cbar=False, vmax=500, square=True, fmt="d", cmap="Reds")
    # sns.heatmap(df_cm , annot=True,annot_kws={"size": 12}, cbar=False, vmax=500, square=True, fmt="d", cmap="Reds")
    # plt.show()

#######################################################
# Show the ROC Graph
#######################################################
def show_ROC_Graph(X_test, y_test, model):
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    import matplotlib.pyplot as plt

    logit_roc_auc = roc_auc_score(y_test, model.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()


###########################################
# PROGRAM BEGINNING
###########################################

###########################################
# Data Preparation Cleansing and Wrangling
###########################################
dfChurnOriginal = pd.read_csv('Telco_Customer_Churn.csv')

if DEBUG:
    print(dfChurnOriginal.head())

if TESTING_SIZE_RATE < 1:
    TESTING_SIZE = int(dfChurnOriginal.shape[0] * TESTING_SIZE_RATE)
    dfChurnOriginal = dfChurnOriginal.loc[:TESTING_SIZE,]
    print("We are in QUICK TESTING mode. The dataset was reduced to {:.2f} %, {} rows".format(TESTING_SIZE_RATE, TESTING_SIZE))

# Drop customer Id from the dataset
dfChurnOriginal =  dfChurnOriginal.drop(['customerID'], axis=1)

# Convert Dummies Variables
dummy_columns_Yes_No = ['gender','Dependents', 'PhoneService','InternetService','DeviceProtection','TechSupport','CableService','PaperlessBilling','Churn']
dfChurn = pd.get_dummies(dfChurnOriginal,columns=dummy_columns_Yes_No, drop_first=True)

dummy_columns_Many = ['Contract','PaymentMethod']
dfChurn = pd.get_dummies(dfChurn,columns=dummy_columns_Many, drop_first=False)

if DEBUG:
    print(dfChurn.dtypes)

# Converting column 'TotalCharges' to numeric
cols = [ 'TotalCharges']
dfChurn[cols] = dfChurn[cols].replace({ " " : "" }, regex=True)
dfChurn[cols]= dfChurn[cols].apply(pd.to_numeric)

if DEBUG:
    print(dfChurn.dtypes)

if DEBUG:
    dfChurn.to_csv("validation.csv")

# Count the lines with num values for each column
# Bonus, to be used in other exercizes, but not in this one. If a column has more then 20% o null values, drop the the entire column
if DEBUG:
    columns_null = dfChurn.isnull().sum() / len(dfChurn)
    print(columns_null)
    drop_cols = []
    for index, value in columns_null.items():
        if value > MAX_COLUMN_NULL_RATE:
            drop_cols.append( index )
    if drop_cols:
        dfChurn =  dfChurn.drop(drop_cols, axis=1)
        columns_null = dfChurn.isnull().sum() / len(dfChurn)
        print(columns_null)

# Drop all lines with a NaN value
# In this case, only in 'TotalCharges' column
if DEBUG:
    print(dfChurn.count())

dfChurn = dfChurn.dropna()

if DEBUG:
    columns_null = dfChurn.isnull().sum() / len(dfChurn)
    print(columns_null)
    print(dfChurn.count())

# Drop the predicted variable from X dataset
X_dfChurn = dfChurn.drop(['Churn_Yes'], axis=1)

# Create the Y series with predicted variable
y_dfChurn = dfChurn['Churn_Yes']

LogisticRegressionAnalysis(X_dfChurn, list(y_dfChurn))

if REDUCE_FEATURES_BEST_MODEL:
    X_final = reduceFeatures(X_dfChurn,y_dfChurn)
else:
    X_final = reduceFeaturesbyLogisticaBinomial(X_dfChurn,y_dfChurn)

if DEBUG:
    print("Dataset shapes after feature reduction")
    print(X_final.shape)
    print(X_dfChurn.shape)
    print(y_dfChurn.shape)

X_train, X_test, y_train, y_test = train_test_split(X_final, y_dfChurn, test_size=0.25, random_state=42)

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, log_loss

classifiers_names = [
"Logistic Regression",
"Nearest Neighbors",
"Linear SVM",
"RBF SVM",
"MPL Neural Net",
"Decision Tree",
"Naive Bayes (GaussianNB)",
"Random Forest",
"Bagging Classifier"
"AdaBoost",
"XGBoost",

# Not used in this exercize
# "Gaussian Process",
# "QDA"
]
scores = []

classifiers = [
    LogisticRegression(),
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, probability=True),
    SVC(gamma=2, C=1, probability=True),
    MLPClassifier(alpha=0.001, solver='lbfgs', learning_rate='adaptive', max_iter=1000), # Neuro Net
    DecisionTreeClassifier(max_depth=5),
    GaussianNB(),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    BaggingClassifier(),
    AdaBoostClassifier(),
    XGBClassifier(),

    # Not used in this exercize
    # GaussianProcessClassifier(1.0 * RBF(1.0)),
    # QuadraticDiscriminantAnalysis()
]

for classifier in classifiers:
    pipe = Pipeline(steps=[
                      ('classifier', classifier)])
    pipe.fit(X_train, y_train)
    print(classifier)
    scores.append(pipe.score(X_test, y_test))
    print("model score: %.3f" % pipe.score(X_test, y_test))
    print("\n -----------------------------------------------------------------------------------")

#end of pipeline
scores_df = pd.DataFrame(zip(classifiers_names,classifiers, scores), columns=['Classifier_Name', 'Classifier','AccuracyScore'])
scores_df.sort_values(by=['AccuracyScore'], ascending=False, inplace=True)
print(scores_df)

# Select the Model with best accuracy
model_final = scores_df.iloc[0]['Classifier']

# Make predictions on Test dataset
y_predicted = model_final.predict(X_test)

# Confusion MATRIX
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_predicted)
print(confusion_matrix)

showConfusionMatrixGraph(confusion_matrix,y_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_predicted))

show_ROC_Graph(X_test, y_test, model_final)

prob = model_final.predict_proba(X_test)
X_test['y_pred'] = y_predicted
X_test["pred_prob"] =prob[:,1]

X_test.to_csv("X_test_y_pred_prob.csv")

# Predict specific case
# gender = Male
# SeniorCitizen = 0
# Dependents = No
# tenure = 32
# PhoneService = Yes
# InternetService = Yes
# DeviceProtection = No
# TechSupport = No
# CableService = Yes
# Contract = month-to-month
# PaperlessBilling = Yes
# PaymentMethod = Credit card (automatic)
# MonthlyCharges = 64.75
# TotalCharges = 2283.30

x_case_dict = { 'SeniorCitizen': 0, 'tenure' : 32, 'MonthlyCharges': 64.75, 'TotalCharges': 2283.30, 'gender_Male' : 1,
            'Dependents_Yes' : 0,  'PhoneService_Yes' : 1, 'InternetService_Yes' : 1, 'DeviceProtection_Yes' : 0,
            'TechSupport_Yes' : 0, 'CableService_Yes' : 0, 'PaperlessBilling_Yes': 1,
            'Contract_Month-to-month' : 1,	'Contract_One year': 0,	'Contract_Two year' : 0,
            'PaymentMethod_Bank transfer (automatic)': 0, 'PaymentMethod_Credit card (automatic)': 1, 'PaymentMethod_Electronic check': 0, 'PaymentMethod_Mailed check': 0 }

print(x_case_dict)

# Eliminate data from the input related to the columns / features removed from the dataset
x_final_columns = X_final.columns
x_case_keys = list(x_case_dict)
for key in x_case_keys:
    if not key in x_final_columns:
        x_case_dict.pop(key)

print(x_case_dict)

x_case = [list(x_case_dict.values())]

y_case_predict = model_final.predict(x_case)[0]
print("Test customer profile below to see if the customer will change Telco operator (Churn)")
if y_case_predict == 0:
    print("PREDICTION - The customer with the profile above will NOT CHURN")
else:
    print("PREDICTION - The customer with the profile above will CHURN")
prob_x_case = model_final.predict_proba(x_case)[:,0][0]
print("The probability that this prediction is right is : {}".format(prob_x_case))
