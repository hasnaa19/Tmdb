import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

####################################################################################################
#                                            read data
###################################################################################################

data = pd.read_csv(r'C:\Users\Karma\Downloads\tmdb-movies (2).csv')

print('\n DATA BEFORE PREPROCESSING :- ')
print('------------------------------\n')
print(data.head())

####################################################################################################
#                                          preprocessing
####################################################################################################

#  (1) removing duplicate rows
data = data.drop_duplicates()

####################################################################################################

# (2)feature selection & dropping columns with ALMOST unique values
print('\nCOLUMNS BEFORE FEATURE SELECTION :- ')
print('-----------------------------------\n')
print(data.columns)
d = ['id', 'imdb_id', 'original_title', 'homepage', 'tagline', 'keywords', 'overview', 'director', 'cast',
     'production_companies']
data.drop(data[d], axis=1, inplace=True)
print('\nCOLUMNS AFTER FEATURE SELECTION :- ')
print('-----------------------------------\n')
print(data.columns)

####################################################################################################

#  (3) Drop the rows that contain missing values
print('\nNUMBER OF NULLS IN EACH COLUMN:- ')
print('---------------------------------\n')
print(data.isnull().sum())
data.dropna(how='any', inplace=True)
print('\n ROWS WITH NULLS DROPPED.')
print('---------------------------------\n')
print(data.isnull().sum())  # == zero
####################################################################################################

#  (4) encoding > applying one hot encoding
data = pd.concat([data.drop('genres', 1), data['genres'].str.get_dummies(sep="|")], 1)

data['release_date_parsed'] = pd.to_datetime(data['release_date'], infer_datetime_format=True)
# convert string Date time into date time object
data['release_date_month'] = data['release_date_parsed'].dt.month
# return a numpy array containing the month of the datetime in the underlying data of the given series object.
data['release_date_dayofweek'] = data['release_date_parsed'].dt.dayofweek
data.drop(['release_date', 'release_date_parsed'], axis=1, inplace=True)

print('\nCOLUMNS ADDED TO THE DATAFRAME AFTER GENRES ENCODING :- ')
print('---------------------------------------------------------\n')
print(data.iloc[:, 7:27].head())
print('\nCOLUMNS ADDED TO THE DATAFRAME AFTER HANDLING RELEASE DATE :- ')
print('---------------------------------------------------------\n')
print(data.iloc[:, 27:].head())

####################################################################################################

#  add net_profit column then drop revenue and budget

data['net_profit'] = data.revenue_adj - data.budget_adj
k = ['revenue_adj', 'budget_adj']
data.drop(data[k], axis=1, inplace=True)

print("\nData After Adding net profit column and dropping revenue and budget columns ")
print('---------------------------------------------------------------------------\n')
print(data.head())

####################################################################################################

#  loading data
X = data.loc[:, data.columns != 'net_profit']
# print("\n\nX\n\n", X.dtypes)
Y = data['net_profit']

print("Data after Loading")
print('------------------------\n')
print("X: ",  X.head(), "\n\nY: ", Y.head())
####################################################################################################

#  Get correlation
print('---------------------------------------------------------\n')
Total_data = data.iloc[:, :]
correlation = Total_data.corr()
Impact = correlation.index[abs(correlation['net_profit'] > 0.5)]
Total_data = Total_data[Impact]
Highest_Impact = Total_data.loc[:, Total_data.columns != 'net_profit']

print("Features within highest impact/ correlation are ")
print('--------------------------------------------------\n', Highest_Impact.columns, '\n')
###################################################################################################

#  data splitting

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, shuffle=True, random_state=50)


####################################################################################################
#                                           Feature Scaling
####################################################################################################

# Normalization Scratch
print('\t\tBEFORE SCALING:')
print('------------------------------\n')
print("X_Train:\n ", x_train.head())
print('------------------------------\n')
print("X_Test:\n ", x_test.head())
print('------------------------------\n')

def feature_normalize(X):
    X__norm = (X - X.min()) / (X.max() - X.min())
    return X__norm

x_train = feature_normalize(x_train)
x_test = feature_normalize(x_test)

# OR using Standardization
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.transform(x_test)
print('\t\tAFTER SCALING:')
print('------------------------------\n')
print("X_Train:\n ", x_train.head())
print('------------------------------\n')
print("X_Test:\n ", x_test.head())
print('------------------------------\n')
####################################################################################################
#                                       Regression From Scratch
####################################################################################################

# get Features Size
n_Features = len(Y)
print("\nM (number of features) = ", n_Features)
print('------------------------------\n')

# Cost Function Mean ^ error

def Cost_FunctionMSE(X, y_actual, theta):
    y_prediction = X.dot(theta)  # h(Xi) = theta * X
    errors = np.subtract(y_prediction, y_actual) ** 2  # (h(Xi) - Yi)squared
    MSE = 1 / (2 * n_Features) * np.sum(errors)  # summation/sigma((h(Xi) - Yi)squared)
    # OR MSE = 1/(2 * n_Features) * errors.T.dot(errors)
    return MSE


##################################################################################

#  gradient descent

def gradient_descent(X, y, theta, alpha, iterations):
    MSE_history = np.zeros(iterations)
    for i in range(iterations):
        y_predictions = X.dot(theta)  # current iteration theta
        # print('predictions= ', predictions[:5])
        errors = np.subtract(y_predictions, y)  # computing cost of theta of i-1
        # print('errors= ', errors[:5])
        sum_delta = (alpha / n_Features) * X.transpose().dot(
            errors)  # theta(i-1) - learning rate(derivative cost function)

        # print('sum_delta= ', sum_delta[:5])
        theta = theta - sum_delta

        MSE_history[i] = Cost_FunctionMSE(X, y, theta)

    return theta, MSE_history


#############################################################################################
#                                                Modeling
###############################################################################################

# Model Train
print('\n---------------------------------------------------------------')
print("\t\t\t\t\t\t Model TRAIN")
print('--------------------------------------------------------------')
i = 1500
a = 0.40
theta, MSE_history = gradient_descent(X=x_train, y=y_train, theta=np.zeros(27), alpha=a, iterations=i)
print('\nFinal value of Theta Theta AKA Coefficients = ')
print('--------------------------------------------------\n', theta)
print('\nFirst 5 values from MSE_history = ')
print('--------------------------------------------------\n', MSE_history[:5])
print('\nLast 5 values from MSE_history = ')
print('--------------------------------------------------\n', MSE_history[-5:])
# Model Test
print('\n---------------------------------------------------------------')
print("\t\t\t\t\t\t Model TEST")
print('---------------------------------------------------------------\n')
print("MEAN SQUARE ERROR: ", Cost_FunctionMSE(x_test, y_test, theta))

####################################################################################################
#                                                Plotting
####################################################################################################

#  correlation plot
plt.subplots(figsize=(12, 8))
highCO = Total_data[Impact].corr()
sns.heatmap(highCO, annot=True)
plt.show()
# print(data[Impact])

#   Plot the cost function...
plt.title('Cost Function MSE')
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
plt.plot(MSE_history)
plt.show()

#  Plot X impact on Y
sns.pairplot(Total_data)
plt.show()
###########################################################################################################