from sklearn.linear_model import LogisticRegression
import argparse
import os
import joblib
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core.run import Run
import numpy as np
from azureml.data.dataset_factory import TabularDatasetFactory

#Functions to clean data
def Impute_missing_values(df):
    df.drop(columns=['Cabin'],inplace=True)
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna('S', inplace=True)
    df['Fare'].fillna(df['Fare'].median()) 
    return df

def Family_type(number):
    if number==0:
        return 'Alone'
    elif number>0 and number<=4:
        return 'Medium'
    else:
        return 'Large'

def Transform_data(df):
    df['Family_size']=df['Parch']+df['SibSp']
    df['Family_type']=df['Family_size'].apply(Family_type)
    df.drop(columns=['SibSp', 'Parch', 'Family_size'], inplace=True)
    df.loc[ df['Age'] <= 16, 'Age'] = 1
    df.loc[(df['Age'] > 16) & (df['Age'] <= 26), 'Age'] = 2
    df.loc[(df['Age'] > 26) & (df['Age'] <= 36), 'Age'] = 3
    df.loc[(df['Age'] > 36) & (df['Age'] <= 62), 'Age'] = 4
    df.loc[ df['Age'] > 62, 'Age'] = 5
    df.loc[df['Fare'] <= 17, 'Fare'] = 1,
    df.loc[(df['Fare'] > 17) & (df['Fare'] <= 30), 'Fare'] = 2,
    df.loc[(df['Fare'] > 30) & (df['Fare'] <= 100), 'Fare'] = 3,
    df.loc[ df['Fare'] > 100, 'Fare'] = 4
    return df

def clean_data(df):
    df = Impute_missing_values(df)
    df.head()
    x_df = Transform_data(df)
    x_df=pd.get_dummies(data=x_df, columns=['Age' ,'Fare',  'Pclass', 'Sex', 'Embarked', 'Family_type'], drop_first=True)
    x_df.drop(columns=['Ticket', 'PassengerId', 'Name','Age_5.0'],inplace=True)
    y_df = x_df.pop("Survived")
    return x_df, y_df

#Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/parvatijay2901/Machine-Learning-with-the-Titanic-dataset-on-Azure/main/Training_data.csv')

#Clean the dataset
x, y = clean_data(df)

#Split the dataset into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

run = Run.get_context()

def main():
    #Add arguments to script
    parser = argparse.ArgumentParser()
    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")
    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))
    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    run.log("accuracy", np.float(accuracy))
   
   #Dump the model using joblib
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(value=model, filename='outputs/model.pkl')

if __name__ == '__main__':
    main()
