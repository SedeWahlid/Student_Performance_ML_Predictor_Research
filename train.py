import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from joblib import dump
import json 


#################################################
# DATA TRANSFORMATIONS | MANIPULATION
#################################################

# encoding the data based on all 3 targets or only one target
def transform_data_into_numbers(data:pd.DataFrame, useALL:bool,whichScore = None)-> pd.DataFrame:
    if useALL:
        X = data.drop(columns=["math score","reading score","writing score"])
        Y = data[["math score","reading score","writing score"]]
        encoded_data = pd.get_dummies(X,drop_first=True)
        return encoded_data,Y
    else:
        X = data.drop(columns=[whichScore])
        Y = data[[whichScore]]
        encoded_data = pd.get_dummies(X,drop_first=True)
        return encoded_data,Y
        
# splitting data into training and test data 80/20
def split_data_into_train_test(X,Y)-> list:
    X_train, X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=14)
    return X_train, X_test,y_train,y_test

# saving the models
def freeze_model(model:LinearRegression|RandomForestRegressor,filename:str):
    if isinstance(model,LinearRegression):
        modelfile = f'models/{filename}_linear.joblib'
    else : 
        modelfile = f'models/{filename}_randomForest.joblib'
    try:
        dump(model, modelfile, compress=3)
        print(f"Model successfully saved to {modelfile}")
    except Exception as e:
        print(f"Error saving object: {e}")

# saving all the combined reports for each model training report 
def save_report(all_reports: dict)-> None:
    with open("reports/all_reports.json",'w') as file:
        json.dump(all_reports,file,indent=4)
    

#################################################
# MODELS
#################################################

def training_model_linearRegression(X_train:list, X_test:list,y_train:list,y_test:list)-> LinearRegression:
    model = LinearRegression()
    model.fit(X_train,y_train)
    predict = model.predict(X_test)
    r2_per_target = r2_score(y_test,predict,multioutput="raw_values")
    r2_average = r2_score(y_test,predict,multioutput="uniform_average")
    mse = mean_squared_error(y_test,predict,multioutput="uniform_average")
    report = f"""Linear Regression:
    r2 per target: {r2_per_target}
    r2 average: {r2_average}
    mean squared error: {mse}"""
    return model,report

def training_model_randomForest(X_train:list, X_test:list,y_train:list,y_test:list)-> RandomForestRegressor:
    model = RandomForestRegressor(n_estimators=100,random_state=14)
    model.fit(X_train,y_train)
    predict = model.predict(X_test)
    r2_per_target = r2_score(y_test,predict,multioutput="raw_values")
    r2_average = r2_score(y_test,predict,multioutput="uniform_average")
    mse = mean_squared_error(y_test,predict,multioutput="uniform_average")
    report = f""" RandomForest:
    r2 per target: {r2_per_target}
    r2 average: {r2_average}
    mean squared error: {mse}"""
    return model,report
    
#################################################
# MAIN 
#################################################

data = pd.read_csv("data/StudentsPerformance.csv")
all_reports = {"all_scores_linear":None,"all_scores_randomForest": None, "math score":[], "reading score": [],"writing score":[]}
# -- All scores included --
X_all_scores,Y_all_scores = transform_data_into_numbers(data,useALL=True)
X_train, X_test,y_train,y_test = split_data_into_train_test(X_all_scores,Y_all_scores)
model,report = training_model_linearRegression(X_train, X_test,y_train,y_test)
all_reports["all_scores_linear"] = report
freeze_model(model,"all_scores_linear")
model,report = training_model_randomForest(X_train, X_test,y_train,y_test)
all_reports["all_scores_randomForest"] = report
freeze_model(model,"all_scores_randomForest")

# -- specific scores --
for score in ["math score","reading score","writing score"]:
    X,Y = transform_data_into_numbers(data,useALL=False, whichScore=score)
    X_train, X_test,y_train,y_test = split_data_into_train_test(X,Y)
    model,report = training_model_linearRegression(X_train, X_test,y_train,y_test)
    all_reports[score].append(report)
    freeze_model(model,score)
    model,report = training_model_randomForest(X_train, X_test,y_train,y_test)
    all_reports[score].append(report)
    freeze_model(model,score)

save_report(all_reports)