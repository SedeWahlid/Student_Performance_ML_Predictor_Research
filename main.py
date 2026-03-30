import streamlit as st
from joblib import load
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


#################################################
# LOADING DATA | MODELS
#################################################

reports = pd.read_json("reports/all_reports.json")

# loading the models based on the prediction that we want 
def load_model(whatpredict:str):
    if whatpredict =="math score":
        model_linear = load("models/math score_linear.joblib")
        model_randomForest = load("models/math score_randomForest.joblib")
    elif whatpredict =="reading score":
        model_linear = load("models/reading score_linear.joblib")
        model_randomForest = load("models/reading score_randomForest.joblib")
    elif whatpredict =="writing score":
        model_linear = load("models/writing score_linear.joblib")
        model_randomForest = load("models/writing score_randomForest.joblib")
    return model_linear,model_randomForest
    

#################################################
# INTERFACE
#################################################

# creating columns for the playground and 
def open_playground(whatpredict:str):
    col6,col7,col8,col9,col10,col11 = st.columns(6)
    cat1 = "writing score" 
    cat2 = "reading score"
    cat3 = "math score"
    with col6:
        gender = st.radio("Gender",options=["male","female"])

    with col7:
        race = st.radio("Race/ethnicity",options=["group A","group B","group C","group D","group E"])

    with col8:
        parental_level_education = st.radio("Parental level education", options=["high school","associate's degree", "some college", "bachelor's degree"])

    with col9: 
        lunch = st.radio("Lunch",options=["standard", "free/reduced"])

    with col10:
        test_preparation_course = st.radio("Test preparation course",options=["none","completed"])
    
    # dynamically choosing the slider based on the score we want to predict 
    with col11 :
        if  what_predict == "math score":
            writing_score= st.slider("What writing score ?", min_value=0,max_value=100)
            reading_score = st.slider("What reading score ?", min_value=0,max_value=100)
            return gender,race,parental_level_education,lunch,test_preparation_course,reading_score,writing_score,cat1, cat2
        elif what_predict == "reading score":
            writing_score= st.slider("What writing score ?", min_value=0,max_value=100)
            math_score = st.slider("What math score ?", min_value=0,max_value=100)
            return gender,race,parental_level_education,lunch,test_preparation_course,math_score,writing_score,cat3,cat1
        elif what_predict == "writing score":
            math_score= st.slider("What math score ?", min_value=0,max_value=100)
            reading_score = st.slider("What reading score ?", min_value=0,max_value=100)
            return gender,race,parental_level_education,lunch,test_preparation_course,math_score,reading_score,cat3,cat2

# displaying all results of the reports of each models training result | test result
st.set_page_config(page_title="Machine learning Experiment",layout="wide")
st.title("Student Performance Prediction and Research")
st.warning("Model Performance Reports:")
col1,col2 = st.columns(2)
with col1:
    st.subheader("All scores of Linear Regression with 3 targets at the time")
    for report in reports["all_scores_linear"]:
        st.code(report)

with col2:
    st.subheader("All scores of RandomForest Regressor with 3 targets at the time")
    for report in reports["all_scores_randomForest"]:
        st.code(report)

st.warning("Individual Score for one target at a time:")
col3, col4, col5 = st.columns(3)
with col3:
    st.subheader("Math Score")
    for report in reports["math score"]:
        st.code(report)

with col4:
    st.subheader("Reading Score")
    for report in reports["reading score"]:
        st.code(report)

with col5:
    st.subheader("Writing Score")
    for report in reports["writing score"]:
        st.code(report)

# the result of our research as a descriptive text
st.warning("Research result:")
st.code("""
        We observe that predictive performance improves significantly when individual scores are modeled separately, 
        rather than attempting to predict all outcomes simultaneously. This improvement occurs because the current dataset 
        lacks meaningful correlations between the input features and the target variables making accurate 
        prediction inherently challenging. In such cases the model cannot establish reliable relationships between the inputs
        and outcomes leading to limited predictive power. For instance, including irrelevant features such as a "lunch" 
        variable adds no meaningful signal, as student performance is unlikely to be influenced by meal choices.
        Instead, performance is more strongly associated with historical academic indicators, such as prior test results 
        or mock exam scores. If such relevant data were available the model could leverage demonstrated 
        academic trends to generate more accurate and justifiable predictions. Therefore the feasibility of effective 
        prediction depends critically on the inclusion of informative, performance related features.""")


st.subheader("Playground to test prediction")
what_predict = st.radio("What do you wanna Predict ?", options=["math score", "reading score", "writing score"])
gender,race,parental_level_education,lunch,test_preparation_course, score_1,score_2,categorie_1,categorie_2 = open_playground(what_predict)

#################################################
# PREDICTION SEQUENCE
#################################################

# transforming the chosen values into the trained data format by using the encoding 
data = [[gender,race,parental_level_education,lunch,test_preparation_course,score_1,score_2]]
X = pd.DataFrame(data= data,columns=["gender","race/ethnicity","parental level of education","lunch","test preparation course",categorie_1,categorie_2])
encoded_data = pd.get_dummies(X)
model_linear, model_randomForest = load_model(what_predict)
expected_columns = model_linear.feature_names_in_
encoded_data = encoded_data.reindex(columns=expected_columns, fill_value=0)

# predicting the outcome and displaying it
pred_linear = model_linear.predict(encoded_data)
pred_randomForest = model_randomForest.predict(encoded_data)
st.warning(f"Linear Regression predicts: {pred_linear} . Random Forest predicts: {pred_randomForest}")