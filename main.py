from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

app = FastAPI()
templates = Jinja2Templates(directory="templates")

model = None
label_encoders = {}
imputer = None

def prepare_data():
    global model, label_encoders, imputer
    
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    
    df = df.drop(['id', 'smoking_status'], axis=1)
    
    categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type']
    
    for column in categorical_columns:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])
    
    imputer = SimpleImputer(strategy='mean')
    df['bmi'] = imputer.fit_transform(df[['bmi']])
    
    X = df.drop('stroke', axis=1)
    y = df['stroke']
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

prepare_data()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    gender: str = Form(...),
    age: float = Form(...),
    hypertension: int = Form(...),
    heart_disease: int = Form(...),
    ever_married: str = Form(...),
    work_type: str = Form(...),
    residence_type: str = Form(...),
    avg_glucose_level: float = Form(...),
    bmi: float = Form(...)
):
    try:
        input_data = pd.DataFrame({
            'gender': [gender],
            'age': [age],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            'ever_married': [ever_married],
            'work_type': [work_type],
            'Residence_type': [residence_type],
            'avg_glucose_level': [avg_glucose_level],
            'bmi': [bmi]
        })
        
        categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type']
        for column in categorical_columns:
            input_data[column] = label_encoders[column].transform(input_data[column])
        
        input_data['bmi'] = imputer.transform(input_data[['bmi']])
        
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]
        
        context = {
            "request": request,
            "prediction": int(prediction[0]),
            "probability": f"{probability * 100:.1f}%",
            "risk_level": "High" if prediction[0] == 1 else "Low",
            "input_data": {
                "Gender": gender,
                "Age": age,
                "Hypertension": "Yes" if hypertension == 1 else "No",
                "Heart Disease": "Yes" if heart_disease == 1 else "No",
                "Ever Married": ever_married,
                "Work Type": work_type,
                "Residence Type": residence_type,
                "Average Glucose Level": f"{avg_glucose_level:.1f}",
                "BMI": f"{bmi:.1f}"
            },
            "error": None
        }
        
        return templates.TemplateResponse("result.html", context)
    
    except Exception as e:
        context = {
            "request": request,
            "error": f"Prediction error: {str(e)}",
            "valid_values": {
                "gender": list(label_encoders['gender'].classes_),
                "ever_married": list(label_encoders['ever_married'].classes_),
                "work_type": list(label_encoders['work_type'].classes_),
                "Residence_type": list(label_encoders['Residence_type'].classes_)
            }
        }
        return templates.TemplateResponse("result.html", context)