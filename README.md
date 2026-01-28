
```
# 🌾 Crop Recommendation System (ML + FastAPI)

An **end-to-end Machine Learning–powered Crop Recommendation System** that suggests the most suitable crop based on soil nutrients and environmental conditions.  
The model is deployed as a **FastAPI backend**, making it ready for real-world use.

---

##  Project Overview

Agricultural productivity depends on soil quality and climatic conditions.  
This project uses **Machine Learning (Random Forest Classifier)** to recommend the most suitable crop based on:

- Soil nutrients (NPK)
- Temperature and humidity
- Soil pH
- Rainfall

The trained ML model is exposed via a **REST API using FastAPI**.

---

##  Features

-  Multi-class crop recommendation (58+ crops)
-  Machine Learning model using Random Forest
-  FastAPI-based REST API
-  Input validation with Pydantic
-  Swagger UI for testing
-  Production-ready ML deployment
-  Clean and scalable project structure

---

##  Machine Learning Details

- **Algorithm**: Random Forest Classifier  
- **Dataset**: Crop Recommendation Dataset  
- **Number of Features**: 7  
- **Number of Classes**: 58  
- **Accuracy**: ~79%  
- **Cross-Validation Accuracy**: ~79%  

###  Input Features

| Feature | Description |
|-------|------------|
| Nitrogen | Nitrogen content in soil |
| Phosphorus | Phosphorus content in soil |
| Potassium | Potassium content in soil |
| temperature | Temperature (°C) |
| humidity | Relative humidity (%) |
| ph | Soil pH value |
| rainfall | Rainfall (mm) |

---

##  Tech Stack

- **Python 3.10**
- **scikit-learn**
- **pandas**
- **numpy**
- **FastAPI**
- **Uvicorn**
- **Joblib**

---

##  Project Structure

```

Crop_recommendation_model/
│── main.py
│── model.pkl
│── encoder.pkl
│── requirements.txt
│── README.md
│── venv/

````

---

##  Setup Instructions

### 1️ Create Virtual Environment
```bash
python -m venv venv
````

Activate it (Windows):

```bash
venv\Scripts\activate
```

---

### 2 Install Dependencies

```bash
pip install -r requirements.txt
```

---

###  Run the FastAPI Server

```bash
uvicorn main:app --reload
```

---

###  Open API Documentation

Open your browser and visit:

```
http://127.0.0.1:8000/docs
```

---

## 🔍 API Endpoints

### 🔹 Health Check

```
GET /
```

**Response**

```json
{
  "message": "Crop Recommendation API is running 🌱"
}
```

---

### 🔹 Predict Best Crop

```
POST /predict
```

**Request Body**

```json
{
  "Nitrogen": 48,
  "Phosphorus": 78,
  "Potassium": 27,
  "temperature": 23.6,
  "humidity": 69.3,
  "ph": 6.7,
  "rainfall": 80.9
}
```

**Response**

```json
{
  "recommended_crop": "Pea"
}
```
### 🔹 Top-3 Crop Recommendations with Confidence Scores

This endpoint returns the **top 3 most suitable crops** along with their **prediction confidence**, helping farmers make better-informed decisions.

---

####  Endpoint

---

####  Request Body
```json
{
  "Nitrogen": 48,
  "Phosphorus": 78,
  "Potassium": 27,
  "temperature": 23.6,
  "humidity": 69.3,
  "ph": 6.7,
  "rainfall": 80.9
}
{
  "top_3_recommendations": [
    {
      "crop": "Pea",
      "confidence": 0.41
    },
    {
      "crop": "Mango",
      "confidence": 0.32
    },
    {
      "crop": "Jowar",
      "confidence": 0.18
    }
  ]
}

---

##  Model Serialization

* Model is saved using **joblib**
* Compressed to reduce memory usage
* Loaded once at API startup for fast inference

---
