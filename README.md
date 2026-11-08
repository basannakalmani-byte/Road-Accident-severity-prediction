# Road Accident Severity Prediction Using Machine Learning

## Project overview
This Diploma-level educational project predicts an accident severity category (`Minor`, `Moderate`, or `Severe`) from road and environmental conditions using a Random Forest classifier.

**Important:** The included CSV is a synthetically generated demonstration dataset, not official police, hospital, or government accident data. Model accuracy on this dataset does not represent real-world predictive performance. Do not use the application for emergency response, insurance, legal, or operational decisions.

## Project files
- `app.py` — Streamlit web application
- `train_model.py` — trains and evaluates the machine-learning model
- `road_accident_dataset.csv` — synthetic demonstration dataset
- `requirements.txt` — Python dependencies

## Requirements
- Python 3.10 or newer
- Internet connection for the first-time package installation

## Run on Windows PowerShell
Open PowerShell in the extracted project folder and run:

```powershell
python -m pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

Open the URL shown in the terminal, normally `http://localhost:8501`.

## Input features
Road type, weather, lighting, traffic density, road surface, speed limit, vehicles involved, driver age, and alcohol involvement.

## Output
The model displays a predicted severity category and class probabilities.

## Machine-learning workflow
1. Load the CSV dataset.
2. Separate input features and target label.
3. Convert categorical features using One-Hot Encoding.
4. Train a Random Forest classifier.
5. Evaluate using a held-out test set.
6. Save the trained pipeline.
7. Load the pipeline in the Streamlit application and predict from user inputs.

## Suggested future improvements
Use a verified, legally usable real-world dataset; document its source and limitations; handle missing values; test on data from a separate time period or region; and assess fairness and calibration before any real-world use.
