# HeartDiseasePredict

A local website that estimates heart-disease risk from a saved k-nearest neighbors model. You enter the clinical fields, and the server encodes them into the same 15 columns the model was trained on, scales them, and returns the vote of the five nearest patients.

This is a class-project estimate, not a medical diagnosis.

## Run it

```bash
python -m pip install -r requirements.txt
python server.py
```

Open [http://127.0.0.1:5050](http://127.0.0.1:5050).

The page includes two example records from `heart.csv`. The lower-risk example is the first row of the dataset. The higher-risk example is an asymptomatic patient with exercise-induced angina and a flat ST slope.

## How a prediction is made

`POST /predict` accepts the original fields (age, sex, chest pain, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, max heart rate, exercise angina, oldpeak, and ST slope). `server.py` turns those into the columns stored in `columns.pkl`:

| Form value | Encoded columns |
| --- | --- |
| Sex `M` / `F` | `Sex_M` is 1 for male. Female is the reference (0). |
| Chest pain `ATA`, `NAP`, `TA`, `ASY` | `ChestPainType_ATA`, `ChestPainType_NAP`, `ChestPainType_TA`. Asymptomatic (`ASY`) is the reference. |
| Resting ECG `Normal`, `ST`, `LVH` | `RestingECG_Normal`, `RestingECG_ST`. `LVH` is the reference. |
| Exercise angina `Y` / `N` | `ExerciseAngina_Y`. No angina is the reference. |
| ST slope `Flat`, `Up`, `Down` | `ST_Slope_Flat`, `ST_Slope_Up`. Downsloping is the reference. |

Numeric fields (`Age`, `RestingBP`, `Cholesterol`, `FastingBS`, `MaxHR`, `Oldpeak`) are passed through as numbers. `FastingBS` is 1 when fasting blood sugar is above 120 mg/dL.

That row is scaled with `scaler.pkl` and scored with `KNN_heart.pkl` (`KNeighborsClassifier`, 5 neighbors, uniform weights). The response includes the predicted class, the share of neighbors labeled with heart disease, and the encoded feature row.

The model and scaler were saved with scikit-learn 1.6.1. A newer scikit-learn install can still load them, and it may print a version warning.

## Files

| File | Purpose |
| --- | --- |
| `index.html`, `style.css`, `app.js` | The form and result page |
| `server.py` | Serves the page and the `/predict` endpoint |
| `KNN_heart.pkl` | Trained 5-neighbor classifier |
| `scaler.pkl` | `StandardScaler` fit on the training features |
| `columns.pkl` | Feature names, in model order |
| `heart.csv` | Source data, 918 rows |
| `requirements.txt` | Python dependencies |

## Dataset fields

`heart.csv` columns are `Age`, `Sex`, `ChestPainType`, `RestingBP`, `Cholesterol`, `FastingBS`, `RestingECG`, `MaxHR`, `ExerciseAngina`, `Oldpeak`, `ST_Slope`, and `HeartDisease`. `HeartDisease` is the label and is not sent to the model.
