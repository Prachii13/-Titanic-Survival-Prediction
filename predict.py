import joblib
import pandas as pd

model = joblib.load("model.pkl")

sample = {
    "Pclass": [2],
    "Sex": [1],            # 1: male, 0: female
    "Age": [30],
    "SibSp": [0],
    "Parch": [0],
    "Fare": [20],
    "Embarked": [2]        # 0: C, 1: Q, 2: S
}

df = pd.DataFrame(sample)
pred = model.predict(df)[0]
print("🚢 Survival Prediction:", "Survived ✅" if pred else "Did not survive ❌")
