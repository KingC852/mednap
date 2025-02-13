import pickle
import pandas as pd

# Load the saved Random Forest model
with open('/Users/yiuyiucc/Documents/Med*nap/random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Check model attributes
# print("Number of trees:", model.n_estimators)
# print("Maximum depth of trees:", model.max_depth)
# print("Feature importances:", model.feature_importances_)
# print("Number of features in model:", model.n_features_in_)
# print("Class labels:", model.classes_)
# print("Out-of-bag score:", model.oob_score)

# Column names for your features (ensure these match the ones used during training)
feature_names = [
    'AVNN', 'SDNN', 'MeanHR', 'StdHR', 'MinHR', 
    'MaxHR', 'RMSSD', 'NN50', 'pNN50'
]

# Example data to predict (ensure it matches the model's expected input format)
# new_data = [[858.6856617647059, 53.82285372421226, 70.14175319941805, 4.278961295457792, 60.71146245059288, 76.8, 59.31921472273524, 16, 47.05882352941177]] #my own ECG data
new_data = [[1384.5703125,
 34.8833811543795,
 43.36229610410521,
 1.0940846817984973,
 41.06951871657754,
 45.309734513274336,
 44.84359240298733,
 7,
 35.0]]

# Convert new_data into a pandas DataFrame with the correct feature names
new_data_df = pd.DataFrame(new_data, columns=feature_names)

# Make a prediction
prediction = model.predict(new_data_df)

# Print the prediction
print(prediction)