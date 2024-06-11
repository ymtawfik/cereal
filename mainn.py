import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Load the dataset
file_path = r"C:\AI project\cereal.csv"  # Use raw string or double backslashes for Windows paths
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())

# Prepare the data
X = data[['calories', 'protein', 'fat', 'sodium', 'fiber', 'carbo', 'sugars', 'potass', 'vitamins', 'shelf', 'weight', 'cups']]
y = data['rating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = DecisionTreeRegressor(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Load the dataset
file_path = r"C:\AI project\cereal.csv"  # Use the correct path
data = pd.read_csv(file_path)

# Prepare the data
X = data[['calories', 'protein', 'fat', 'sodium', 'fiber', 'carbo', 'sugars', 'potass', 'vitamins', 'shelf', 'weight', 'cups']]
y = data['rating']

# Train the model
model = DecisionTreeRegressor(random_state=42)
model.fit(X, y)

# Streamlit app
st.title('Cereal Rating Predictor')

# Display the image
image_path = "C:\AI project\ceraalll.png"  # Path to the image
st.image(image_path, caption='A bowl of colorful cereal', use_column_width=True)

st.write("""
### Predict the rating of a cereal based on its nutritional information.
""")

# User input
calories = st.number_input('Calories', min_value=0, value=70)
protein = st.number_input('Protein', min_value=0, value=4)
fat = st.number_input('Fat', min_value=0, value=1)
sodium = st.number_input('Sodium', min_value=0, value=130)
fiber = st.number_input('Fiber', min_value=0.0, value=10.0)
carbo = st.number_input('Carbohydrates', min_value=0.0, value=5.0)
sugars = st.number_input('Sugars', min_value=0, value=6)
potass = st.number_input('Potassium', min_value=-1, value=280)
vitamins = st.number_input('Vitamins', min_value=0, value=25)
shelf = st.number_input('Shelf', min_value=1, value=3)
weight = st.number_input('Weight (oz)', min_value=0.0, value=1.0)
cups = st.number_input('Cups', min_value=0.0, value=0.33)

# Prediction
input_data = pd.DataFrame({
    'calories': [calories],
    'protein': [protein],
    'fat': [fat],
    'sodium': [sodium],
    'fiber': [fiber],
    'carbo': [carbo],
    'sugars': [sugars],
    'potass': [potass],
    'vitamins': [vitamins],
    'shelf': [shelf],
    'weight': [weight],
    'cups': [cups]
})

prediction = model.predict(input_data)[0]
st.write(f'Predicted Rating: {prediction:.2f}')
