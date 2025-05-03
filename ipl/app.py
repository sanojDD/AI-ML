import streamlit as st
import pickle
import pandas as pd
import pickle
from sklearn.compose._column_transformer import _RemainderColsList

# Define the missing class (for scikit-learn <1.4 compatibility)
if not hasattr(pickle, '_RemainderColsList'):
    pickle._RemainderColsList = _RemainderColsList

pipe = pickle.load(open('pipe_new.pkl', 'rb'))

teams = ['Sunrisers Hyderabad',
         'Mumbai Indians',
         'Royal Challengers Bangalore',
         'Kolkata Knight Riders',
         'Kings XI Punjab',
         'Chennai Super Kings',
         'Rajasthan Royals',
         'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

pipe = pickle.load(open('pipe.pkl', 'rb'))
st.title('IPL Win Predictor')

# Update to st.columns
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select host city', sorted(cities))

target = st.number_input('Target', min_value=1)

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score', min_value=0)
with col4:
    overs = st.number_input('Overs completed', min_value=0.0)
with col5:
    wickets = st.number_input('Wickets out', min_value=0, max_value=10)

if st.button('Predict Probability'):
    # Calculate runs left, balls left, and other stats
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets  # Store the number of wickets left, not overwrite the variable
    crr = score / overs if overs > 0 else 0  # Handle division by zero
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0  # Avoid division by zero

    input_df = pd.DataFrame({
    'batting_team': [batting_team],
    'bowling_team': [bowling_team],
    'city': [selected_city],
    'runs_left': [runs_left],
    'balls_left': [balls_left],
    'wickets_left': [wickets],  # ✅ Rename it to 'wickets_left'
    'total_runs_x': [target],
    'crr': [crr],
    'rrr': [rrr]
})


    # Make prediction
    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]

    # Display results
    st.header(batting_team + " - " + str(round(win * 100)) + "%")
    st.header(bowling_team + " - " + str(round(loss * 100)) + "%")
