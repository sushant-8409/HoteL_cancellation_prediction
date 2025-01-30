
import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder="templates", static_folder="staticFiles")

# Load the model and label encoders
model = pickle.load(open('model.pkl', 'rb'))
label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Extract form data
#     data = {
#         'lead_time': request.form.get('lead_time', type=int, default=0),
#         'arrival_date_month': request.form.get('arrival_date_month', type=str),
#         'arrival_date_week_number': request.form.get('arrival_date_week_number', type=int, default=1),
#         'arrival_date_day_of_month': request.form.get('arrival_date_day_of_month', type=int, default=1),
#         'stays_in_weekend_nights': request.form.get('stays_in_weekend_nights', type=int, default=0),
#         'stays_in_week_nights': request.form.get('stays_in_week_nights', type=int, default=0),
#         'adults': request.form.get('adults', type=int, default=1),
#         'is_repeated_guest': request.form.get('is_repeated_guest', type=int, default=0),
#         'previous_cancellations': request.form.get('previous_cancellations', type=int, default=0),
#         'previous_bookings_not_canceled': request.form.get('previous_bookings_not_canceled', type=int, default=0),
#         'booking_changes': request.form.get('booking_changes', type=int, default=0),
#         'days_in_waiting_list': request.form.get('days_in_waiting_list', type=int, default=0),
#         'adr': request.form.get('adr', type=float, default=0.0),
#         'required_car_parking_spaces': request.form.get('required_car_parking_spaces', type=int, default=0),
#         'total_of_special_requests': request.form.get('total_of_special_requests', type=int, default=0),
#         'kids': request.form.get('kids', type=int, default=0),
#         'hotel': request.form.get('hotel'),
#         'meal': request.form.get('meal'),
#         'country': request.form.get('country'),
#         'market_segment': request.form.get('market_segment'),
#         'distribution_channel': request.form.get('distribution_channel'),
#         'assigned_room_type': request.form.get('assigned_room_type'),
#         'deposit_type': request.form.get('deposit_type'),
#         'customer_type': request.form.get('customer_type')
#     }

#     # Convert to DataFrame
#     user_input_df = pd.DataFrame(data, index=[0])

#     # # Encode categorical columns
#     # for column, encoder in label_encoders.items():
#     #     if column in user_input_df.columns:
#     #         user_input_df[column] = encoder.transform(user_input_df[column])

#     # Make prediction
#     # try:
#     #     prediction = model.predict(user_input_df)
#     #     prediction_text = "Not Canceled" if prediction[0] == 0 else "Canceled"
#     # except Exception as e:
#     #     prediction_text = f"Error: {str(e)}"



#     # Encode categorical columns
#     for column, encoder in label_encoders.items():
#         if column in user_input_df.columns:
#             if user_input_df[column].dtype == 'object':
#                 user_input_df[column] = encoder.transform(user_input_df[column])
#             else:
#                 user_input_df[column] = user_input_df[column].astype(float)
    
#     # Ensure all columns are numeric
#     user_input_df = user_input_df.apply(pd.to_numeric, errors='coerce').fillna(0)

#     # Make prediction
#     try:
#         prediction = model.predict(user_input_df)
#         prediction_text = "Not Canceled" if prediction[0] == 0 else "Canceled"
#     except Exception as e:
#         prediction_text = f"Error: {str(e)}"

#     return render_template('prediction.html', prediction_text=prediction_text)




@app.route('/predict', methods=['POST'])
def predict():
    # Extract JSON data from request
    data = request.json  # Data sent from the frontend as JSON
    
    # Convert data into a DataFrame
    user_input_df = pd.DataFrame(data, index=[0])

    # Encode categorical columns
    for column, encoder in label_encoders.items():
        if column in user_input_df.columns:
            if user_input_df[column].dtype == 'object':
                user_input_df[column] = encoder.transform(user_input_df[column])
            else:
                user_input_df[column] = user_input_df[column].astype(float)
    
    # Ensure all columns are numeric
    user_input_df = user_input_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Make prediction
    try:
        prediction = model.predict(user_input_df)
        prediction_text = "Not Canceled" if prediction[0] == 0 else "Canceled"
    except Exception as e:
        prediction_text = f"Error: {str(e)}"

    # Return the result as JSON
    return jsonify({'prediction': prediction_text})





@app.route('/prediction')
def prediction():
    return render_template('prediction.html')  


@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/graph')
def graph():
    return render_template('graph.html')

# below code is written in wspi.py for deployment

if __name__ == "__main__":
    app.run(debug=True)