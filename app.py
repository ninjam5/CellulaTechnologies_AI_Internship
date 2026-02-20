from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_pipeline.pkl")
pipeline = joblib.load(MODEL_PATH)


def validate_input(form_data: dict) -> dict:
    required_fields = [
        "number_of_adults", "number_of_children", "number_of_weekend_nights",
        "number_of_week_nights", "type_of_meal", "car_parking_space",
        "room_type", "lead_time", "market_segment_type", "repeated",
        "p_c", "p_not_c", "average_price", "special_requests",
        "date_of_reservation",
    ]

    missing = [f for f in required_fields if not form_data.get(f)]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    valid_meals = ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"]
    valid_rooms = [f"Room_Type {i}" for i in range(1, 8)]
    valid_segments = ["Aviation", "Complementary", "Corporate", "Offline", "Online"]

    if form_data["type_of_meal"] not in valid_meals:
        raise ValueError(f"Invalid meal type: {form_data['type_of_meal']}")
    if form_data["room_type"] not in valid_rooms:
        raise ValueError(f"Invalid room type: {form_data['room_type']}")
    if form_data["market_segment_type"] not in valid_segments:
        raise ValueError(f"Invalid market segment: {form_data['market_segment_type']}")

    return form_data


def preprocess_input(form_data: dict) -> pd.DataFrame:
    date_str = form_data.get("date_of_reservation")
    reservation_month = pd.to_datetime(date_str).month

    data = {
        "number of adults": int(form_data["number_of_adults"]),
        "number of children": int(form_data["number_of_children"]),
        "number of weekend nights": int(form_data["number_of_weekend_nights"]),
        "number of week nights": int(form_data["number_of_week_nights"]),
        "type of meal": form_data["type_of_meal"],
        "car parking space": int(form_data["car_parking_space"]),
        "room type": form_data["room_type"],
        "lead time": int(form_data["lead_time"]),
        "market segment type": form_data["market_segment_type"],
        "repeated": int(form_data["repeated"]),
        "P-C": int(form_data["p_c"]),
        "P-not-C": int(form_data["p_not_c"]),
        "average price": float(form_data["average_price"]),
        "special requests": int(form_data["special_requests"]),
        "reservation_month": reservation_month,
    }

    return pd.DataFrame([data])


def get_prediction(input_df: pd.DataFrame) -> dict:
    prediction = pipeline.predict(input_df)[0]
    probabilities = pipeline.predict_proba(input_df)[0]

    return {
        "prediction": "Canceled" if prediction == 1 else "Not Canceled",
        "confidence": round(float(max(probabilities)) * 100, 2),
        "cancel_probability": round(float(probabilities[1]) * 100, 2),
        "not_cancel_probability": round(float(probabilities[0]) * 100, 2),
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        form_data = validate_input(request.form.to_dict())
        input_df = preprocess_input(form_data)
        result = get_prediction(input_df)
        return jsonify({"success": True, **result})
    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
