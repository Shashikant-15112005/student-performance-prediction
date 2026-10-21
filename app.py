import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "student-mat.csv"
MODEL_PATH = BASE_DIR / "student_performance_model.pkl"

st.title("Student Performance Predictor")
st.write("Enter student information to estimate the final mathematics grade using the trained Random Forest model.")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH, sep=";")


try:
    df = load_dataset()
    model = load_model()
except FileNotFoundError:
    st.error("Required file not found. Keep student-mat.csv and student_performance_model.pkl in the same folder as app.py.")
    st.stop()
except Exception as exc:
    st.error(f"Could not load the model or dataset: {exc}")
    st.stop()

if "G3" not in df.columns:
    st.error("The dataset does not contain the expected G3 target column.")
    st.stop()

# Match the feature columns used to train the model.
X = df.drop(columns=["G3"])

# Original dataset values are retained as keys; labels are user-friendly.
categorical_info = {
    "school": {"label": "School", "help": "Which school does the student attend?", "options": {"GP": "Gabriel Pereira (GP)", "MS": "Mousinho da Silveira (MS)"}},
    "sex": {"label": "Sex", "help": "Select the student's recorded sex.", "options": {"F": "Female", "M": "Male"}},
    "address": {"label": "Home Address Type", "help": "Is the student's home in an urban or rural area?", "options": {"U": "Urban", "R": "Rural"}},
    "famsize": {"label": "Family Size", "help": "How many people are in the student's family?", "options": {"LE3": "3 or fewer members", "GT3": "More than 3 members"}},
    "Pstatus": {"label": "Parents' Living Arrangement", "help": "Do the student's parents live together?", "options": {"T": "Living together", "A": "Living apart"}},
    "Mjob": {"label": "Mother's Occupation", "help": "Choose the category that best describes the mother's occupation.", "options": {"teacher": "Teacher", "health": "Healthcare", "services": "Services", "at_home": "At home", "other": "Other"}},
    "Fjob": {"label": "Father's Occupation", "help": "Choose the category that best describes the father's occupation.", "options": {"teacher": "Teacher", "health": "Healthcare", "services": "Services", "at_home": "At home", "other": "Other"}},
    "reason": {"label": "Reason for Choosing the School", "help": "What was the main reason for choosing this school?", "options": {"home": "Close to home", "reputation": "School reputation", "course": "Course preference", "other": "Other"}},
    "guardian": {"label": "Primary Guardian", "help": "Who is primarily responsible for the student's care?", "options": {"mother": "Mother", "father": "Father", "other": "Other"}},
    "schoolsup": {"label": "Extra Educational Support", "help": "Does the student receive extra educational support from school?", "options": {"yes": "Yes", "no": "No"}},
    "famsup": {"label": "Family Educational Support", "help": "Does the student receive educational support from family members?", "options": {"yes": "Yes", "no": "No"}},
    "paid": {"label": "Paid Extra Classes", "help": "Does the student take paid extra classes for the subject?", "options": {"yes": "Yes", "no": "No"}},
    "activities": {"label": "Extracurricular Activities", "help": "Does the student participate in extracurricular activities?", "options": {"yes": "Yes", "no": "No"}},
    "nursery": {"label": "Preschool Attendance", "help": "Did the student attend nursery school before primary school?", "options": {"yes": "Yes", "no": "No"}},
    "higher": {"label": "Plans for Higher Education", "help": "Does the student want to pursue higher education?", "options": {"yes": "Yes", "no": "No"}},
    "internet": {"label": "Internet Access at Home", "help": "Does the student have internet access at home?", "options": {"yes": "Yes", "no": "No"}},
    "romantic": {"label": "Romantic Relationship", "help": "Does the student report being in a romantic relationship?", "options": {"yes": "Yes", "no": "No"}},
    "Medu": {"label": "Mother's Education Level", "help": "Highest education level completed by the student's mother.", "options": {0: "No formal education", 1: "Primary education (up to 4th grade)", 2: "5th to 9th grade", 3: "Secondary education", 4: "Higher education (college/university)"}},
    "Fedu": {"label": "Father's Education Level", "help": "Highest education level completed by the student's father.", "options": {0: "No formal education", 1: "Primary education (up to 4th grade)", 2: "5th to 9th grade", 3: "Secondary education", 4: "Higher education (college/university)"}},
    "famrel": {"label": "Family Relationship Quality", "help": "How would the student rate relationships within their family?", "options": {1: "Very bad", 2: "Bad", 3: "Average", 4: "Good", 5: "Excellent"}},
    "goout": {"label": "Going Out with Friends", "help": "How often does the student go out with friends?", "options": {1: "Very rarely", 2: "Rarely", 3: "Sometimes", 4: "Often", 5: "Very often"}},
    "Dalc": {"label": "Workday Alcohol Consumption", "help": "Self-reported alcohol consumption on weekdays, from very low to very high.", "options": {1: "Very low", 2: "Low", 3: "Moderate", 4: "High", 5: "Very high"}},
    "Walc": {"label": "Weekend Alcohol Consumption", "help": "Self-reported alcohol consumption on weekends, from very low to very high.", "options": {1: "Very low", 2: "Low", 3: "Moderate", 4: "High", 5: "Very high"}},
    "health": {"label": "Self-Rated Health", "help": "How would the student rate their current health?", "options": {1: "Very poor", 2: "Poor", 3: "Fair", 4: "Good", 5: "Very good"}},
}

numeric_info = {
    "age": {"label": "Student's Age", "help": "Age in years."},
    "traveltime": {"label": "Travel Time to School", "help": "1 = under 15 min, 2 = 15–30 min, 3 = 30 min–1 hour, 4 = over 1 hour (one-way)."},
    "studytime": {"label": "Weekly Study Time", "help": "1 = under 2 hours, 2 = 2–5 hours, 3 = 5–10 hours, 4 = over 10 hours per week."},
    "failures": {"label": "Past Class Failures", "help": "Number of past class failures recorded (dataset values are capped at 4)."},
    "absences": {"label": "School Absences", "help": "Number of school absences recorded in the dataset."},
    "G1": {"label": "First Period Grade (G1)", "help": "Grade from the first grading period, on a 0–20 scale."},
    "G2": {"label": "Second Period Grade (G2)", "help": "Grade from the second grading period, on a 0–20 scale."},
    "freetime": {"label": "Free Time After School", "help": "1 = very low, 2 = low, 3 = moderate, 4 = high, 5 = very high."},
}

st.info("Prediction scenario: after the first and second grading periods. The model also uses absences, so enter only absences known at the time of prediction.")
st.caption("Predictions are estimates, not guaranteed grades. This model was trained on students from two Portuguese schools; results may differ in other education systems.")

st.subheader("Student Information")

with st.form("student_form"):
    student_data = {}
    left_col, right_col = st.columns(2)

    for i, column in enumerate(X.columns):
        series = X[column]
        target_col = left_col if i % 2 == 0 else right_col

        with target_col:
            if column in categorical_info:
                info = categorical_info[column]
                options = list(info["options"].keys())
                selected = st.selectbox(
                    info["label"],
                    options=options,
                    format_func=lambda value, opts=info["options"]: opts[value],
                    help=info["help"],
                    key=f"input_{column}",
                )
                student_data[column] = selected
            else:
                info = numeric_info.get(
                    column,
                    {"label": column.replace("_", " ").capitalize(), "help": "Enter the value for this field."},
                )
                student_data[column] = st.number_input(
                    info["label"],
                    min_value=int(series.min()),
                    max_value=int(series.max()),
                    value=int(series.median()),
                    step=1,
                    help=info["help"],
                    key=f"input_{column}",
                )

    submitted = st.form_submit_button("Predict Final Grade", type="primary", use_container_width=True)

if submitted:
    try:
        student_df = pd.DataFrame([student_data], columns=X.columns)
        prediction = float(model.predict(student_df)[0])
        prediction = max(0.0, min(20.0, prediction))

        st.divider()
        st.subheader("Prediction Result")
        st.metric("Estimated Final Mathematics Grade", f"{prediction:.2f} / 20")
        st.progress(prediction / 20.0)
        st.caption("The progress bar shows the predicted grade relative to 20. It is not a probability or a measure of prediction confidence.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
