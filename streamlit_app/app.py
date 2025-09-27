import streamlit as st
import joblib
import os

#path to file
#PKL_FILE_PATH = 'E:/Downloads/week1/Assignment_1/sentiment_model.pk1'
PKL_FILE_PATH = os.path.join(os.path.dirname(__file__), 'sentiment_model.pk1')

@st.cache_data
def load_model(path):
    """Loads the model from the given path using joblib."""
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {path}")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    st.title("My Streamlit Movie Review App")

    # Load the model
    model = load_model(PKL_FILE_PATH)

    if model is not None:
        st.success("Model loaded successfully!")

        # Add a text area for user input
        user_input = st.text_area("Enter text here:", "")

        # Add an Analyze button
        if st.button("Analyze"):
            if user_input:
                # Use the loaded model to make a prediction
                try:
                    prediction_proba = model.predict_proba([user_input]) 
                    st.write("Prediction Probabilities:", prediction_proba)
                    if prediction_proba[0][1] >  prediction_proba[0][0]:
                        st.markdown("<span style='color: green;'>Prediction: Positive</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("<span style='color: red;'>Prediction: Negative</span>", unsafe_allow_html=True)
                except AttributeError:
                    st.error("Error: The loaded model does not have a .predict_proba() method.")
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
            else:
                st.warning("Please enter some text to analyze.")

    else:
        st.error("Model could not be loaded. Please check the PKL_FILE_PATH.")


if __name__ == "__main__":
    main()

