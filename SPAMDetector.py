import pickle
import streamlit as st
from playsound import playsound
import gi

model = pickle.load(open("spam.pkl", "rb"))
cv = pickle.load(open("vectorizer.pkl", "rb"))

def main():
    st.title("Email Spam Classification App")
    st.subheader("Built With Streamlit & Python")
    msg = st.text_input("Enter a Text: ")
    if st.button("Predict"):
        data = [msg]
        vect = cv.transform(data).toarray()
        prediction = model.predict(vect)
        result = prediction[0]
        if result == 1:
            st.error("This is a spam mail")
            playsound("welcome.mp3")  # Play the audio file
        else:
            st.success("This is a ham mail")

if __name__ == "__main__":
    main()
