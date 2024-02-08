import pickle
import streamlit as st
import pygame

# Load the model and vectorizer
model = pickle.load(open("spam.pkl", "rb"))
cv = pickle.load(open("vectorizer.pkl", "rb"))

# Initialize Pygame's mixer module
pygame.mixer.init()

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
            # Load and play the sound file
            pygame.mixer.music.load("welcome.mp3")
            pygame.mixer.music.play()
        else:
            st.success("This is a ham mail")

if __name__ == "__main__":
    main()

