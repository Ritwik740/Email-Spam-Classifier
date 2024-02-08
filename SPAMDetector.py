import pickle
import streamlit as st
from gtts import gTTS
import pygame

# Set the SDL_AUDIODRIVER environment variable to "dummy" to use the dummy audio driver
import os
os.environ['SDL_AUDIODRIVER'] = 'dummy'

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
            pygame.mixer.init()  # Initialize Pygame's mixer module
            pygame.mixer.music.load("welcome.mp3")  # Load the sound file
            pygame.mixer.music.play()  # Play the sound
        else:
            st.success("This is a ham mail")

if __name__ == "__main__":
    main()
