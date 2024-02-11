import pickle
import streamlit as st

# Load the model and vectorizer
model = pickle.load(open("spam.pkl", "rb"))
cv = pickle.load(open("vectorizer.pkl", "rb"))


def main():
  page_bg_img = '''
  <style>

[class="main st-emotion-cache-uf99v8 ea3mdgi8"]{
  background-image: url("https://images.unsplash.com/photo-1562813733-b31f71025d54?q=80&w=1769&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
  background-size: cover;
  }
  </style>
  '''

  st.markdown(page_bg_img, unsafe_allow_html=True)
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

    else:
      st.success("This is a ham mail")
  st.write(
      "To know more about Spamming and spam emails, you can visit the following links:"
  )
  st.write(
      "[Spamming](https://en.wikipedia.org/wiki/Spamming) | [Spam Emails](https://en.wikipedia.org/wiki/Email_spam)"
  )

  st.write(
      "Contact: +91 6307463343 | [Linkedin](https://linkedin.com/in/saurabh-mishra-638276278) | [Github](https://github.com/Saurabh6307)"
  )


if __name__ == "__main__":
  main()
