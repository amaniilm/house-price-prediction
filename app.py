import pickle
import  streamlit as st

model=pickle.load(open("C:\\Users\\Lenovo\\Desktop\\langchain\\mlprojects\\house_price_prediction.pkl","rb"))
LE=pickle.load(open("C:\\Users\\Lenovo\\Desktop\\langchain\\mlprojects\\labelencoder.pkl","rb"))

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)
st.title("🏠 House Price Prediction")
Location=st.selectbox("Location",["Bangalore","Delhi","Gurgaon","Mumbai","Noida","Pune"])
Size=st.number_input("size",min_value=1)
Bedrooms=st.number_input("Bedrooms",min_value=1)
Bathrooms=st.number_input("Bathrooms",min_value=1)
Age=st.number_input("Age",min_value=1)

Location_encoded=LE.transform([Location])[0]
if st.button("predict_price"):
    prediction=model.predict([[Location_encoded,Size,Bedrooms,Bathrooms,Age]])
    st.success(f"predicted house  price:{prediction[0]:,.2f}")
    predicted_price=prediction[0]
    st.metric(
        label="estimated value(crores)",
        value=f"{predicted_price/10000000:.2f} Cr"
    )
    