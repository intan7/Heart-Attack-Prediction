# Heart Attack Prediction
> This is the machine learning model to predict the ***chance of a person having heart attack***.

# Descriptions
>Clinicians believe that prevention of heart attack is always better than curing 
it. After many years of research, scientists and clinicians discovered that, the 
probability of one’s getting heart attack can be determined by analysing the
patient’s age, gender, exercise induced angina, number of major vessels, chest 
pain indication, resting blood pressure, cholesterol level, fasting blood sugar, 
resting electrocardiographic results, and maximum heart rate achieved.

>Thus, this app will predict the ***chance of a person having heart attack*** by analysing below features: 
- Age
- Maximum heart rate achieved
- ST depression induced by exercise relative to rest
- Chest pain type
- Exercise induced angina
- Number of major vessels colored by flourosopy
- Thallium stress test

# 📙 Requirement
>> ![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)
>>
>>  - To run streamlit, you need to open to your cmd/terminal and go to the file directory containing your streamlit app file.
>>  - Then, type streamlit run streamlit.py file.
>>  - Your browser then will automatically open your app.
![alt text](https://github.com/intan7/Heart-Attack-Prediction/blob/main/static/run_streamlit.png)

# Streamlit App
![alt text](https://github.com/intan7/Heart-Attack-Prediction/blob/main/static/app.gif)

# Test App
Below is the data tested on the Streamlit App and 9/10 datasets able to predict the same output as True Output which gives the accuracy of the app as 90%.

![alt text](https://github.com/intan7/Heart-Attack-Prediction/blob/main/static/test.jpg)

# Results
The model accuracy is 0.7912087912087912 and this accuracy can be increased with more data collected, better features selection and model parameter tuning.

![alt text](https://github.com/intan7/Heart-Attack-Prediction/blob/main/static/cr.jpg)

In term of sensitivity and specificity, the number are 0.7045454545454546 and 0.8723404255319149 respectively; which means the model has ability to correctly identify patients with a low chance of heart attack for 70% and the ability to correctly identify people with high chance of heart attack for 87%.

![alt text](https://github.com/intan7/Heart-Attack-Prediction/blob/main/static/cm.jpg)



## Powered by
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)


## This project is able to successfully run thanks to
 >https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset
