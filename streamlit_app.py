import streamlit as st
from src.predict import predict_credit_risk

st.set_page_config(page_title='Alternative Credit Scoring', layout='centered')
st.title('AI-Powered Alternative Credit Scoring')
st.caption('Decision-support prototype; not a production lending decision engine.')

limit = st.number_input('Credit limit', min_value=1.0, value=50000.0)
age = st.number_input('Age', min_value=18, max_value=100, value=35)
pay = [st.number_input(f'PAY_{c}', value=0, step=1) for c in ['0','2','3','4','5','6']]
bills = [st.number_input(f'BILL_AMT{i}', value=0.0) for i in range(1,7)]
payments = [st.number_input(f'PAY_AMT{i}', value=0.0) for i in range(1,7)]
if st.button('Score customer'):
    data = {'LIMIT_BAL': limit, 'AGE': age}
    for c, v in zip(['0','2','3','4','5','6'], pay): data[f'PAY_{c}'] = v
    for i, v in enumerate(bills, 1): data[f'BILL_AMT{i}'] = v
    for i, v in enumerate(payments, 1): data[f'PAY_AMT{i}'] = v
    try:
        st.json(predict_credit_risk(data))
    except Exception as e:
        st.error(str(e))
