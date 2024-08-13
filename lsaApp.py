# Basic libraries
import pandas as pd
import numpy as np
# Loading model files
import joblib
# Ui and logic library
import streamlit as sl

#################################
# lodaing model files
ohe=joblib.load("ohe.pkl")
ss=joblib.load("ss.pkl")
rfr=joblib.load("lsa_rfr.pkl")

###################################
# UI code
sl.header("Loan Amount Sanction Prediction...")
sl.write("This app built on the below features..")
df=pd.read_csv("x_loan_sanction_amount.csv")
sl.write(df.head(5))
sl.subheader("Enter The Application Details To Estimate Loan Sanction Amount")
sl.image("logo.jpg")

# form type input
col1,col2,col3,col4 = sl.columns(4)

with col1:
    gender=sl.selectbox("Gender",df.Gender.unique())
with col2:
    age=sl.number_input("Age")
with col3:
    income=sl.number_input("income(USD)")
with col4:
    incomeStabi=sl.selectbox("Income Stability",df["Income Stability"].unique())


col5,col6,col7,col8 = sl.columns(4)
with col5:
    prof=sl.selectbox("Profession",df.Profession.unique())
with col6:
    toe=sl.selectbox("Type of Employment",df["Type of Employment"].unique())
with col7:
    loc=sl.selectbox("Location",df.Location.unique())
with col8:
    loanAmoReq=sl.number_input("Loan Amount Request (USD)")

col9,col10,col11,col12 = sl.columns(4)
with col9:
    cle=sl.number_input("Current Loan Expenses(USD)")
with col10:
    t1=sl.selectbox("Expense Type 1",df["Expense Type 1"].unique())
with col11:
    t2=sl.selectbox("Expense Type 2",df["Expense Type 2"].unique())
with col12:
    depend=sl.number_input("Dependents")

col13,col14,col15,col16=sl.columns(4)
with col13:
    cs=sl.number_input("credit score")
with col14:
    nod=sl.number_input("Number of Defaults")
with col15:
    hac=sl.selectbox("Has Active Credit Card",df["Has Active Credit Card"].unique())
with col16:
    pa=sl.number_input("Property Age")

col17,col18,col19,col20 =sl.columns(4)
with col17:
    pt=sl.selectbox("Property Type",df["Property Type"].unique())
with col18:
    ploc=sl.selectbox("Property Location",df["Property Location"].unique())
with col19:
    coApp=sl.number_input("Co-Applicant")
with col20:
    pp=sl.number_input("Property Price")

# Logic Code
if sl.button("Estimate"):
    row=pd.DataFrame([[gender,age,income,incomeStabi,prof,toe,loc,loanAmoReq,cle,t1,t2,depend,cs,nod,hac,pa,pt,ploc,coApp,pp,]],columns=df.columns)
    sl.write("given data")
    sl.dataframe(row)
    
    #binary encoding
    row["Gender"].replace({'m':1,'f':0},inplace=True)
    row["Income Stability"].replace({'high':1,'low':0},inplace=True)
    row["Expense Type 1"].replace({'y':1,'n':0},inplace=True)
    row["Expense Type 2"].replace({'y':1,'n':0},inplace=True)
    
    #odinal enoding
    row["Property Type"].replace("residential",1,inplace=True)
    row["Property Type"].replace("commercial",2,inplace=True)
    row["Property Type"].replace("industrial",3,inplace=True)
    row["Property Type"].replace("agriculture",4,inplace=True)
    
    #onehotencoding
    rowohe=ohe.transform(row[["Profession","Type of Employment","Location","Has Active Credit Card","Property Location"]]).toarray()
    rowohe=pd.DataFrame(rowohe, columns=ohe.get_feature_names_out())
    row=row.drop(["Profession","Type of Employment","Location","Has Active Credit Card","Property Location"],axis=1)
    row=pd.concat([row,rowohe],axis=1)
    
    #scaling
    row[["Age","Income (USD)","Loan Amount Request (USD)","Current Loan Expenses (USD)","Dependents","Credit Score","No. of Defaults","Property Age","Co-Applicant","Property Price"]]=ss.transform(row[["Age","Income (USD)","Loan Amount Request (USD)","Current Loan Expenses (USD)","Dependents","Credit Score","No. of Defaults","Property Age","Co-Applicant","Property Price"]])
    
    loan=round(rfr.predict(row)[0])
    
    sl.write(f"predicted loan sanction amount:$ {loan}")