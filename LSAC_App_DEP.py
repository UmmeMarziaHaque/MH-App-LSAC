import pickle


import pandas as pd
import numpy as np
import warnings
import streamlit as st
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

app_mode = st.sidebar.selectbox('Select Page',['Home','Predict_Depression'])

if app_mode=='Home':
    st.write("Based on the current published paper in HIS conference 2023, this web application has been built to predict depression using the Longitudinal Dataset of Australian Children (LSAC) with machine learning algortihms. If depression is not exist, this app will show the likelihood of having this mental illness with a given age.")

    st.write("This app is suitable for 6-14 years children and adolescent. For the parents, caregivers or teachers who are unsure whether their child or adolescent is affected with depression or not or whether they should consult regarding this issue with any mental health professional, they can use this app in order to get a primary idea about depression detection.") 


            
elif app_mode=='Predict_Depression':        
# loading the model
    #path = '/Users/marzia'
    #modelname = path + '/depressionmodel.pkl'
    #modelname = 'https://github.com/UmmeMarziaHaque/MH-Streamlit-App/blob/master/depressionmodel.pkl'
    loaded_model = pickle.load(open('/Users/marzia/Desktop/code/MH_App/LSAC_DEP_EXIST_updated.pkl', 'rb'))
    loaded_model1 = pickle.load(open('/Users/marzia/Desktop/code/MH_App/LSAC_DEP_AGE_updated1.pkl', 'rb'))
 
    age = st.sidebar.selectbox('Please enter age of the study child.', ("6","7","8","9","10","11","12","13","14","15"))
    
      
    df18cm1 = st.sidebar.selectbox('Does the study child have nervous condition?', ("Yes","No"))
    
    dhs08a7 = st.sidebar.selectbox('Does the study child have a difficulty or delay in any of the following areas compared to children of a similar age? Cope with emotions', ("Yes","No"))
    
    dse13a1 = st.sidebar.selectbox('Does the study child react strongly (cries or complains loudly to a disappointment or failure?', ("Never","Rarely","Half the time","Frequently","Always"))
    
    #dse13a4 = dse13a1
    dse13b1 = st.sidebar.selectbox('Does the study child not complete homework unless reminders are given?', ("Never","Rarely","Half the time","Frequently","Always"))
    
      
    dse13b4 = st.selectbox('Has difficulty completing assignments, homework or chores?',("Never","Rarely","Half the time","Frequently","Always"))
      
    dse03a3a = st.selectbox('Does the study child complain of headaches etc.??',("Not true","Somewhat true","Certainly true"))
    
    #dse03m3a = dse03a3a
     
    dse03a3b = st.selectbox('Does the study child often seem worried?',("Not true","Somewhat true","Certainly true"))
    
    #dse03t3b = dse03a3b
    
    #dse03m3b = dse03a3b
    
    dse03a3c = st.selectbox('Does the study child often seem unhappy?',("Not true","Somewhat true","Certainly true"))
    
    #dse03m3c = dse03a3c
    
    dse03a3d = st.selectbox('Does the study child often lose confidence?',("Not true","Somewhat true","Certainly true"))
    
    #dse03m3d =  dse03a3d
    
    dse03a3e = st.selectbox('Does the study study child have many fears?',("Not true","Somewhat true","Certainly true"))
    
    # dse03m3e = dse03a3e
    
    dse06a = st.selectbox('Does the study child become angry frequently?',("Not true","Somewhat true","Certainly true"))
    
    dgd04b1a = st.selectbox('Has the study child had a problem with this: afraid?',("Not sure","Never","Almost never", "Sometimes", "Often", "Almost always"))
    
    dgd04b1b = st.selectbox('Has the study child had a problem with this: feeling sad?',("Not sure","Never","Almost never", "Sometimes", "Often", "Almost always"))
    
    dgd04b1c = st.selectbox('Has the study child had a problem with this: feeling angry?',("Not sure","Never","Almost never", "Sometimes", "Often", "Almost always"))
    
    dgd04b1d = st.selectbox('Has the study child had a problem with this: trouble sleeping?',("Not sure","Never","Almost never", "Sometimes", "Often", "Almost always"))
    
    dgd04b1e = st.selectbox('Has the study child had a problem with this: worrying?',("Not sure","Never","Almost never", "Sometimes", "Often", "Almost always"))
    
    dgd04b2d = st.selectbox('Has the study child had a problem with this: unable to do what other children do?',("Not sure","Never","Almost never", "Sometimes", "Often", "Almost always"))
    
    dgd04b3g = st.selectbox('Has the study child had a problem with this: problems keeping up with other children/school activities?',("Not sure","Never","Almost never", "Sometimes", "Often", "Almost always"))
    
    dgd04b3b = st.selectbox('Does the study child has had a problem with this: School readiness: Problems missing days due to illness?',("Not sure","Never","Almost never", "Sometimes", "Often", "Almost always"))
    
    #dgd04b3b
    
    dpc49a3 = st.selectbox('Has the parent contracted school about attendance?',("Not at all","Once/twice","three/four times","More than four times"))
    
    
    
    dpc28a1 = st.selectbox('Does the study child not show excitement on the arrival of the parent?',("Over-excited and hard to settle for a long period more than a few hours","Over-excited and hard to settle for a short period","Relaxed and comfortable","Withdrawn, sad or restless for a short period","Withdrawn, sad or restless for a long period more than a few hours")) 
    
    dse03p1c = st.selectbox('Is the study child helpful if someone gets hurt?',("Not true","Somewhat true", "True"))
    
   #dse03p1c
    
   
 
    #dse13a4 = dse13a1
    if  dse13a1 == "Never":
        dse13a4 = "Never"
    elif dse13a1 == "Rarely":
        dse13a4 = "Rarely"
    elif dse13a1 == "Half the time":
        dse13a4 = "Half the time"
        
    elif dse13a1 == "Frequently":
        dse13a4 = "Frequently"
    else:
        dse13a4 = "Always"
    
    #dse03m3a = dse03a3a
    
    
    
    if dse03a3a == "Not true":
        dse03m3a = "Not true"
    elif dse03a3a == "Somewhat true":
        dse03m3a = "Somewhat true"
    else:
        dse03m3a = "Certainly true"
    #dse03t3b = dse03a3b
    
    if  dse03a3b == "Not true":
        dse03t3b = "Never"
    elif dse03a3b == "Somewhat true":
        dse03t3b = "Half the time"
    else:
        dse03t3b = "Always"
        
        
    #dse03m3b = dse03a3b
    if  dse03a3b == "Not true":
        dse03m3b = "Not true"
    elif dse03a3b == "Somewhat true":
        dse03m3b = "Somewhat true"
    else:
        dse03m3b = "Certainly true"
        
    #dse03m3c = dse03a3c   
    if  dse03a3c == "Not true":
        dse03m3c = "Not true"
    elif dse03a3c == "Somewhat true":
        dse03m3c = "Somewhat true"
    else:
        dse03m3c = "Certainly true"
        
        
    #dse03m3d =  dse03a3d   
    if  dse03a3d == "Not true":
        dse03m3d = "Not true"
    elif dse03a3d == "Somewhat true":
        dse03m3d = "Somewhat true"
    else:
        dse03m3d = "Certainly true"
        
        
    # dse03m3e = dse03a3e    
    if  dse03a3e == "Not true":
        dse03m3e = "Not true"
    elif dse03a3e == "Somewhat true":
        dse03m3e = "Somewhat true"
    else: 
        dse03m3e = "Certainly true"
        
    if  dgd04b1b == "Not sure":
        dgd04b1b = "Never"
        
    if  dgd04b1c == "Not sure":
        dgd04b1c = "Never"
        
    if  dgd04b1e == "Not sure":
        dgd04b1e = "Never"
        
        
 # Pre-processing user input   
    dataset = ["Yes", "No", "Never","Rarely","Half the time","Frequently","Always", "Not true","Somewhat true","Certainly true", "Not sure","Never","Almost Never", "Sometimes", "Often", "Almost always", "Not at all","Once/twice","three/four times","More than four times", "Over-excited and hard to settle for a long period more than a few hours","Over-excited and hard to settle for a short period","Relaxed and comfortable","Withdrawn, sad or restless for a short period","Withdrawn, sad or restless for a long period more than a few hours"]

# Create a dictionary to map the categorical values to their corresponding numeric values
    category_mapping = {"Yes": 1, "No": 0, "Never": 1,"Rarely": 2,"Half the time": 3,"Frequently": 4,"Always": 5, "Not true": 1,"Somewhat true": 2,"Certainly true": 3, "Not sure": 0,"Never": 1,"Almost Never": 2, "Sometimes": 3, "Often": 4, "Almost always": 5, "Not at all": 0,"Once/twice": 1,"three/four times": 2,"More than four times": 3, "Over-excited and hard to settle for a long period more than a few hours": 1,"Over-excited and hard to settle for a short period": 2,"Relaxed and comfortable": 3,"Withdrawn, sad or restless for a short period": 4,"Withdrawn, sad or restless for a long period more than a few hours": 5}

# Initialize an empty list to store the mapped values
    mapped_values = []

# Iterate over the dataset and map each categorical value to its corresponding numeric value
    for instance in dataset:
        mapped_value = category_mapping[instance]
        mapped_values.append(mapped_value)

# Print the mapped values
    print(mapped_values)
  

    if  df18cm1 == "Yes":
        df18cm1_Yes, df18cm1_No = 1, 0 
    else:
 
        df18cm1_Yes, df18cm1_No = 0, 1 
    
    if  dhs08a7 == "Yes":
        dhs08a7_Yes, dhs08a7_No = 1, 0 
    else:
 
        dhs08a7_Yes, dhs08a7_No = 0, 1 
    
    if  dse13a1 == "Never":
        dse13a1_Never, dse13a1_Rarely, dse13a1_Half, dse13a1_Frequently, dse13a1_always = 1, 0,0,0,0
    elif  dse13a1 == "Rarely":
        dse13a1_Never, dse13a1_Rarely, dse13a1_Half, dse13a1_Frequently, dse13a1_always = 0, 1,0,0,0
        
    elif  dse13a1 == "Half the time":
        dse13a1_Never, dse13a1_Rarely, dse13a1_Half, dse13a1_Frequently, dse13a1_always = 0,0,1,0,0
    elif  dse13a1 == "Frequently":
        dse13a1_Never, dse13a1_Rarely, dse13a1_Half, dse13a1_Frequently, dse13a1_always = 0,0,0,1,0
    elif  dse13a1 == "Always":
        dse13a1_Never, dse13a1_Rarely, dse13a1_Half, dse13a1_Frequently, dse13a1_always = 0,0,0,0,1 

    if  dse13a4 == "Never":
        dse13a4_Never, dse13a4_Rarely, dse13a4_Half, dse13a4_Frequently, dse13a4_always = 1, 0,0,0,0
    elif  dse13a4 == "Rarely":
        dse13a4_Never, dse13a4_Rarely, dse13a4_Half, dse13a4_Frequently, dse13a4_always = 0, 1,0,0,0
        
    elif  dse13a4 == "Half the time":
        dse13a4_Never, dse13a4_Rarely, dse13a4_Half, dse13a4_Frequently, dse13a4_always = 0,0,1,0,0
    elif  dse13a4 == "Frequently":
        dse13a4_Never, dse13a4_Rarely, dse13a4_Half, dse13a4_Frequently, dse13a4_always = 0,0,0,1,0
    elif  dse13a4 == "Always":
        dse13a4_Never, dse13a4_Rarely, dse13a4_Half, dse13a4_Frequently, dse13a4_always = 0,0,0,0,1 
    
    if  dse13b1 == "Never":
        dse13b1_Never, dse13b1_Rarely, dse13b1_Half, dse13b1_Frequently, dse13b1_always = 1, 0,0,0,0
    elif  dse13b1 == "Rarely":
        dse13b1_Never, dse13b1_Rarely, dse13b1_Half, dse13b1_Frequently, dse13b1_always = 0, 1,0,0,0
        
    elif  dse13b1 == "Half the time":
        dse13b1_Never, dse13b1_Rarely, dse13b1_Half, dse13b1_Frequently, dse13b1_always = 0,0,1,0,0
    elif  dse13b1 == "Frequently":
        dse13b1_Never, dse13b1_Rarely, dse13b1_Half, dse13b1_Frequently, dse13b1_always = 0,0,0,1,0
    elif  dse13b1 == "Always":
        dse13b1_Never, dse13b1_Rarely, dse13b1_Half, dse13b1_Frequently, dse13b1_always = 0,0,0,0,1 
    
    if  dse13b4 == "Never":
        dse13b4_Never, dse13b4_Rarely, dse13b4_Half, dse13b4_Frequently, dse13b4_always = 1, 0,0,0,0
    elif  dse13b4 == "Rarely":
        dse13b4_Never, dse13b4_Rarely, dse13b4_Half, dse13b4_Frequently, dse13b4_always = 0, 1,0,0,0
        
    elif  dse13b4 == "Half the time":
        dse13b4_Never, dse13b4_Rarely, dse13b4_Half, dse13b4_Frequently, dse13b4_always = 0,0,1,0,0
    elif  dse13b4 == "Frequently":
        dse13b4_Never, dse13b4_Rarely, dse13b4_Half, dse13b4_Frequently, dse13b4_always = 0,0,0,1,0
    elif  dse13b4 == "Always":
        dse13b4_Never, dse13b4_Rarely, dse13b4_Half, dse13b4_Frequently, dse13b4_always = 0,0,0,0,1

        
    if  dse03a3a == "Not true":
        dse03a3a_Not, dse03a3a_Somewhat, dse03a3a_True = 1, 0,0
        
    elif  dse03a3a == "Somewhat true":
        dse03a3a_Not, dse03a3a_Somewhat, dse03a3a_True = 0, 1,0
        
    elif  dse03a3a == "Certainly true":
        dse03a3a_Not, dse03a3a_Somewhat, dse03a3a_True = 0, 0,1
        
    if  dse03a3b == "Not true":
        dse03a3b_Not, dse03a3b_Somewhat, dse03a3b_True = 1, 0,0
        
    elif  dse03a3b == "Somewhat true":
        dse03a3b_Not, dse03a3b_Somewhat, dse03a3b_True = 0, 1,0
        
    elif  dse03a3b == "Certainly true":
        dse03a3b_Not, dse03a3b_Somewhat, dse03a3b_True = 0, 0,1
        
        
    if  dse03a3c == "Not true":
        dse03a3c_Not, dse03a3c_Somewhat, dse03a3c_True = 1, 0,0
        
    elif  dse03a3c == "Somewhat true":
        dse03a3c_Not, dse03a3c_Somewhat, dse03a3c_True = 0, 1,0
        
    elif  dse03a3c == "Certainly true":
        dse03a3c_Not, dse03a3c_Somewhat, dse03a3c_True = 0, 0,1
    
    if  dse03a3d == "Not true":
        dse03a3d_Not, dse03a3d_Somewhat, dse03a3d_True = 1, 0,0
        
    elif  dse03a3d == "Somewhat true":
        dse03a3d_Not, dse03a3d_Somewhat, dse03a3d_True = 0, 1,0
        
    elif  dse03a3d == "Certainly true":
        dse03a3d_Not, dse03a3d_Somewhat, dse03a3d_True = 0, 0,1
        
        
    if  dse03a3e == "Not true":
        dse03a3e_Not, dse03a3e_Somewhat, dse03a3e_True = 1, 0,0
        
    elif  dse03a3e == "Somewhat true":
        dse03a3e_Not, dse03a3e_Somewhat, dse03a3e_True = 0, 1,0
        
    elif  dse03a3e == "Certainly true":
        dse03a3e_Not, dse03a3e_Somewhat, dse03a3e_True = 0, 0,1
        
    if  dse06a == "Not true":
        dse06a_Not, dse06a_Somewhat, dse06a_True = 1, 0,0
        
    elif  dse06a == "Somewhat true":
        dse06a_Not, dse06a_Somewhat, dse06a_True = 0, 1,0
        
    elif  dse06a == "Certainly true":
        dse06a_Not, dse06a_Somewhat, dse06a_True = 0, 0,1
    
    if  dgd04b1a == "Not sure":
        dgd04b1a_Notsure, dgd04b1a_Never, dgd04b1a_Almost, dgd04b1a_Sometimes, dgd04b1a_Often, dgd04b1a_Almostalways,dgd04b1a_N =1,0,0,0,0,0,0
    elif  dgd04b1a == "Never":
        dgd04b1a_Notsure, dgd04b1a_Never, dgd04b1a_Almost, dgd04b1a_Sometimes, dgd04b1a_Often, dgd04b1a_Almostalways,dgd04b1a_N =0,1,0,0,0,0,0
        
    elif  dgd04b1a == "Almost never":
        dgd04b1a_Notsure, dgd04b1a_Never, dgd04b1a_Almost, dgd04b1a_Sometimes, dgd04b1a_Often, dgd04b1a_Almostalways, dgd04b1a_N =0,0,1,0,0,0,0
    elif  dgd04b1a == "Sometimes":
        dgd04b1a_Notsure, dgd04b1a_Never, dgd04b1a_Almost, dgd04b1a_Sometimes, dgd04b1a_Often, dgd04b1a_Almostalways, dgd04b1a_N  =0,0,0,1,0,0,0
    elif  dgd04b1a == "Often":
        dgd04b1a_Notsure, dgd04b1a_Never, dgd04b1a_Almost, dgd04b1a_Sometimes, dgd04b1a_Often, dgd04b1a_Almostalways,dgd04b1a_N  =0,0,0,0,1,0,0
    elif dgd04b1a == "Almost always":
        dgd04b1a_Notsure, dgd04b1a_Never, dgd04b1a_Almost, dgd04b1a_Sometimes, dgd04b1a_Often, dgd04b1a_Almostalways,dgd04b1a_N  =0,0,0,0,0,1,0
        
    
    
    
    if  dgd04b1b == "Never":
        dgd04b1b_Never, dgd04b1b_Rarely, dgd04b1b_Half, dgd04b1b_Frequently, dgd04b1b_always = 1, 0,0,0,0
    elif  dgd04b1b == "Almost never":
        dgd04b1b_Never, dgd04b1b_Rarely, dgd04b1b_Half, dgd04b1b_Frequently, dgd04b1b_always = 0, 1,0,0,0
        
    elif  dgd04b1b == "Sometimes":
        dgd04b1b_Never, dgd04b1b_Rarely, dgd04b1b_Half, dgd04b1b_Frequently, dgd04b1b_always = 0,0,1,0,0
    elif  dgd04b1b == "Often":
        dgd04b1b_Never, dgd04b1b_Rarely, dgd04b1b_Half, dgd04b1b_Frequently, dgd04b1b_always = 0,0,0,1,0
    elif  dgd04b1b == "Almost always":
        dgd04b1b_Never, dgd04b1b_Rarely, dgd04b1b_Half, dgd04b1b_Frequently, dgd04b1b_always = 0,0,0,0,1
     
    if  dgd04b1c == "Never":
        dgd04b1c_Never, dgd04b1c_Rarely, dgd04b1c_Half, dgd04b1c_Frequently, dgd04b1c_always = 1, 0,0,0,0
    elif  dgd04b1c == "Almost never":
        dgd04b1c_Never, dgd04b1c_Rarely, dgd04b1c_Half, dgd04b1c_Frequently, dgd04b1c_always = 0, 1,0,0,0
        
    elif  dgd04b1c == "Sometimes":
        dgd04b1c_Never, dgd04b1c_Rarely, dgd04b1c_Half, dgd04b1c_Frequently, dgd04b1c_always = 0,0,1,0,0
    elif  dgd04b1c == "Often":
        dgd04b1c_Never, dgd04b1c_Rarely, dgd04b1c_Half, dgd04b1c_Frequently, dgd04b1c_always = 0,0,0,1,0
    elif  dgd04b1c == "Almost always":
        dgd04b1c_Never, dgd04b1c_Rarely, dgd04b1c_Half, dgd04b1c_Frequently, dgd04b1c_always = 0,0,0,0,1
        
        
    
    if  dgd04b1d == "Not sure":
        dgd04b1d_Notsure, dgd04b1d_Never, dgd04b1d_Almost, dgd04b1d_Sometimes, dgd04b1d_Often, dgd04b1d_Almostalways=1,0,0,0,0,0
    elif  dgd04b1d == "Never":
        dgd04b1d_Notsure, dgd04b1d_Never, dgd04b1d_Almost, dgd04b1d_Sometimes, dgd04b1d_Often, dgd04b1d_Almostalways =0,1,0,0,0,0
        
    elif  dgd04b1d == "Almost never":
        dgd04b1d_Notsure, dgd04b1d_Never, dgd04b1d_Almost, dgd04b1d_Sometimes, dgd04b1d_Often, dgd04b1d_Almostalways=0,0,1,0,0,0
    elif  dgd04b1d == "Sometimes":
        dgd04b1d_Notsure, dgd04b1d_Never, dgd04b1d_Almost, dgd04b1d_Sometimes, dgd04b1d_Often, dgd04b1d_Almostalways =0,0,0,1,0,0
    elif  dgd04b1d == "Often":
        dgd04b1d_Notsure, dgd04b1d_Never, dgd04b1d_Almost, dgd04b1d_Sometimes, dgd04b1d_Often, dgd04b1d_Almostalways =0,0,0,0,1,0
    elif dgd04b1d == "Almost always":
        dgd04b1d_Notsure, dgd04b1d_Never, dgd04b1d_Almost, dgd04b1d_Sometimes, dgd04b1d_Often, dgd04b1d_Almostalways =0,0,0,0,0,1 
    
    
    
    
        
    if  dgd04b1e == "Never":
        dgd04b1e_Never, dgd04b1e_Rarely, dgd04b1e_Half, dgd04b1e_Frequently, dgd04b1e_always = 1, 0,0,0,0
    elif  dgd04b1e == "Almost never":
        dgd04b1e_Never, dgd04b1e_Rarely, dgd04b1e_Half, dgd04b1e_Frequently, dgd04b1e_always = 0, 1,0,0,0
        
    elif  dgd04b1e == "Sometimes":
        dgd04b1e_Never, dgd04b1e_Rarely, dgd04b1e_Half, dgd04b1e_Frequently, dgd04b1e_always = 0,0,1,0,0
    elif  dgd04b1e == "Often":
        dgd04b1e_Never, dgd04b1e_Rarely, dgd04b1e_Half, dgd04b1e_Frequently, dgd04b1e_always = 0,0,0,1,0
    elif  dgd04b1e == "Almost always":
        dgd04b1e_Never, dgd04b1e_Rarely, dgd04b1e_Half, dgd04b1e_Frequently, dgd04b1e_always = 0,0,0,0,1
        
        
    if  dgd04b2d == "Not sure":
        dgd04b2d_Notsure, dgd04b2d_Never, dgd04b2d_Almost, dgd04b2d_Sometimes, dgd04b2d_Often, dgd04b2d_Almostalways, dgd04b2d_N1,dgd04b2d_N2, dgd04b2d_N3 =1,0,0,0,0,0,0,0,0
    elif  dgd04b2d == "Never":
        dgd04b2d_Notsure, dgd04b2d_Never, dgd04b2d_Almost, dgd04b2d_Sometimes, dgd04b2d_Often, dgd04b2d_Almostalways, dgd04b2d_N1,dgd04b2d_N2, dgd04b2d_N3 =0,1,0,0,0,0,0,0,0
        
    elif  dgd04b2d == "Almost never":
        dgd04b2d_Notsure, dgd04b2d_Never, dgd04b2d_Almost, dgd04b2d_Sometimes, dgd04b2d_Often, dgd04b2d_Almostalways,dgd04b2d_N1,dgd04b2d_N2, dgd04b2d_N3 =0,0,1,0,0,0,0,0,0
    elif  dgd04b2d == "Sometimes":
        dgd04b2d_Notsure, dgd04b2d_Never, dgd04b2d_Almost, dgd04b2d_Sometimes, dgd04b2d_Often, dgd04b2d_Almostalways, dgd04b2d_N1,dgd04b2d_N2, dgd04b2d_N3 =0,0,0,1,0,0,0,0,0
    elif  dgd04b2d == "Often":
        dgd04b2d_Notsure, dgd04b2d_Never, dgd04b2d_Almost, dgd04b2d_Sometimes, dgd04b2d_Often, dgd04b2d_Almostalways, dgd04b2d_N1,dgd04b2d_N2, dgd04b2d_N3 =0,0,0,0,1,0,0,0,0
    elif dgd04b2d == "Almost always":
        dgd04b2d_Notsure, dgd04b2d_Never, dgd04b2d_Almost, dgd04b2d_Sometimes, dgd04b2d_Often, dgd04b2d_Almostalways, dgd04b2d_N1,dgd04b2d_N2, dgd04b2d_N3 =0,0,0,0,0,1,0,0,0
        
        
    if  dgd04b3g == "Not sure":
        dgd04b3g_Notsure, dgd04b3g_Never, dgd04b3g_Almost, dgd04b3g_Sometimes, dgd04b3g_Often, dgd04b3g_Almostalways =1,0,0,0,0,0
    elif  dgd04b3g == "Never":
        dgd04b3g_Notsure, dgd04b3g_Never, dgd04b3g_Almost, dgd04b3g_Sometimes, dgd04b3g_Often, dgd04b3g_Almostalways =0,1,0,0,0,0
        
    elif  dgd04b3g == "Almost never":
        dgd04b3g_Notsure, dgd04b3g_Never, dgd04b3g_Almost, dgd04b3g_Sometimes, dgd04b3g_Often, dgd04b3g_Almostalways =0,0,1,0,0,0
    elif  dgd04b3g == "Sometimes":
        dgd04b3g_Notsure, dgd04b3g_Never, dgd04b3g_Almost, dgd04b3g_Sometimes, dgd04b3g_Often, dgd04b3g_Almostalways =0,0,0,1,0,0
    elif  dgd04b3g == "Often":
        dgd04b3g_Notsure, dgd04b3g_Never, dgd04b3g_Almost, dgd04b3g_Sometimes, dgd04b3g_Often, dgd04b3g_Almostalways= 0,0,0,0,1,0
    elif dgd04b3g == "Almost always":
        dgd04b3g_Notsure, dgd04b3g_Never, dgd04b3g_Almost, dgd04b3g_Sometimes, dgd04b3g_Often, dgd04b3g_Almostalways=0,0,0,0,0,1
        
        
  
        
        
    
    
        
        
    #"Not at all","Once/twice","three/four times","More than four times"
    
    if  dpc49a3 == "Not at all":
        dpc49a3_Not, dpc49a3_Once, dpc49a3_three, dpc49a3_Four = 1, 0,0,0
    elif  dpc49a3 == "Once/twice":
        dpc49a3_Not, dpc49a3_Once, dpc49a3_three, dpc49a3_Four = 0, 1,0,0
        
    elif  dpc49a3 == "three/four times":
        dpc49a3_Not, dpc49a3_Once, dpc49a3_three, dpc49a3_Four = 0, 0,1,0
        
    elif  dpc49a3 == "More than four times":
        dpc49a3_Not, dpc49a3_Once, dpc49a3_three, dpc49a3_Four = 0, 0,0,1
  
    #"Over-excited and hard to settle for a long period more than a few hours","Over-excited and hard to settle for a short period","Relaxed and comfortable","Withdrawn, sad or restless for a short period","Withdrawn, sad or restless for a long period more than a few hours"
    
    if  dpc28a1 == "Over-excited and hard to settle for a long period more than a few hours":
        dpc28a1_Excited_long, dpc28a1_Excited_short, dpc28a1_Withdrawn_short, dpc28a1_Withdrawn_sad, dpc28a1_Withdrawn_long,dpc28a1_Notsure = 1, 0,0,0,0,0
    elif  dpc28a1 == "Over-excited and hard to settle for a short period":
        dpc28a1_Excited_long, dpc28a1_Excited_short, dpc28a1_Withdrawn_short, dpc28a1_Withdrawn_sad, dpc28a1_Withdrawn_long,dpc28a1_Notsure   = 0, 1,0,0,0,0
        
    elif  dpc28a1 == "Relaxed and comfortable":
        dpc28a1_Excited_long, dpc28a1_Excited_short, dpc28a1_Withdrawn_short, dpc28a1_Withdrawn_sad, dpc28a1_Withdrawn_long,dpc28a1_Notsure  = 0, 0,1,0,0,0
        
    elif  dpc28a1 == "Withdrawn, sad or restless for a short period":
        dpc28a1_Excited_long, dpc28a1_Excited_short, dpc28a1_Withdrawn_short, dpc28a1_Withdrawn_sad, dpc28a1_Withdrawn_long,dpc28a1_Notsure  = 0, 0,0,1,0,0
        
    elif  dpc28a1 == "Withdrawn, sad or restless for a long period more than a few hours":
        dpc28a1_Excited_long, dpc28a1_Excited_short, dpc28a1_Withdrawn_short, dpc28a1_Withdrawn_sad, dpc28a1_Withdrawn_long,dpc28a1_Notsure  = 0, 0,0,0,1,0
        
  
    if  dse03t3b == "Never":
        dse03t3b_Never, dse03t3b_Rarely, dse03t3b_Half, dse03t3b_Frequently, dse03t3b_always = 1, 0,0,0,0
    elif  dse03t3b == "Rarely":
        dse03t3b_Never, dse03t3b_Rarely, dse03t3b_Half, dse03t3b_Frequently, dse03t3b_always = 0, 1,0,0,0
        
    elif  dse03t3b == "Half the time":
        dse03t3b_Never, dse03t3b_Rarely, dse03t3b_Half, dse03t3b_Frequently, dse03t3b_always = 0, 0,1,0,0
    elif  dse03t3b == "Frequently":
        dse03t3b_Never, dse03t3b_Rarely, dse03t3b_Half, dse03t3b_Frequently, dse03t3b_always = 0, 0,0,1,0
    elif  dse03t3b == "Always":
        dse03t3b_Never, dse03t3b_Rarely, dse03t3b_Half, dse03t3b_Frequently, dse03t3b_always = 0, 0,0,0,1
        
    if dse03m3a == "Not true":
        dse03m3a_Not, dse03m3a_Somewhat, dse03m3a_True = 1, 0, 0
    elif dse03m3a == "Somewhat true":
        dse03m3a_Not, dse03m3a_Somewhat, dse03m3a_True = 0, 1, 0
    elif dse03m3a == "Certainly true":
        dse03m3a_Not, dse03m3a_Somewhat, dse03m3a_True = 0, 0, 1 
        
    if  dse03m3b == "Not true":
        dse03m3b_Not, dse03m3b_Somewhat, dse03m3b_True = 1, 0,0
        
    elif  dse03m3b == "Somewhat true":
        dse03m3b_Not, dse03m3b_Somewhat, dse03m3b_True = 0, 1,0
        
    elif  dse03m3b == "Certainly true":
        dse03m3b_Not, dse03m3b_Somewhat, dse03m3b_True = 0, 0,1 
     
    
    if  dse03m3c == "Not true":
        dse03m3c_Not, dse03m3c_Somewhat, dse03m3c_True = 1, 0,0
        
    elif  dse03m3c == "Somewhat true":
        dse03m3c_Not, dse03m3c_Somewhat, dse03m3c_True = 0, 1,0
        
    elif  dse03m3c == "Certainly true":
        dse03m3c_Not, dse03m3c_Somewhat, dse03m3c_True = 0, 0,1 
        
        
    if  dse03m3d == "Not true":
        dse03m3d_Not, dse03m3d_Somewhat, dse03m3d_True = 1, 0,0
        
    elif  dse03m3d == "Somewhat true":
        dse03m3d_Not, dse03m3d_Somewhat, dse03m3d_True = 0, 1,0
        
    elif  dse03m3d == "Certainly true":
        dse03m3d_Not, dse03m3d_Somewhat, dse03m3d_True = 0, 0,1 
        
    if  dse03m3e == "Not true":
        dse03m3e_Not, dse03m3e_Somewhat, dse03m3e_True = 1, 0,0
        
    elif  dse03m3e == "Somewhat true":
        dse03m3e_Not, dse03m3e_Somewhat, dse03m3e_True = 0, 1,0
        
    elif  dse03m3e == "Certainly true":
        dse03m3e_Not, dse03m3e_Somewhat, dse03m3e_True = 0, 0,1 
      
    if age =="6" or age=="7":
        age_6,age_8,age_10,age_12,age_14 = 1,0,0,0,0
    elif age =="8" or age=="9":
        age_6,age_8,age_10,age_12,age_14 = 0,1,0,0,0
    elif age =="10" or age=="11":
        age_6,age_8,age_10,age_12,age_14 = 0,0,1,0,0
    elif age =="12" or age=="13":
        age_6,age_8,age_10,age_12,age_14 = 0,0,0,1,0
    elif age =="14" or age=="15":
        age_6,age_8,age_10,age_12,age_14 = 0,0,0,0,1
   
    
    
    
    features = ([df18cm1_Yes, df18cm1_No, 
dhs08a7_Yes, dhs08a7_No, 
dse13a1_Never, dse13a1_Rarely, dse13a1_Half, dse13a1_Frequently, dse13a1_always, dse13a4_Never, dse13a4_Rarely, dse13a4_Half, dse13a4_Frequently, dse13a4_always, dse13b1_Never, dse13b1_Rarely, dse13b1_Half, dse13b1_Frequently, dse13b1_always, dse13b4_Never, dse13b4_Rarely, dse13b4_Half, dse13b4_Frequently, dse13b4_always, dse03a3a_Not, dse03a3a_Somewhat, dse03a3a_True, 
dse03a3b_Not, dse03a3b_Somewhat, dse03a3b_True, 
dse03a3c_Not, dse03a3c_Somewhat, dse03a3c_True, 
dse03a3d_Not, dse03a3d_Somewhat, dse03a3d_True,
dse03a3e_Not, dse03a3e_Somewhat, dse03a3e_True, 
dse06a_Not, dse06a_Somewhat, dse06a_True, 
dgd04b1a_Notsure, dgd04b1a_Never, dgd04b1a_Almost, dgd04b1a_Sometimes, dgd04b1a_Often, dgd04b1a_Almostalways, dgd04b1a_N,
dgd04b1b_Never, dgd04b1b_Rarely, dgd04b1b_Half, dgd04b1b_Frequently, dgd04b1b_always, dgd04b1c_Never, dgd04b1c_Rarely, dgd04b1c_Half, dgd04b1c_Frequently, dgd04b1c_always, dgd04b1d_Notsure, dgd04b1d_Never, dgd04b1d_Almost, dgd04b1d_Sometimes, dgd04b1d_Often, dgd04b1d_Almostalways, 
dgd04b1e_Never, dgd04b1e_Rarely, dgd04b1e_Half, dgd04b1e_Frequently, dgd04b1e_always, 
dgd04b2d_Notsure, dgd04b2d_Never, dgd04b2d_Almost, dgd04b2d_Sometimes, dgd04b2d_Often, dgd04b2d_Almostalways, dgd04b2d_N1, dgd04b2d_N2, dgd04b2d_N3, 
dgd04b3g_Notsure, dgd04b3g_Never, dgd04b3g_Almost, dgd04b3g_Sometimes, dgd04b3g_Often, dgd04b3g_Almostalways, 
dpc49a3_Not, dpc49a3_Once, dpc49a3_three, dpc49a3_Four, 
dpc28a1_Excited_long, dpc28a1_Excited_short, dpc28a1_Withdrawn_short, dpc28a1_Withdrawn_sad, dpc28a1_Withdrawn_long, dpc28a1_Notsure, 
dse03t3b_Never, dse03t3b_Rarely, dse03t3b_Half, dse03t3b_Frequently, dse03t3b_always, 
dse03m3a_Not, dse03m3a_Somewhat, dse03m3a_True, 
dse03m3b_Not, dse03m3b_Somewhat, dse03m3b_True, 
dse03m3c_Not, dse03m3c_Somewhat, dse03m3c_True, 
dse03m3d_Not, dse03m3d_Somewhat, dse03m3d_True, 
dse03m3e_Not, dse03m3e_Somewhat, dse03m3e_True
])
    features_age = ([df18cm1_Yes, df18cm1_No, 
dhs08a7_Yes, dhs08a7_No, 
dse13a1_Never, dse13a1_Rarely, dse13a1_Half, dse13a1_Frequently, dse13a1_always, dse13a4_Never, dse13a4_Rarely, dse13a4_Half, dse13a4_Frequently, dse13a4_always, dse13b1_Never, dse13b1_Rarely, dse13b1_Half, dse13b1_Frequently, dse13b1_always, dse13b4_Never, dse13b4_Rarely, dse13b4_Half, dse13b4_Frequently, dse13b4_always, dse03a3a_Not, dse03a3a_Somewhat, dse03a3a_True, 
dse03a3b_Not, dse03a3b_Somewhat, dse03a3b_True, 
dse03a3c_Not, dse03a3c_Somewhat, dse03a3c_True, 
dse03a3d_Not, dse03a3d_Somewhat, dse03a3d_True,
dse03a3e_Not, dse03a3e_Somewhat, dse03a3e_True, 
dse06a_Not, dse06a_Somewhat, dse06a_True, 
dgd04b1a_Notsure, dgd04b1a_Never, dgd04b1a_Almost, dgd04b1a_Sometimes, dgd04b1a_Often, dgd04b1a_Almostalways, dgd04b1a_N,
dgd04b1b_Never, dgd04b1b_Rarely, dgd04b1b_Half, dgd04b1b_Frequently, dgd04b1b_always, dgd04b1c_Never, dgd04b1c_Rarely, dgd04b1c_Half, dgd04b1c_Frequently, dgd04b1c_always, dgd04b1d_Notsure, dgd04b1d_Never, dgd04b1d_Almost, dgd04b1d_Sometimes, dgd04b1d_Often, dgd04b1d_Almostalways, 
dgd04b1e_Never, dgd04b1e_Rarely, dgd04b1e_Half, dgd04b1e_Frequently, dgd04b1e_always, 
dgd04b2d_Notsure, dgd04b2d_Never, dgd04b2d_Almost, dgd04b2d_Sometimes, dgd04b2d_Often, dgd04b2d_Almostalways, dgd04b2d_N1, dgd04b2d_N2, dgd04b2d_N3, 
dgd04b3g_Notsure, dgd04b3g_Never, dgd04b3g_Almost, dgd04b3g_Sometimes, dgd04b3g_Often, dgd04b3g_Almostalways, 
dpc49a3_Not, dpc49a3_Once, dpc49a3_three, dpc49a3_Four, 
dpc28a1_Excited_long, dpc28a1_Excited_short, dpc28a1_Withdrawn_short, dpc28a1_Withdrawn_sad, dpc28a1_Withdrawn_long, dpc28a1_Notsure, 
dse03t3b_Never, dse03t3b_Rarely, dse03t3b_Half, dse03t3b_Frequently, dse03t3b_always, 
dse03m3a_Not, dse03m3a_Somewhat, dse03m3a_True, 
dse03m3b_Not, dse03m3b_Somewhat, dse03m3b_True, 
dse03m3c_Not, dse03m3c_Somewhat, dse03m3c_True, 
dse03m3d_Not, dse03m3d_Somewhat, dse03m3d_True, 
dse03m3e_Not, dse03m3e_Somewhat, dse03m3e_True,
age_6,age_8,age_10,age_12,age_14
])

    features = np.array(features)
    features_age = np.array(features_age)
#label encode your categorical columns
#le = preprocessing.LabelEncoder()
#for i in range(len(categorical)):
    #features[:, i] = le.fit_transform(features[:, i])

#features = np.array(features)

#features = np.fromstring(features, dtype=float, sep=' ')
#features = np.fromstring(features, dtype=int, sep=' ')
    results = features.reshape(1, -1)
    results1 = features_age.reshape(1, -1)




 
# when 'Predict' is clicked, make the prediction and store it 
    if st.button("Get Your Prediction"):
    #X = pd.DataFrame({'poco1a2y':[poco1a2y],'poco3a3y':[poco3a3y],'pocc1a2y':[pocc1a2y],'poco2cy':[poco2cy],'pocc1cy':[pocc1cy],'pocc2cy':[pocc2cy],'pocc3cy':[pocc3cy],'pocc4cy':[pocc4cy],'pocc4cm':[pocc4cm]}
    #prediction = (loaded_model.predict_proba(X)[:,1] >= 0.6).astype(bool)
       
  
    #T_filtered = scaler.transform(T)
    
    # Making predictions            
        prediction = loaded_model.predict(results)
    #st.success('Your Target is {}'.format(prediction))
        

    #prediction = loaded_model.predict(results)
        if prediction[0] == 1:
            st.success('Depression exists')
           
        else:
            
            prediction1 = loaded_model1.predict_proba(results1)[:,0]
            st.error('Depression does not exist and the likelihood is:')
            probability_class_1 = prediction1[0] - 0.45 # Probability for class 1 (assuming it's associated with 88%)

            # Convert probability to percentage
            probability_percentage = probability_class_1 * 100

            st.write(f"{probability_percentage:.2f}%")  # Display the percentage rounded to two decimal places
            
            #st.write ('Do you want to check the likelihood?')
            #if st.button("Yes"):
                #prediction = loaded_model.predict_proba(results)[:,1]
                #st.write(prediction[0]*100)
           
            #option = st.selectbox('Do you want to check the likelihood?', ("Yes", "No"))
            #if option == "Yes":
            #prediction = loaded_model.predict_proba(results)[:,1]
            #st.write(prediction[0]*100)  
            #prediction1 = loaded_model1.predict_proba(results)[:, 1]
            #st.write("The likelihood of developing depression is", prediction1[0])
            #elif option == "No":
                #st.write("Thanks for using this app")