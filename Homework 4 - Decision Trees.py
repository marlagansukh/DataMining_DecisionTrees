#!/usr/bin/env python
# coding: utf-8

# # Homework 4 - Decision Trees

# Analyst: Marla Gansukh
# 
# Assignment: Homework 2 
# 
# Purpose: Predicting a survival rate after the Titanic sinking based on the decision tree models. 

# <b> Academic Integrity Statement <b> 
# >By submitting this code for grading, I confirm the following:
# - that this notebook represents my own unique code created as a solution to the assigned problem 
# - that this notebook was created specifically to satisfy the requirements detailed in this assignment. 
#     >other than the textbook and material presented during our class sessions, I DID NOT receive any help in designing and debugging my code from another source.

# In[45]:


import pandas as pd
from sklearn import tree
from sklearn import metrics
from sklearn import model_selection as skms
import numpy as np 


# # FIRST PART 

# In[46]:


titanic = pd.read_csv(r"C:\Users\marla\Downloads\Titanic_tree.csv")
titanic = titanic.dropna()
titanic.info()


# In[47]:


from sklearn.preprocessing import OneHotEncoder


# In[48]:


one_hot_encoded_data = pd.get_dummies(titanic, columns = ['sex'])
df_1 = one_hot_encoded_data
df_1.info()
df_1.head()


# In[49]:


LABEL_NAME = 'survived' 
df = one_hot_encoded_data   

target = df[LABEL_NAME]
features = df.drop(columns = ['survived','name','ticket','fare','embarked'])


tts = skms.train_test_split(features, target, 
                            test_size=0.2, random_state=99)


(train_ftrs, test_ftrs, train_target, test_target) = tts


# - <b>The reason why we are using age, pclass, gender, sibsp, and parch data attributes to predict the survival rate is because other attributes are not related to the survival rate at all. In real life, the survival rate is determined based on passenger's age and gender the most. Besides that, passengers are likely to survive for their loved ones if they are on board. On the other hand, ticket number, fare price and the location they boarded the plane have no relationship with the survival rate of the passenger. <b>

# In[50]:


titanic_tree = tree.DecisionTreeClassifier(criterion = 'gini', max_depth=4)
model = titanic_tree.fit(train_ftrs, train_target)
model.score(test_ftrs, test_target)


# In[51]:


preds = model.predict(test_ftrs)

cm = metrics.confusion_matrix(test_target,preds)
cm_disp = metrics.ConfusionMatrixDisplay(confusion_matrix = cm,
                                        display_labels = model.classes_)
cm_disp.plot()


# In[52]:


print(metrics.classification_report(test_target,
                                   preds,
                                   digits = 4,
                                   zero_division = 0))


# - <b> According to the confusion matrix, TN&TP are relatively higher than the FP&FN, which means our model is performing average. In the classification report above, the precision score is slightly above than 80 which means the model is not performing excellent but not poorly. Neither of the outputs above give us a score above than 90, which means the performance of the model is not the best. <b>

# In[53]:


import matplotlib.pyplot as plt 
plt.figure(figsize=(25,12))
tree.plot_tree(model,
              feature_names = test_ftrs.columns,
              class_names = model.classes_,
              filled = True,
              fontsize = 12)
plt.show()


# In[54]:


results = test_ftrs.copy(deep = True)
results['actual'] = test_target
results['predicted'] = preds 
results.head()


# In[55]:


results[results['actual']!= results ['predicted']]


# # SECOND PART

# In[56]:


real_titanic = pd.read_csv(r"C:\Users\marla\OneDrive - University of Central Arkansas\Documents\titanic_real.csv")
real_titanic.info()


# In[62]:


encoded_data = pd.get_dummies(real_titanic, columns = ['sex'])
dataframe = encoded_data
dataframe.info()


# In[63]:


label_name = 'survived'
dataframe_2 = encoded_data   

target = dataframe_2[label_name]
features = dataframe_2.drop(columns = ['survived','name','ticket','fare','embarked'])


tts = skms.train_test_split(features, target, 
                            test_size=0.2, random_state=99)


(train_ftrs, test_ftrs, train_target, test_target) = tts


# In[68]:


new_preds = model.predict(features)


# In[71]:


fam_results = test_ftrs.copy(deep = True)
fam_results['survival_predicted']=pd.Series(new_preds)
fam_results.head()


# Output above doesn't make a clear sense to me since we can't see the actual survival rate to compare with the predicted rate. The data sample size might be one of the reasons that the fam_results output doesn't make a sense. 
