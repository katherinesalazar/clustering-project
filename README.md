# Zillow: What is driving the errors in the Zestimates?
The Zillow Data Science Team wants to understand what is driving the errors in the zestimates?
- Data Scope:
    - 2017 properties
    - Single units / Single family homes

![alt text](put zillow objectives here)
![alt text](put executive summary here)

# Project Planning
- Refer to trello kanban board for planning overview: https://trello.com/b/aM5Pet9d/zillow-clustering-project
- Utilized agile program management methodology 
# Initial Ideas
- The Zillow Data Science team would like to know:
    - **What is driving the errors (logerror) Zestimate in single unit properties?**

<br />

- Project Objectives:
    - See if bathroomcnt, bedroomcnt and calculatedfinishedsquarefeet are features which drive the error (logerror) Zestimate in single unit properties
    - Documenting process and analysis throughout the data science pipeline.
    - Constructing a clustered regression model that predicts the error (logerror) for Zestimates.
<br /> 
<br /> 

**Hypothesis:**
- H0: There is not a correlation between zestimate logerror with the following features: bathroomcnt, bedroomcnt and calculatedfinishedsquarefeet.
- HA: There a correlation between between zestimate logerror with the following features: bathroomcnt, bedroomcnt and calculatedfinishedsquarefeet.


# Data Dictionary
![alt text](https://github.com/katherinesalazar/visuals/blob/main/zillow_data_dict.png)
# How to Reproduce
-  Read this README.md
- Download the acquire.py, prepare.py, model.py, final_prepare_explore.ipynb and final_notebook.ipynb into your working directory
- Have your own SQL database connection with your own env.py file
- Run the final_notebook.ipynb 
