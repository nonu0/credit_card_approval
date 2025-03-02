import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

app_df = pd.read_csv(r'C:\Users\Administrator\work\credit_card_elig\core\credit_card_approval\data\application_record.csv')
credit_df = pd.read_csv(r'C:\Users\Administrator\work\credit_card_elig\core\credit_card_approval\data\credit_record.csv')

def emp_edit(data):
    if data >= 0:
        return 0
    else:
        return abs(data // 365.25)

app_df.drop_duplicates(subset='ID',inplace=True)

app_df['DAYS_BIRTH'] = abs((app_df['DAYS_BIRTH'] // 365.25).astype(int))
app_df['DAYS_EMPLOYED'] = app_df['DAYS_EMPLOYED'].apply(emp_edit)
app_df['DAYS_EMPLOYED'] = app_df['DAYS_EMPLOYED'].astype(int)
app_df['CNT_FAM_MEMBERS'] = app_df['CNT_FAM_MEMBERS'].astype(int)
app_df = app_df.rename(columns={'DAYS_BIRTH':'Age','DAYS_EMPLOYED':'Years of Employment','CNT_FAM_MEMBERS':'Family members'})
print(app_df.describe())


# app_df.info()
