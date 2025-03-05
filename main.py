import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import skew
import matplotlib.pyplot as plt

app_df = pd.read_csv(r'C:\Users\Administrator\work\credit_card_elig\core\credit_card_approval\data\application_record.csv')
credit_df = pd.read_csv(r'C:\Users\Administrator\work\credit_card_elig\core\credit_card_approval\data\credit_record.csv')

def emp_edit(data):
    if data >= 0:
        return 0
    else:
        return abs(data // 365.25)


def edu_edit(data):
    data = data.split('/')[0]
    return data

app_df.drop_duplicates(subset='ID',inplace=True)

app_df['DAYS_BIRTH'] = abs((app_df['DAYS_BIRTH'] // 365.25).astype(np.int8))
app_df['DAYS_EMPLOYED'] = app_df['DAYS_EMPLOYED'].apply(emp_edit)
app_df['DAYS_EMPLOYED'] = app_df['DAYS_EMPLOYED'].astype(np.int8)
app_df['CNT_FAM_MEMBERS'] = app_df['CNT_FAM_MEMBERS'].astype(np.int8)
app_df = app_df.rename(columns={'DAYS_BIRTH':'Age','DAYS_EMPLOYED':'Years of Employment','CNT_FAM_MEMBERS':'Family members',
                                'AMT_INCOME_TOTAL':'Annual Income','NAME_EDUCATION_TYPE':'Education level',
                                'OCCUPATION_TYPE':'Occupation','CNT_CHILDREN':'Children','NAME_INCOME_TYPE':'Income Type'})
# print(app_df.describe())



app_df['Education level'] = app_df['Education level'].apply(edu_edit)
app_df['NAME_HOUSING_TYPE'] = app_df['NAME_HOUSING_TYPE'].apply(edu_edit)
app_df['NAME_FAMILY_STATUS'] = app_df['NAME_FAMILY_STATUS'].apply(edu_edit)
# occ = app_df.groupby(['Income Type','Education level'])['Occupation'].transform(lambda x:x.fillna(x.mode() if not x.mode().empty else 'Unknown'))
app_df['Occupation'] = app_df.groupby(['Income Type','Education level'])['Occupation'].transform(lambda x:x.fillna(np.random.choice(x.mode()) if not x.mode().empty else 'Unknown'))
app_df['Children'] = np.log1p(app_df['Children'])
app_df['Family members'] = np.log1p(app_df['Family members'])
app_df['Annual Income'] = np.log1p(app_df['Annual Income'])
# upper_limit = app_df['Children'].quantile(0.95)
# ch = app_df['Children'].clip(upper_limit)
# skewness = skew(app_df['Children'])
# print(app_df)
# print(occ['Family members'].value_counts())
# print(app_df['Annual Income'].describe())
# print(occ.value_counts())
# skewness = skew(occ)
# skewness1 = skew(app_df['Children'])
# print(skewness)
# print(skewness1)
# print(occ.value_counts())
# print(app_df['NAME_HOUSING_TYPE'].value_counts())
# print(app_df['NAME_FAMILY_STATUS'].value_counts())
# print(app_df['Education level'].value_counts())
# print(credit_df.describe())
# print(credit_df.value_counts().index)
# ann = set(app_df['ID'])
# print(len(ann))
# print(credit_df.pivot(columns=credit_df['MONTHS_BALANCE'],index=credit_df['ID'],values=credit_df['STATUS']))
# print(credit_df.describe())
# print(credit_df.nunique())
# print(credit_df['MONTHS_BALANCE'].value_counts())
# print(app_df.info())
# print(occ.sum())
# print(occ.isnull().sum() / len(app_df) * 100)
# print(len(app_df))
# app = app_df['Children'].clip(upper=app_df['Children'].quantile(0.95))
# print(app.value_counts())
# plt.figure(figsize=(10,7))
# sns.histplot(occ,bins=20)

# df = credit_df.groupby('ID').agg({'STATUS':['max','min','count'],
#                                   'MONTHS_BALANCE':'count'}).reset_index()
begin_month = pd.DataFrame(credit_df.groupby('ID')['MONTHS_BALANCE'].agg('min'))
begin_month = begin_month.rename(columns={'MONTHS_BALANCE':'Account Age'})
full_data = pd.merge(app_df,begin_month,how='left',on='ID')
credit_df['default_value'] = None
# sss = credit_df['default_value'][credit_df['STATUS'] == '2'] = 'Yes'
# print(sss.unique())
# credit_df['default_value'][credit_df['STATUS'] == '2'] = 'Yes'
credit_df.loc[credit_df['STATUS'].isin(['2','3','4','5']),'default_value'] = 'Yes'
# print(credit_df['default_value'].value_counts())
defaulted_count = credit_df.groupby('ID').count()
defaulted_count.loc[defaulted_count['default_value'] > 0,'default_value'] = 'Yes'
defaulted_count.loc[defaulted_count['default_value'] == 0, 'default_value'] = 'No'
default_status = defaulted_count[['default_value']]
print(default_status)


# sample_id = credit_df['ID'].sample(1).values[0]
sample_id = 5001715
sample_data = credit_df[credit_df['ID'] == sample_id]
# print(sample_data)
sns.lineplot(data=sample_data, x='MONTHS_BALANCE',y='STATUS',marker='o')
plt.title(f'credit history for ID:{sample_id}')
plt.xlabel('MONTHS_BALANCE')
plt.ylabel('STATUS')
# plt.show()
# app_df.info()
