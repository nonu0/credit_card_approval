import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

app_df = pd.read_csv(r'C:\Users\Administrator\work\credit_card_elig\core\credit_card_approval\data\application_record.csv')
credit_df = pd.read_csv(r'C:\Users\Administrator\work\credit_card_elig\core\credit_card_approval\data\credit_record.csv')

application_df = pd.DataFrame(app_df)

def edit_name(data):
    data = data.split('/')[0]
    return data


def edit_birthday(data):
    data = abs(data // 365.25)
    return data

def edit_days_employed(data):
    if data >= 0 :
        return 0
    else:
        data = abs(data // 365.25)
        return data

def detect_outlier(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]


def cap_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    upper_limit = Q3 + 1.5 * IQR
    # print(upper_limit.dtype)
    # print(upper_limit)
    # df.loc[df[column] > upper_limit,column] = upper_limit
    df[column] = np.where(df[column] > upper_limit,upper_limit,df[column])
    return df

otlier_ch = cap_outliers(application_df,'CNT_CHILDREN')
otlier_fa = cap_outliers(application_df,'CNT_FAM_MEMBERS')

# print(f'children outliers:{otlier_ch}')
# print(f'children outliers:{otlier_fa}')

application_df['NAME_EDUCATION_TYPE'] = application_df['NAME_EDUCATION_TYPE'].apply(edit_name)
application_df['NAME_FAMILY_STATUS'] = application_df['NAME_FAMILY_STATUS'].apply(edit_name)
application_df['NAME_HOUSING_TYPE'] = application_df['NAME_HOUSING_TYPE'].apply(edit_name)

application_df['DAYS_EMPLOYED'] = application_df['DAYS_EMPLOYED'].apply(edit_days_employed)
application_df['DAYS_EMPLOYED'] = application_df['DAYS_EMPLOYED'].astype(np.int8)


application_df['DAYS_BIRTH'] = application_df['DAYS_BIRTH'].apply(edit_birthday)
application_df['DAYS_BIRTH'] = application_df['DAYS_BIRTH'].astype(np.int8)

application_df['FLAG_MOBIL'] = application_df['FLAG_MOBIL'].astype(np.int8)
application_df['FLAG_WORK_PHONE'] = application_df['FLAG_WORK_PHONE'].astype(np.int8)
application_df['FLAG_PHONE'] = application_df['FLAG_PHONE'].astype(np.int8)
application_df['FLAG_EMAIL'] = application_df['FLAG_EMAIL'].astype(np.int8)

application_df['OCCUPATION_TYPE'] = application_df.groupby(['NAME_INCOME_TYPE','NAME_EDUCATION_TYPE'])['OCCUPATION_TYPE'].transform(lambda x:x.fillna(np.random.choice(x.mode()) if not x.mode().empty else 'unknown'))

application_df = application_df.rename(columns={'CODE_GENDER':'Gender','CNT_CHILDREN':'No of Children',
                                                'AMT_INCOME_TOTAL':'Annual Income','NAME_INCOME_TYPE':'Job',
                                                'NAME_EDUCATION_TYPE':'Education','NAME_FAMILY_STATUS':'Family Status',
                                                'NAME_HOUSING_TYPE':'Housing','DAYS_BIRTH':'Age','DAYS_EMPLOYED':'Employment Period',
                                                'OCCUPATION_TYPE':'Occupation','CNT_FAM_MEMBERS':'No of Family members'})

application_df['Annual Income'] = np.log1p(application_df['Annual Income'])
categorical_cols = ["Gender", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "Job", "Education", "Family Status", "Housing", "Occupation"]
application_df[categorical_cols] = application_df[categorical_cols].astype('category')


# credit df 
begin_month = pd.DataFrame(credit_df.groupby('ID')['MONTHS_BALANCE'].agg('min'))
begin_month = begin_month.rename(columns={'MONTHS_BALANCE':'Account Age'}).abs()
combined_df = pd.merge(application_df,begin_month,how='inner',on='ID')
credit_df['defaulted'] = None

credit_df.loc[credit_df['STATUS'].isin(['2','3','4','5']),'defaulted'] = 'Yes'
defaulted_df = credit_df.groupby('ID').count()
defaulted_df.loc[defaulted_df['defaulted'] > 0,'defaulted'] = 1
defaulted_df.loc[defaulted_df['defaulted'] == 0,'defaulted'] = 0
defaulted_status = defaulted_df[['defaulted']]
full_data = pd.merge(combined_df,defaulted_status,how='left',on='ID')

full_data['Is High Risk'] = full_data['defaulted']
full_data = full_data.drop('defaulted',axis=1)
print(full_data['Is High Risk'].value_counts())
# print(ssss)

# print(combined_df)
# credit_df['defaulted'] = None
# credit_df.loc[credit_df['STATUS'].isin('2','3','4','5'),'defaulted'] = 'yes'
# print(credit_df.info())


