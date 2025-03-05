import pandas as pd

app_df = pd.read_csv(r'C:\Users\Administrator\work\credit_card_elig\core\credit_card_approval\data\application_record.csv')

app_df = pd.DataFrame(app_df)
age = app_df.loc[app_df['NAME_INCOME_TYPE'] == 'Pensioner',['DAYS_BIRTH','NAME_EDUCATION_TYPE','NAME_INCOME_TYPE']]
# print(app_df)
print(age)
# print(age.value_counts())