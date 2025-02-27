import pandas as pd


app_df = pd.read_csv(r'C:\Users\Administrator\work\credit_card_elig\core\credit_card_approval\data\application_record.csv')
credit_df = pd.read_csv(r'C:\Users\Administrator\work\credit_card_elig\core\credit_card_approval\data\credit_record.csv')

print(app_df.value_counts().sum())
# print(app_df.isnull().sum())
occ = app_df['OCCUPATION_TYPE']
id = app_df['ID']
# print(occ.unique())
# print(occ.nunique())
# print(occ.value_counts(dropna=False))
# print(occ.value_counts(dropna=False,normalize=True) *100)
duplicates = id.duplicated().value_counts()
# duplicates = duplicates_c[duplicates_c > 1]
print(duplicates)
# print(id.nunique())
# print(id.count())
# print()
# dup = app_df[app_df.duplicated(subset=['ID'], keep=False)]
app_df.drop_duplicates(subset='ID',inplace=True)
print(app_df)
print(app_df.duplicated(subset='ID').value_counts())
# print(dup.head(30))
# print(app_df[app_df.duplicated(subset=['ID'], keep=True)])

# occ_sch = app_df[['OCCUPATION_TYPE','NAME_EDUCATION_TYPE','NAME_INCOME_TYPE']]
# occ_sch.set_index([''])
# print(occ_sch.head(20))