import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

app_df = pd.read_csv(r'C:\Users\Administrator\work\credit_card_elig\core\credit_card_approval\data\application_record.csv')
credit_df = pd.read_csv(r'C:\Users\Administrator\work\credit_card_elig\core\credit_card_approval\data\credit_record.csv')

# print(app_df.value_counts().sum())
# print(app_df.isnull().sum())
occ = app_df['OCCUPATION_TYPE']
id = app_df['ID']
# print(occ.unique())
# print(occ.nunique())
# print(occ.value_counts(dropna=False))
# print(occ.value_counts(dropna=False,normalize=True) *100)
# duplicates = id.duplicated().value_counts()
duplicate_counts = app_df["ID"].value_counts()
duplicates = duplicate_counts[duplicate_counts > 1]
print(duplicates.sum())
sss = app_df[app_df['ID'].isin(duplicates.index)].sort_values(by='ID')
# print(sss.head(30))
# print(sss.iloc[1:7,:])
# print(sss.iloc[5,:])
# print(sss.iloc[1,:])
# print(sss.iloc[[1,5]])
# duplicates = duplicates_c[duplicates_c > 1]
# print(duplicates)
# print(id.nunique())
# print(id.count())
# print()
# dup = app_df[app_df.duplicated(subset=['ID'], keep=False)]
app_df.drop_duplicates(subset='ID',inplace=True)
# print(app_df)
# print(app_df.duplicated(subset='ID').value_counts())
# print(dup.head(30))
# print(app_df[app_df.duplicated(subset=['ID'], keep=True)])

occ_sch = app_df[['OCCUPATION_TYPE','NAME_EDUCATION_TYPE','NAME_INCOME_TYPE']]
# occ_sch.set_index([''])
# print(occ_sch.head(20))

cross = pd.crosstab(app_df['OCCUPATION_TYPE'],app_df['NAME_EDUCATION_TYPE'])
print(cross)
# print(occ.nunique())
# print(occ.values)
# print(occ.value_counts(dropna=False) / len(app_df) * 100)

# print(app_df.info())

plt.figure(figsize=(10,7))
sns.heatmap(cross,annot=True,fmt='d',cmap='coolwarm')
plt.xticks(rotation=45)
plt.show()