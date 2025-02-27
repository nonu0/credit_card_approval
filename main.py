import pandas as pd


app_df = pd.read_csv(r'C:\Users\Administrator\work\credit_card_elig\core\credit_card_approval\data\application_record.csv')
credit_df = pd.read_csv(r'C:\Users\Administrator\work\credit_card_elig\core\credit_card_approval\data\credit_record.csv')


app_df.drop_duplicates(subset='ID',inplace=True)

print(app_df.value_counts().sum())
print(app_df.duplicated().sum())