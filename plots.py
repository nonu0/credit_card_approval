import matplotlib.pyplot as plt


bins = [0, 1, 2, 3, 4, 5, 6, 7, 8,9,10,11,12,13,14,15,16,17,18,19,20]
plt.hist(app_df['CNT_CHILDREN'],bins=bins,density=False,alpha=0.9,edgecolor='black')
# sns.kdeplot(app_df['Family members'],color='red',lw=2)
plt.title('Children Distribution')
plt.xlabel('Children members')
plt.ylabel('frequency')
plt.show()


ax = sns.boxplot(data={'Family members':app_df['Family members'],'CNT_CHILDREN':app_df['CNT_CHILDREN']},orient='h')
data={'Family members':app_df['Family members'],'CNT_CHILDREN':app_df['CNT_CHILDREN']}
for i, category in enumerate(['Family members','CNT_CHILDREN']):
    Q1 = np.percentile(app_df[category],25)
    Q3 = np.percentile(app_df[category],75)
    # print(Q1,Q3)
    IQR = Q3 - Q1
    # print(IQR)
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # print(lower_bound,upper_bound)
    outliers = app_df[(app_df[category] < lower_bound) | (app_df[category] > upper_bound)][category]
    print(outliers.sum())
    # print(app_df.sum())
    for outlier in outliers:
        ax.text(outlier, i, f'{outlier:.1f}', ha='left',va='center',color='red',fontsize=12)
        
        
# sample_id = credit_df['ID'].sample(1).values[0]
sample_id = 5001715
sample_data = credit_df[credit_df['ID'] == sample_id]
# print(sample_data)

sns.lineplot(data=sample_data, x='MONTHS_BALANCE',y='STATUS',marker='o')
plt.title(f'credit history for ID:{sample_id}')
plt.xlabel('MONTHS_BALANCE')
plt.ylabel('STATUS')
plt.show()


# Plot new distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(application_df["No of Children"], bins=20, kde=True, ax=axes[0])
sns.histplot(application_df["No of Family members"], bins=20, kde=True, ax=axes[1])
plt.show()