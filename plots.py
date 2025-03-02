import matplotlib.pyplot as plt


bins = [0, 1, 2, 3, 4, 5, 6, 7, 8,9,10,11,12,13,14,15,16,17,18,19,20]
plt.hist(app_df['CNT_CHILDREN'],bins=bins,density=False,alpha=0.9,edgecolor='black')
# sns.kdeplot(app_df['Family members'],color='red',lw=2)
plt.title('Children Distribution')
plt.xlabel('Children members')
plt.ylabel('frequency')
plt.show()