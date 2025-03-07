import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
"""
take a df,list or numpy array, split into train and test
"""

def custom_train_test_split(X,y,test_size=0.2,random_state=None,stratify=None):
    if random_state is not None:
        np.random.seed(random_state)
        
    num_samples = len(X)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    print(indices)

    test_count = int(num_samples * test_size)
    
    if stratify is not None:
        sss = StratifiedShuffleSplit(n_splits=1,test_size=test_size,random_state=None)
        train_idx,test_idx = next(sss.split(X,stratify))
    else:
        train_idx,test_idx = indices[:-test_count],indices[-test_count:]
    return X[train_idx],X[test_idx],y[train_idx],y[test_idx]

X = np.random.rand(100,5)
y = np.random.randint(0,2,100)

# X_train,X_test,y_train,y_test = custom_train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
print(custom_train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
)
# print(f'X_train:{X_train}')
# print(f'X_train:{len(X_train)}')
# print(f'X_test:{X_test}')
# print(f'y_train:{y_train}')
# print(f'y_test:{y_test}')
# print(f'y_test:{len(y_test)}')