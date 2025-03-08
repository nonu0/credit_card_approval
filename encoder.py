

class CustomEncoder():
    def __init__(self):
        self.classes = {}
        self.inverse_classes = {}
        self.fitted = True
        
    def fit(self,categories):
        unique_classes = sorted(set(categories))
        self.classes = {label:idx for idx,label in enumerate(unique_classes)}
        self.inverse_classes = {idx:label for idx,label in enumerate(unique_classes)}
        
        return self.inverse_classes
    
    def transform(self,data):
        if not self.fitted:
            print('Data not encoded')
        return [self.classes[label] for label in data]
    
    def inverse_transform(self,data):
        if not self.fitted:
            print('Data not encoded')
        return [self.inverse_classes[label] for label in data]
        

# example use case

# categories = ['dog', 'cat', 'rabbit', 'dog', 'cat', 'rabbit']

# if __name__ == __ma
# encoder = CustomEncoder()
# print(encoder.fit(categories))
# encoded = encoder.transform(categories)
# print(encoded)
# print(encoder.inverse_transform(encoded))