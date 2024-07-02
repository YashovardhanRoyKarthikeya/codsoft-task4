from google.colab import files
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

uploaded=files.upload()
t=[]
with open('/content/groceryitems.csv') as file:
    temp=[i.strip() for i in file.readlines()]
    for i in temp:
        t.append(i.split(','))
x=TransactionEncoder()
y=x.fit_transform(t)
dataFrame = pd.DataFrame(y,columns=x.columns_)

item=apriori(dataFrame,min_support=0.02,use_colnames=True)
item.sample(10,random_state=45)

ar=association_rules(item,metric='lift',min_threshold=1)
ar.sample(10)
print()
print("Based on your previous purchases, we recommend the following items for you:")
ar.sort_values('confidence',ascending=False)[0:50]
