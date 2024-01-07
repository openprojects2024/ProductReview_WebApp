import pandas as pd
import numpy as np

#product = {"Product_Name":["IPhone 15 Pro", " Samsung Refrigerator", "Oppo Reno 5G","American Tourist Bag"]}

#data = pd.read_csv("product_review.csv")
#data2 = pd.read_csv("image_path.csv")

#data.to_csv('product_review.csv',)
#data[data['Product_Name']=="IPhone 15 Pro"]['image_path'] = "C:/Users/krishnendu/VSCODE_projects/ML-recruitment-assessment/products/iphone15_pro"
Review_customer = {"Product_Name":[],"Review":[],"Sentiment":[]}
Review_customer = pd.DataFrame(Review_customer)

Review_customer.loc[Review_customer.shape[0]] = {"Product_Name":"Iphone","Review":"good","Sentiment":"ok ok"}
print(Review_customer)

#print(data3)

