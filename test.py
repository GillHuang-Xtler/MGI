import pandas as pd
df = pd.read_csv("./data/celeba/list_attr_celeba.csv")
df = df[['image_igd', 'Smiling']]