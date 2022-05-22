import pandas as pd

df = pd.read_csv('Attributes.csv')
df2 = pd.read_csv('OneHotEncodedClasses.csv')
df3 = pd.concat([df, df2], axis=1,)
print(df)
print(df2)
print(df3)

ds = df3.sample(frac=1)


ds_sol1 = pd.concat([ds["sepal.length"], ds["sepal.width"], ds["petal.length"], ds["petal.width"]], axis=1,)
ds_sol2 = pd.concat([ds["Setosa"], ds["Versicolor"], ds["Virginica"]], axis=1,)

print(ds_sol1)
print(ds_sol2)

ds_sol1.to_csv('./ds_sol1.csv', index=False)
ds_sol2.to_csv('./ds_sol2.csv', index=False)
