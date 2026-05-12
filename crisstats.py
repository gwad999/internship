import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
file_path="/home/akku/assignment3/data.csv"
df=pd.read_csv(file_path)
print(df.shape)
print(df.columns)
print(df.head(10))
print(df.info()) 
pd.DataFrame(df.apply(lambda column: len(column.unique())), columns=['Unique Values'])
df.describe().T
pd.DataFrame(df.apply(lambda column: column.isnull().sum()), columns=['Null Values Count']) 

#data visualization



#position played and contributed



plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Playing_Position', order=df['Playing_Position'].value_counts().index)
plt.title('Distribution of Playing Positions')
plt.xticks(rotation=45)
plt.ylabel('Count')
plt.savefig('playing_positions_distribution.png')

 #goals scored in differnt clubs


goals_per_club = df['Club'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=goals_per_club.index, y=goals_per_club.values)
plt.title('Total Goals by Club')
plt.xticks(rotation=45)
plt.ylabel('Goals Count')
plt.savefig('goals_by_club.png')

#number of matches in each seasons

matches_per_season = df['Season'].value_counts().sort_index()
plt.figure(figsize=(20, 6))
sns.lineplot(x=matches_per_season.index, y=matches_per_season.values, marker='o')
plt.title('Number of Matches per Season')
plt.xticks(rotation=45)
plt.ylabel('Number of Matches')
plt.savefig('matches_per_season.png')

#type of goals scored in pie chart





plt.figure(figsize=(8, 8))
df['Type'].value_counts().plot.pie(autopct='%10.1f%%', startangle=180)
plt.title('Goals by Type')
plt.ylabel('')
plt.savefig('goals_by_type.png')

#goals by type in bar chart

goals_by_type = df['Type'].value_counts()

# Create a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=goals_by_type.index, y=goals_by_type.values)
plt.title('Goals by Type')
plt.xlabel('Type')
plt.ylabel('Count of Goals')
plt.xticks(rotation=45)
plt.savefig('goals_by_type_bar.png')    
