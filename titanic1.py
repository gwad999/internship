import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path="/home/akku/assingment2/titanic.csv"
df=pd.read_csv("titanic.csv")
print(df.head(10))
print(df.info())
#counting the duplicate values
print(df.duplicated().value_counts())
# Check how many missing values exist in each column
print(df.isnull().sum())
missing = df.isnull().sum()
# Create the bar plot
ax = missing.plot(kind='bar', color='skyblue')
plt.title('Missing Values per Column')
plt.ylabel('Count of Missing Values')
plt.xlabel('Column')
plt.xticks(rotation=45)

for index, value in enumerate(missing):
    plt.text(index, value + 5, str(value), ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('missing_values.png')
df = df.drop(columns=['Cabin'])  # Cabin has 687 missing — too many!
df['Age'] = df['Age'].fillna(df['Age'].median())
# Count the number of people who survived and didn't
survival_counts = df['Survived'].value_counts()

print("Survival Counts:\n", survival_counts)
sns.countplot(x='Survived', data=df, palette='pastel')
plt.title('Survival Count (0 = Died, 1 = Survived)')
plt.xlabel('Survived')
plt.ylabel('Passenger Count')
plt.savefig('survival_counts.png')
# Count of passengers by gender
gender_counts = df['Sex'].value_counts()
print("\nGender Counts:\n", gender_counts)
sns.countplot(x='Sex', hue='Survived', data=df, palette='Set2')
plt.title('Survival by Gender')
plt.xlabel('Gender')
plt.ylabel('Passenger Count')
plt.legend(title='Survived')
plt.savefig('survival_by_gender.png')
# Count of passengers by class
pclass_counts = df['Pclass'].value_counts()
print("\nPassenger Class Counts:\n", pclass_counts)
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 40, 60, 100], labels=['Child', 'Teen', 'Adult', 'Middle Age', 'Senior'])
sns.countplot(x='AgeGroup', hue='Survived', data=df)
plt.title('Survival by Age Group')
plt.savefig('survival_by_age_group.png')
