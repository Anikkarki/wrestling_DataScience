import pandas as pd
import matplotlib.pyplot as plt

#read the csv file and assign it to df
df = pd.read_csv('WWE_History_1000.csv')

# Count wins
wins = df['Winner'].value_counts()
print("Wins:\n", wins)

# Count lost
loose = df['Loser'].value_counts()
print("Lost :\n", loose)

# Bar chart
wins.plot(kind='bar', figsize=(10,5))
plt.title("Most Wins")
plt.xlabel("Wrestler")
plt.ylabel("Number of Wins")
plt.show()

#density chart
loose.plot (kind='box', figsize=(10,5))
plt.show()
