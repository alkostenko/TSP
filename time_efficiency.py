import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

time_data=pd.read_csv('time.csv')
print(time_data)
fig=sns.lineplot(data=time_data, x='NumberOfPoints', y='Time', hue='Method', palette='Dark2')
plt.title("Methods' time consumption")
plt.show()  

methods=['BruteForce', 'LexicographicOrder', 'GeneticAlgorithm', 'AntColonyACS', 'AntColonyElitist', 'AntColonyMaxMin']

for method in methods:
    fig=sns.lineplot(data=time_data[time_data['Method']==f'{method}'], x='NumberOfPoints', y='Time', hue='Method', palette='Dark2')
    plt.title(f'{method}')
    plt.show()