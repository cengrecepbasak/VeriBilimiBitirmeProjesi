# 50_Startups veri görselleştirme ödevi
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Veri yükleme
rec = pd.read_csv("50_Startups.csv")

# 1. R&D harcaması ile kâr arasındaki ilişki (scatter plot)
num1 = rec['R&D Spend']
num2 = rec['Profit']
plt.figure(figsize=(8,5))
plt.scatter(num1, num2, color='blue')
plt.title("R&D Harcaması vs Kâr")
plt.xlabel("R&D Harcaması")
plt.ylabel("Kâr")
plt.grid(True)
plt.show()

# 2. Yönetim harcaması ile kâr arasındaki ilişki (scatter plot)
num1 = rec['Administration']
num2 = rec['Profit']
plt.figure(figsize=(8,5))
plt.scatter(num1, num2, color='green')
plt.title("Yönetim Harcaması vs Kâr")
plt.xlabel("Yönetim Harcaması")
plt.ylabel("Kâr")
plt.grid(True)
plt.show()

# 3. Eyaletlere göre ortalama kârlar (bar chart)
ort = rec.groupby('State')['Profit'].mean()
plt.figure(figsize=(8,5))
ort.plot(kind='bar', color='orange')
plt.title("Eyaletlere Göre Ortalama Kârlar")
plt.xlabel("Eyalet")
plt.ylabel("Ortalama Kâr")
plt.grid(axis='y')
plt.show()

# 4. Harcama dağılımları (boxplot)
bas = rec[['R&D Spend', 'Administration', 'Marketing Spend']]
plt.figure(figsize=(8,5))
sns.boxplot(data=bas)
plt.title("Harcama Dağılımları")
plt.show()
