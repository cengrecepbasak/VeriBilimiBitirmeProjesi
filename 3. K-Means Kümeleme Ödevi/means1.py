#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os
import warnings
warnings.filterwarnings('ignore')

# Görseller için klasör oluştur
output_dir = "kumeleme_sonuclari"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"'{output_dir}' klasörü oluşturuldu")

# In[5]:


data = pd.read_csv('dava.csv')
print("Veri setinin orijinal sütun isimleri:")
print(data.columns.tolist())

# Index sütununu kaldır
data_clean = data.drop('Unnamed: 0', axis=1)

# Sütun isimlerini basitleştir
data_clean.columns = ['Case_Duration', 'Witnesses', 'Legal_Fees', 'Evidence_Items', 'Severity', 'Outcome']

print(f"\nDüzeltilmiş veri boyutu: {data_clean.shape}")
print(data_clean.head())

# ## GÖREV 1: Özellik Seçimi
features = ['Case_Duration', 'Witnesses', 'Legal_Fees', 'Evidence_Items', 'Severity']
X = data_clean[features]

# Özellikleri standardize etme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nÖzelliklerin boyutu: {X_scaled.shape}")

# ## GÖREV 2: Optimal Küme Sayısını Belirleme

# Elbow yöntemi ile optimal küme sayısı
inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Elbow grafiği
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, 'bo-', markersize=8, linewidth=2)
plt.xlabel('Küme Sayısı (k)')
plt.ylabel('Inertia (Within-cluster sum of squares)')
plt.title('Elbow Yöntemi - Optimal Küme Sayısı')
plt.grid(True, alpha=0.3)
plt.savefig(f'{output_dir}/elbow_method.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/elbow_method.pdf', bbox_inches='tight')
plt.show()

# Silhouette skorları
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Daha iyi yorumlanabilirlik için 3 küme seçelim
optimal_k = 3

print(f"\nSeçilen optimal küme sayısı: {optimal_k}")

# ## GÖREV 3: K-Means ile Kümeleme

print(f"\nK-Means kümeleme işlemi (k={optimal_k})...")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

data_clean['Cluster'] = cluster_labels

print("\nKüme Dağılımı:")
cluster_distribution = data_clean['Cluster'].value_counts().sort_index()
print(cluster_distribution)

print("\nKüme Özellikleri (Ortalamalar):")
cluster_summary = data_clean.groupby('Cluster')[features + ['Outcome']].mean()
print(cluster_summary.round(2))

# ## GÖREV 4: Görselleştirme ve Yorumlama

print("\nGelişmiş Görselleştirme...")

# 1. ANA GÖRSELLEŞTİRME PANELİ
plt.figure(figsize=(20, 15))

# 1.1 Ana Özelliklerin Dağılımı
plt.subplot(3, 3, 1)
scatter = plt.scatter(data_clean['Legal_Fees'], data_clean['Case_Duration'], 
                     c=data_clean['Cluster'], cmap='Set2', alpha=0.8, s=80, edgecolors='black')
plt.xlabel('Hukuk Maliyetleri (USD)')
plt.ylabel('Dava Süresi (Gün)')
plt.title('Kümeler: Maliyet vs Süre')
plt.colorbar(scatter, label='Küme')
plt.grid(True, alpha=0.3)

# 1.2 Tanık ve Delil Dağılımı
plt.subplot(3, 3, 2)
scatter = plt.scatter(data_clean['Witnesses'], data_clean['Evidence_Items'], 
                     c=data_clean['Cluster'], cmap='Set2', alpha=0.8, s=80, edgecolors='black')
plt.xlabel('Tanık Sayısı')
plt.ylabel('Delil Sayısı')
plt.title('Kümeler: Tanık vs Delil')
plt.colorbar(scatter, label='Küme')
plt.grid(True, alpha=0.3)

# 1.3 Küme Büyüklükleri
plt.subplot(3, 3, 3)
cluster_counts = data_clean['Cluster'].value_counts().sort_index()
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
plt.pie(cluster_counts.values, labels=[f'Küme {i}' for i in cluster_counts.index], 
        autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('Kümelere Göre Dava Dağılımı')

# 1.4 Ciddiyet Dağılımı
plt.subplot(3, 3, 4)
severity_by_cluster = pd.crosstab(data_clean['Cluster'], data_clean['Severity'])
severity_by_cluster.plot(kind='bar', ax=plt.gca(), color=['#FF9999', '#66B2FF', '#99FF99'])
plt.xlabel('Küme')
plt.ylabel('Dava Sayısı')
plt.title('Kümelere Göre Ciddiyet Düzeyi')
plt.legend(title='Ciddiyet', labels=['Düşük', 'Orta', 'Yüksek'])
plt.xticks(rotation=0)

# 1.5 Outcome Dağılımı
plt.subplot(3, 3, 5)
outcome_by_cluster = pd.crosstab(data_clean['Cluster'], data_clean['Outcome'])
outcome_by_cluster.plot(kind='bar', ax=plt.gca(), color=['#FF6B6B', '#4ECDC4'])
plt.xlabel('Küme')
plt.ylabel('Dava Sayısı')
plt.title('Kümelere Göre Sonuç Dağılımı')
plt.legend(title='Sonuç', labels=['Aleyhte', 'Lehte'])
plt.xticks(rotation=0)

# 1.6 Özellik Ortalamaları Karşılaştırması
plt.subplot(3, 3, 6)
cluster_means = data_clean.groupby('Cluster')[['Legal_Fees', 'Case_Duration']].mean()
cluster_means.plot(kind='bar', ax=plt.gca(), color=['#FF6B6B', '#4ECDC4'])
plt.xlabel('Küme')
plt.ylabel('Ortalama Değer')
plt.title('Kümelere Göre Maliyet ve Süre Ortalamaları')
plt.legend(title='Özellik', labels=['Maliyet (USD)', 'Süre (Gün)'])
plt.xticks(rotation=0)

# 1.7 Witnesses vs Legal Fees
plt.subplot(3, 3, 7)
scatter = plt.scatter(data_clean['Witnesses'], data_clean['Legal_Fees'], 
                     c=data_clean['Cluster'], cmap='Set2', alpha=0.8, s=80, edgecolors='black')
plt.xlabel('Tanık Sayısı')
plt.ylabel('Hukuk Maliyetleri (USD)')
plt.title('Kümeler: Tanık vs Maliyet')
plt.colorbar(scatter, label='Küme')
plt.grid(True, alpha=0.3)

# 1.8 Evidence vs Duration
plt.subplot(3, 3, 8)
scatter = plt.scatter(data_clean['Evidence_Items'], data_clean['Case_Duration'], 
                     c=data_clean['Cluster'], cmap='Set2', alpha=0.8, s=80, edgecolors='black')
plt.xlabel('Delil Sayısı')
plt.ylabel('Dava Süresi (Gün)')
plt.title('Kümeler: Delil vs Süre')
plt.colorbar(scatter, label='Küme')
plt.grid(True, alpha=0.3)

# 1.9 Küme Özellik Karşılaştırması
plt.subplot(3, 3, 9)
cluster_features = data_clean.groupby('Cluster')[['Witnesses', 'Evidence_Items', 'Severity']].mean()
cluster_features.plot(kind='bar', ax=plt.gca(), color=['#FF9999', '#66B2FF', '#99FF99'])
plt.xlabel('Küme')
plt.ylabel('Ortalama Değer')
plt.title('Diğer Özellik Ortalamaları')
plt.legend(title='Özellik', labels=['Tanık', 'Delil', 'Ciddiyet'])
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(f'{output_dir}/ana_kumeleme_paneli.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/ana_kumeleme_paneli.pdf', bbox_inches='tight')
plt.show()

# 2. KORELASYON MATRİSİ
plt.figure(figsize=(12, 10))
correlation_matrix = data_clean[features + ['Outcome', 'Cluster']].corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, fmt='.2f', cbar_kws={'shrink': 0.8},
            mask=mask)
plt.title('Özellikler ve Kümeler Arası Korelasyon Matrisi', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(f'{output_dir}/korelasyon_matrisi.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/korelasyon_matrisi.pdf', bbox_inches='tight')
plt.show()

# 3. PAIRPLOT GÖRSELLEŞTİRMESİ
plt.figure(figsize=(16, 12))
pairplot = sns.pairplot(data_clean, vars=features, hue='Cluster', 
                        palette='Set2', diag_kind='hist', 
                        plot_kws={'alpha': 0.7, 's': 60, 'edgecolor': 'black'})
pairplot.fig.suptitle('Özelliklerin Kümelere Göre Dağılımı', y=1.02, fontsize=16)
plt.savefig(f'{output_dir}/pairplot_kumeleme.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/pairplot_kumeleme.pdf', bbox_inches='tight')
plt.show()

# 4. RADAR CHART GÖRSELLEŞTİRMESİ
from math import pi

def normalize_features(df, features):
    normalized = df[features].copy()
    for feature in features:
        normalized[feature] = (df[feature] - df[feature].min()) / (df[feature].max() - df[feature].min())
    return normalized

# Radar chart için veri hazırla
cluster_means_normalized = data_clean.groupby('Cluster')[features].mean()
cluster_means_normalized = normalize_features(cluster_means_normalized, features)

# Radar chart
fig = plt.figure(figsize=(18, 6))
categories = ['Dava Süresi', 'Tanık Sayısı', 'Hukuk Maliyetleri', 'Delil Sayısı', 'Ciddiyet Düzeyi']
N = len(categories)

for cluster in range(optimal_k):
    values = cluster_means_normalized.loc[cluster].values.tolist()
    values += values[:1]  # Çemberi tamamlamak için
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    ax = plt.subplot(1, optimal_k, cluster+1, polar=True)
    plt.xticks(angles[:-1], categories, color='grey', size=10)
    ax.plot(angles, values, linewidth=3, linestyle='solid', label=f'Küme {cluster}', 
            color=colors[cluster])
    ax.fill(angles, values, alpha=0.25, color=colors[cluster])
    plt.title(f'Küme {cluster} Profili', size=14, color='navy', y=1.1)
    plt.ylim(0, 1)

plt.tight_layout()
plt.savefig(f'{output_dir}/radar_chart_kume_profilleri.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/radar_chart_kume_profilleri.pdf', bbox_inches='tight')
plt.show()

# 5. KÜME BAŞARI ORANLARI GRAFİĞİ
plt.figure(figsize=(10, 6))
outcome_percentage = pd.crosstab(data_clean['Cluster'], data_clean['Outcome'], normalize='index') * 100
outcome_percentage.plot(kind='bar', stacked=True, color=['#FF6B6B', '#4ECDC4'], ax=plt.gca())
plt.xlabel('Küme')
plt.ylabel('Yüzde (%)')
plt.title('Kümelere Göre Lehte/Aleyhte Sonuç Dağılımı')
plt.legend(title='Sonuç', labels=['Aleyhte', 'Lehte'])
plt.xticks(rotation=0)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{output_dir}/kume_basari_oranlari.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/kume_basari_oranlari.pdf', bbox_inches='tight')
plt.show()

# 6. 3D GÖRSELLEŞTİRME
try:
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(15, 10))
    
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data_clean['Legal_Fees'], 
                        data_clean['Case_Duration'], 
                        data_clean['Witnesses'],
                        c=data_clean['Cluster'], 
                        cmap='Set2', 
                        s=80, 
                        alpha=0.8,
                        edgecolors='black')
    
    ax.set_xlabel('Hukuk Maliyetleri (USD)')
    ax.set_ylabel('Dava Süresi (Gün)')
    ax.set_zlabel('Tanık Sayısı')
    ax.set_title('3D Kümeleme Görselleştirmesi\n(Maliyet vs Süre vs Tanık)')
    plt.colorbar(scatter, label='Küme')
    plt.savefig(f'{output_dir}/3d_kumeleme.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/3d_kumeleme.pdf', bbox_inches='tight')
    plt.show()
except ImportError:
    print("3D görselleştirme için mpl_toolkits gerekli")

# ## DETAYLI KÜME PROFİL ANALİZİ
print("\n" + "="*70)
print("DETAYLI KÜME PROFİL ANALİZİ")
print("="*70)

for cluster in range(optimal_k):
    cluster_data = data_clean[data_clean['Cluster'] == cluster]
    print(f"\n🔍 --- KÜME {cluster} PROFİLİ ---")
    print(f"   Dava Sayısı: {len(cluster_data)} ({len(cluster_data)/len(data_clean)*100:.1f}%)")
    print(f"   📅 Ortalama Dava Süresi: {cluster_data['Case_Duration'].mean():.1f} gün")
    print(f"   👥 Ortalama Tanık Sayısı: {cluster_data['Witnesses'].mean():.1f}")
    print(f"   💰 Ortalama Hukuk Maliyeti: ${cluster_data['Legal_Fees'].mean():.2f}")
    print(f"   📋 Ortalama Delil Sayısı: {cluster_data['Evidence_Items'].mean():.1f}")
    print(f"   ⚖️  Ortalama Ciddiyet: {cluster_data['Severity'].mean():.2f}")
    print(f"   ✅ Lehte Sonuç Oranı: {cluster_data['Outcome'].mean():.2%}")

# ## SONUÇLARI CSV OLARAK KAYDET
data_clean.to_csv(f'{output_dir}/kumeleme_sonuclari.csv', index=False)
cluster_summary.to_csv(f'{output_dir}/kume_ozet_istatistikleri.csv')

print(f"\n📁 SONUÇLAR KAYDEDİLDİ:")
print(f"   ✅ Görseller: '{output_dir}' klasörüne kaydedildi")
print(f"   ✅ Veriler: '{output_dir}/kumeleme_sonuclari.csv'")
print(f"   ✅ İstatistikler: '{output_dir}/kume_ozet_istatistikleri.csv'")

# ## FİNAL RAPOR
print("\n" + "="*80)
print("KÜMELEME ANALİZİ RAPORU")
print("="*80)

print(f"""
📊 ANALİZ ÖZETİ:

• Toplam Dava Sayısı: {len(data_clean)}
• Kullanılan Özellikler: {len(features)}
• Optimal Küme Sayısı: {optimal_k}
• Silhouette Skoru: {silhouette_score(X_scaled, cluster_labels):.4f}

🎯 KÜME PROFİLLERİ:

{cluster_summary.round(2)}

💡 İŞ İÇGÖRÜLERİ:

1. KAYNAK OPTİMİZASYONU: Yüksek maliyetli kümeler için özel kaynak ayırın
2. RİSK YÖNETİMİ: Uzun süreli ve yüksek maliyetli davaları yakından takip edin  
3. SÜREÇ İYİLEŞTİRME: Benzer profildeki davalar için standart süreçler geliştirin
4. BAŞARI ANALİZİ: Lehte sonuç oranlarını kümeler bazında değerlendirin

📈 GÖRSELLER:
• {output_dir}/ana_kumeleme_paneli.png - Ana analiz paneli
• {output_dir}/korelasyon_matrisi.png - Korelasyon analizi
• {output_dir}/pairplot_kumeleme.png - Detaylı dağılım analizi
• {output_dir}/radar_chart_kume_profilleri.png - Küme profilleri
• {output_dir}/kume_basari_oranlari.png - Başarı oranları

✅ ANALİZ TAMAMLANDI!
""")