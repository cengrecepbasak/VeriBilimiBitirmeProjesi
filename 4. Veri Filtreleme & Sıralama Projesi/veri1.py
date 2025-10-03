#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


data = pd.read_csv('country.csv')
data


# ##  Country.csv dosyasının özelliği
# Bu tablo, çeşitli ülkelerle ilgili bir dizi demografik, ekonomik ve coğrafi veriyi içermektedir. Tabloda her bir satır bir ülkeyi temsil ederken, sütunlar bu ülkelerle ilgili farklı özellikleri gösterir. İşte sütunların anlamları:
# 
# Country: Ülkenin adı.  
# Region: Ülkenin bulunduğu bölge (örneğin, Asya, Doğu Avrupa).  
# Population: Ülkenin toplam nüfusu.  
# Area (sq. mi.): Ülkenin yüzölçümü (mil kare olarak).  
# Pop. Density (per sq. mi.): Nüfus yoğunluğu (mil kare başına düşen kişi sayısı).  
# Coastline (coast/area ratio): Sahil uzunluğunun, ülkenin toplam alanına oranı.  
# Net migration: Net göç oranı (göçmenlerin ülkeye giren veya ülkeden çıkan kişi sayısına göre oranı).  
# Infant mortality (per 1000 births): Bebek ölüm oranı (1000 doğum başına).  
# GDP ($ per capita): Kişi başına düşen Gayri Safi Yurtiçi Hasıla (GSYİH).  
# Literacy (%): Okur-yazarlık oranı.  
# Phones (per 1000): Her 1000 kişi başına düşen telefon sayısı.  
# Arable (%): Tarıma elverişli arazi yüzdesi.  
# Crops (%): Ekilebilir ürünlerin yüzdesi.  
# Other (%): Diğer arazi kullanımı yüzdesi.  
# Climate: Ülkenin iklim kategorisi (numerik bir değer olarak gösterilmiş).  
# Birthrate: Doğum oranı.  
# Deathrate: Ölüm oranı.  
# Agriculture: Tarım sektörünün ekonomideki payı.  
# Industry: Sanayi sektörünün ekonomideki payı.  
# Service: Hizmet sektörünün ekonomideki payı.  
# 

# ## Bu Dosyada Yapacağınız görevleri alt taraftan bakabilirsiniz.

# ## 1. Görev : Nüfusa Göre Azalan Sırada Sıralama:

# In[4]:


# Nüfusa Göre Azalan Sırada Sıralama kodunu buraya yazınız
nufusa_gore_siralama = data.sort_values('Population', ascending=False)
print("Nüfusa Göre Azalan Sırada Ülkeler:")
print(nufusa_gore_siralama[['Country', 'Population']].head())
print("\n")


# ## 2. Görev: GDP per capita sütununa göre ülkeleri artan sırada sıralamak(Kişi başına düşen Gayri Safi Yurtiçi Hasıla).

# In[5]:


# GDP per capita sütununa göre ülkeleri artan sırada sıralamak(Kişi başına düşen Gayri Safi Yurtiçi Hasıla). kodunu buradan yazınız.
gdp_artan_sira = data.sort_values('GDP ($ per capita)')
print("GDP per capita'ya Göre Artan Sırada Ülkeler:")
print(gdp_artan_sira[['Country', 'GDP ($ per capita)']].head())
print("\n")


# ## 3. Görev: Population sütunu 10 milyonun üzerinde olan ülkeleri seçmek.

# In[6]:


# Kodunu buraya yazınız.
nufus_10_milyon_ustu = data[data['Population'] > 10000000]
print("Nüfusu 10 Milyonun Üzerinde Olan Ülkeler:")
print(nufus_10_milyon_ustu[['Country', 'Population']])
print("\n")


# ## 4. Görev: Literacy (%) sütununa göre ülkeleri sıralayıp, en yüksek okur-yazarlık oranına sahip ilk 5 ülkeyi seçmek.

# In[7]:


# Kodunu buraya yazınız.
literacy_siralama = data.sort_values('Literacy (%)', ascending=False)
en_yuksek_literacy = literacy_siralama.head(5)
print("En Yüksek Okur-Yazarlık Oranına Sahip İlk 5 Ülke:")
print(en_yuksek_literacy[['Country', 'Literacy (%)']])
print("\n")


# ## 5. Görev:  Kişi Başı GSYİH 10.000'in Üzerinde Olan Ülkeleri Filtreleme: GDP ( per capita) sütunu 10.000'in üzerinde olan ülkeleri seçmek.

# In[8]:


# Kodunu buraya yazınız.
gdp_10000_ustu = data[data['GDP ($ per capita)'] > 10000]
print("GDP per capita'sı 10.000'in Üzerinde Olan Ülkeler:")
print(gdp_10000_ustu[['Country', 'GDP ($ per capita)']])
print("\n")


# ## Görev 6 : En Yüksek Nüfus Yoğunluğuna Sahip İlk 10 Ülkeyi Seçme:
# Pop. Density (per sq. mi.) sütununa göre ülkeleri sıralayıp, en yüksek nüfus yoğunluğuna sahip ilk 10 ülkeyi seçmek.

# In[9]:


# Kodunu buraya yazınız.
nufus_yogunlugu_siralama = data.sort_values('Pop. Density (per sq. mi.)', ascending=False)
en_yuksek_nufus_yogunlugu = nufus_yogunlugu_siralama.head(10)
print("En Yüksek Nüfus Yoğunluğuna Sahip İlk 10 Ülke:")
print(en_yuksek_nufus_yogunlugu[['Country', 'Pop. Density (per sq. mi.)']])