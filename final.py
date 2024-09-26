#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 12:25:23 2024

@author: youssrakacha
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 18:39:45 2024

@author: youssrakacha
"""

import os
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

#%%

print("Current working directory: {0}".format(os.getcwd()))

#%%

employee = pd.read_csv('/Users/youssrakacha/Downloads/Minor Data Science/Employee.csv', delimiter = ',')
performancerate = pd.read_csv('/Users/youssrakacha/Downloads/Minor Data Science/PerformanceRating.csv', delimiter = ',')

#combineren van de datasets 
performancerate['ReviewDate'] = pd.to_datetime(performancerate['ReviewDate'])  # Zorg dat de aanstellingsdatum in datetime-formaat is
recent_performance = performancerate.loc[performancerate.groupby('EmployeeID')['ReviewDate'].idxmax()]
combined_dataset =  pd.merge(employee, performancerate, on='EmployeeID')

#%%

combined_dataset.describe()

#%%

st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 40px;
        color: #0A1172;  /* Dark blue color */
        font-family: 'Arial', sans-serif;  /* You can change the font here */
    }
    </style>
    <h1 class="title">HR Dashboard: Medewerkerstevredenheid en attritie</h1>
    """, unsafe_allow_html=True)
    
    
    
#%%

combined_dataset['ReviewDate'] = pd.to_datetime(combined_dataset['ReviewDate'], errors='coerce')

# 1. Basisinformatie van de dataset
st.write("""
    Dit dashboard omvangt informatie over de werknemerstevredenheid en attritie. We beginnen eerst met inzicht geven over de werktevredenheidsscore
    en leggen vervolgens een verband met de attritie van de werknemers. We analyseren een uitgebreide dataset en gebruiken geavanceerde machine learning-modellen 
    om te voorspellen welke werknemers het grootste risico lopen te vertrekken. Je kunt patronen en verbanden ontdekken tussen variabelen zoals werkervaring, salaris, 
    werktevredenheid en promotiefrequentie, en zien hoe deze bijdragen aan de beslissing om te blijven of te vertrekken.
    
    Voor de HR-afdeling is dit dashboard een krachtig hulpmiddel om potentiële risico’s vroegtijdig te signaleren en gericht in te grijpen. D
    oor inzicht te krijgen in de belangrijkste drijfveren achter werknemerstevredenheid en vertrekrisico's, kan HR strategieën ontwikkelen om talent te behouden, 
    de werktevredenheid te verhogen en de algehele bedrijfsperformance te verbeteren
    """)

#%%

# Tevredenheidsmetrics
st.subheader('Algemene Tevredenheid van Medewerkers')

# Gemiddelde tevredenheidscijfers
col1, col2, col3, col4, col5 = st.columns(5)

avg_job_satisfaction = combined_dataset['JobSatisfaction'].mean()
avg_env_satisfaction = combined_dataset['EnvironmentSatisfaction'].mean()
avg_relation_satisfaction = combined_dataset['RelationshipSatisfaction'].mean()
avg_manager_rating = combined_dataset['ManagerRating'].mean()
avg_self_rating = combined_dataset['SelfRating'].mean()

# Metrics visualiseren
col1.metric("Job Tevredenheid", f"{avg_job_satisfaction:.2f}/5")
col2.metric("Omgeving Tevredenheid", f"{avg_env_satisfaction:.2f}/5")
col3.metric("Relatie Tevredenheid", f"{avg_relation_satisfaction:.2f}/5")
col4.metric("Leidinggevende Beoordeling", f"{avg_manager_rating:.2f}/5")
col5.metric("Zelf Beoordeling", f"{avg_self_rating:.2f}/5")

#%%

# Box plot voor job tevredenheid per functie
st.subheader("Job Tevredenheid per Functie")

fig_box = px.box(combined_dataset, x='JobRole', y='JobSatisfaction', title="Job Tevredenheid Verdeling per Functie", 
                 labels={'JobRole': 'Functie', 'JobSatisfaction': 'Job Tevredenheid'})
fig_box.update_layout(xaxis_title='Functie', yaxis_title='Job Tevredenheid')
st.plotly_chart(fig_box)


#%%
# 6. Extra: Distributie van numerieke kolommen
st.subheader("Distributie van numerieke kolommen")
numeric_columns = combined_dataset.select_dtypes(include=['float64', 'int64']).columns
selected_column = st.selectbox("Kies een numerieke kolom om te visualiseren:", numeric_columns)

# Voeg een slider toe waarmee de gebruiker het aantal bins kan instellen
bins = st.slider("Selecteer het aantal bins voor het histogram", min_value=5, max_value=50, value=20)

fig, ax = plt.subplots()
# Plot het histogram met de gekozen hoeveelheid bins
sns.histplot(combined_dataset[selected_column], bins=bins, kde=True, ax=ax)
ax.set_title(f"Histogram van {selected_column} (met {bins} bins)")
st.pyplot(fig)

#%%

combined_dataset['HireDate']=combined_dataset['HireDate'].astype('datetime64[ns]')

#%%

combined_dataset.insert(0, 'FullName', combined_dataset['FirstName'] + ' ' + combined_dataset['LastName'])  # Plaats op positie 0 (eerste kolom)

# Verwijder de originele kolommen 'FirstName', 'LastName' en andere onodige kolomen 
combined_dataset.drop(columns=['FirstName', 'LastName','SelfRating','ManagerRating'], inplace=True)

#%%

# Meerdere kolommen verwijderen
combined_dataset = combined_dataset.drop(columns=['EmployeeID', 'PerformanceID'])

#%%

# Bekijk de unieke waarden in de 'Gender' kolom
print(combined_dataset['Gender'].unique())
print(combined_dataset['BusinessTravel'].unique())
print(combined_dataset['Attrition'].unique())
print(combined_dataset['Department'].unique())

# Verwijder leidende en volgende spaties in 'BusinessTravel' en 'Gender'
combined_dataset['BusinessTravel'] = combined_dataset['BusinessTravel'].str.strip()
combined_dataset['Gender'] = combined_dataset['Gender'].str.strip()

# Omzetten van de volgende kolomen om in numerieke warden 
combined_dataset['Gender'] = combined_dataset['Gender'].map({"Prefer Not To Say" :0, "Male": 1, "Female": 2, "Non-Binary":3})
combined_dataset['BusinessTravel'] = combined_dataset['BusinessTravel'].map({"No Travel": 0, "Some Travel": 1, "Frequent Traveller": 2})
combined_dataset['Attrition'] = combined_dataset['Attrition'].map({'Yes': 1, 'No': 0})

#Een nieuwe variabele toevoegen aan dataset
combined_dataset['PromotionFrequency'] = (combined_dataset['YearsAtCompany'] / (combined_dataset['YearsSinceLastPromotion'] + 1)).round().astype(int)

st.subheader("Data filteren op basis van Attritie ")

attrition_yes = st.checkbox("Attrition: Ja", key="attrition_yes")
attrition_no = st.checkbox("Attrition: Nee", key="attrition_no", value=not attrition_yes)

# Zorg ervoor dat als de ene checkbox wordt geselecteerd, de andere wordt gedeselecteerd
if attrition_yes:
    attrition_no = False
elif attrition_no:
    attrition_yes = False

# Toepassen van filtering op basis van de checkboxes
if attrition_yes and not attrition_no:
    filtered_dataset = combined_dataset[combined_dataset['Attrition'] == 1]
elif attrition_no and not attrition_yes:
    filtered_dataset = combined_dataset[combined_dataset['Attrition'] == 0]
else:
    filtered_dataset = combined_dataset

# Toon de gefilterde data
st.write("Gefilterde data op basis van Attrition:")
st.write(filtered_dataset)

#%%

import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

#%%

# Bereken het percentage werknemers die het bedrijf verlaten of blijven
attrition_rate = combined_dataset['Attrition'].value_counts(normalize=True) * 100

# Titel en beschrijving toevoegen
st.subheader("Percentage werknemersattritie binnen de organisatie")
st.write("""
Deze tabel toont het percentage werknemers die het bedrijf verlaten ('1') of blijven ('0').
""")

# Weergeven van de resultaten in Streamlit
st.table(attrition_rate)

#%%

# Alleen correlaties met Attrition extraheren
# Eerst ervoor zorgen dat de 'Attrition' kolom goed is omgezet
if combined_dataset['Attrition'].dtype == 'object':
    combined_dataset['Attrition'] = combined_dataset['Attrition'].map({'No': 0, 'Yes': 1})

# Controleer ook of de andere kolommen die je wilt analyseren de juiste datatypes hebben
# Converteer waar nodig
numeric_columns = ['JobSatisfaction', 'WorkLifeBalance', 'Age', 
                   'YearsSinceLastPromotion', 'DistanceFromHome (KM)', 
                   'Salary', 'PromotionFrequency']

for col in numeric_columns:
    if combined_dataset[col].dtype == 'object':
        combined_dataset[col] = pd.to_numeric(combined_dataset[col], errors='coerce')

# Alleen numerieke waarden behouden en na de conversie de correlatie berekenen
corr_matrix = combined_dataset[['Attrition'] + numeric_columns].corr()

# Correlaties met Attrition extraheren
attrition_corr = corr_matrix['Attrition'].drop('Attrition')

# Correlatiedrempel instellen
threshold = 0.1
strong_corr = attrition_corr[(attrition_corr >= threshold) | (attrition_corr <= -threshold)]

# Tabel weergeven met sterke correlaties
st.subheader("Sterke Verbanden met Attrition")
st.write("""
Uit de bovenstaande correlatiematrix kunnen we een aantal belangrijke verbanden met betrekking tot werknemersattritie afleiden. Hieronder een overzicht van de sterkste correlaties:
""")

# Lijst van sterke correlaties in tabelvorm weergeven
st.table(strong_corr)

# Kort overzicht van de verbanden als tekst
if not strong_corr.empty:
    st.write("We zien dat de volgende factoren een sterke correlatie hebben met werknemersattritie:")
    for factor, corr_value in strong_corr.items():
        st.write(f"- **{factor}** heeft een correlatie van **{corr_value:.2f}** met werknemersattritie.")
else:
    st.write("Er zijn geen sterke correlaties (> |0.1|) gevonden tussen de factoren en werknemersattritie.")

# **Heatmap van correlaties**
st.subheader('Correlatiematrix van Factoren die Werknemersattritie Beïnvloeden')
fig, ax = plt.subplots(figsize=(10, 6))

# Heatmap tekenen
sns.heatmap(combined_dataset[['Attrition', 'JobSatisfaction', 'WorkLifeBalance', 
                           'Age', 'YearsSinceLastPromotion', 'DistanceFromHome (KM)', 'Salary','PromotionFrequency']].corr(), 
            annot=True, cmap='Blues', ax=ax)

# Heatmap in Streamlit tonen
st.pyplot(fig) 

#%%

st.subheader('Aantal werknemers die het bedrijf hebben verlaten vs. gebleven')

OverTime_yes = st.checkbox("OverTime: Ja", key="OverTime_yes")
OverTime_no = st.checkbox("OverTime: Nee", key="OverTime_no", value=not OverTime_yes)

# Zorg ervoor dat als de ene checkbox wordt geselecteerd, de andere wordt gedeselecteerd
if OverTime_yes:
    OverTime_no = False
elif OverTime_no:
    OverTime_yes = False


# Toepassen van filtering op basis van de checkboxes
if OverTime_yes and not OverTime_no:
    filtered_dataset = combined_dataset[combined_dataset['OverTime'] == "Yes"]
elif OverTime_no and not OverTime_yes:
    filtered_dataset = combined_dataset[combined_dataset['OverTime'] == "No"]
else:
    filtered_dataset = combined_dataset

# Visualisatie van attritie
fig, ax = plt.subplots()
sns.countplot(x='Attrition', hue='Attrition', data=filtered_dataset, ax=ax, palette={1: 'red', 0: 'green'})
plt.xticks(ticks=[0, 1], labels=['Gebleven', 'Verlaten'], rotation=0)
for p in ax.patches:
    ax.text(p.get_x() + p.get_width() / 2, p.get_height() + 0.1, int(p.get_height()), ha='center')

# Toon de plot
st.pyplot(fig)

#%%

# Visualiseer de relatie tussen YearsSinceLastPromotion en attritie
st.subheader('Effect van YearsSinceLastPromotion op werknemersattritie')
median_values = combined_dataset.groupby('Attrition')['YearsSinceLastPromotion'].median()

# Maak een figuur en as-object
fig, ax = plt.subplots()

# Boxplot tekenen
sns.boxplot(x='Attrition', y='YearsSinceLastPromotion', data=combined_dataset, ax=ax, 
            hue='Attrition', palette={0: 'green', 1: 'red'}, dodge=False, legend=False)

plt.xticks(ticks=[0, 1], labels=['Gebleven', 'Verlaten'], rotation=0)

# Voeg de medianen toe aan de grafiek
for i, median in enumerate(median_values):
    ax.annotate(f'Mediaan: {median}', 
                xy=(i, median),  # xy is de positie op de x- en y-as
                xytext=(i, median + 1.5),  # Verplaats de tekst hoger (pas dit aan voor meer of minder ruimte)
                ha='center', va='bottom',  # ha is horizontal alignment, va is vertical alignment
                color='black', fontsize=10, weight='bold',
                arrowprops=dict(facecolor='black', shrink=0.05))  # Optionele pijl toevoegen

st.pyplot(fig) 

# Conclusie afhankelijk van de mediaanwaarden
if median_values[0] < median_values[1]:
    conclusion = f"De mediane jaren sinds de laatste promotie voor werknemers die zijn vertrokken is **{median_values[1]}** " \
                 f"terwijl de mediane jaren voor werknemers die gebleven zijn **{median_values[0]}** is. Dit suggereert " \
                 "dat werknemers die langer geen promotie hebben gehad, een grotere kans hebben om het bedrijf te verlaten."
else:
    conclusion = f"De mediane jaren sinds de laatste promotie voor werknemers die zijn vertrokken is **{median_values[1]}** " \
                 f"terwijl de mediane jaren voor werknemers die gebleven zijn **{median_values[0]}** is. Dit suggereert " \
                 "dat werknemers die een recente promotie hebben gehad, meer geneigd zijn om bij het bedrijf te blijven."

st.write(conclusion)

#%%

# Visualiseer de relatie tussen leeftijd en attritie van werknemers

# Leeftijd categoriseren
bins = [18, 30, 40, 50]  # Leeftijdsgrenzen
labels = ['18-29', '30-39', '40-49']  # Labels voor de groepen
combined_dataset['Leeftijdsgroep'] = pd.cut(combined_dataset['Age'], bins=bins, labels=labels, right=False)

# Aantal werknemers per leeftijdsgroep en attritiestatus
age_attrition = combined_dataset.groupby(['Leeftijdsgroep', 'Attrition'], observed=False).size().unstack(fill_value=0)

# Bereken het percentage vertrokken werknemers per leeftijdsgroep
age_attrition['Percentage_Verlaten'] = age_attrition[1] / (age_attrition[1] + age_attrition[0]) * 100

st.subheader('Effect van Leeftijd op werknemersattritie')

fig, ax1 = plt.subplots(figsize=(10, 6))

# Staafgrafiek: aantal werknemers per leeftijdsgroep en attritiestatus
age_attrition[[1, 0]].plot(kind='bar', stacked=False, ax=ax1, color=['red', 'green'], alpha=0.7)

# Secundaire y-as voor het percentage vertrokken werknemers
ax2 = ax1.twinx()
ax2.plot(age_attrition.index, age_attrition['Percentage_Verlaten'], color='blue', marker='o', linestyle='-', linewidth=2)
ax2.set_ylabel('Percentage Vertrokken Werknemers', color='blue')

# Grafiek labels
ax1.set_xlabel('Leeftijdsgroep')
ax1.set_ylabel('Aantal Werknemers')
ax1.set_title('Aantal Werknemers per Leeftijdsgroep en Attritie')
ax1.legend(['Verlaten', 'Gebleven'], title='Attritie')
ax1.grid(axis='y')

# Toon de grafiek in Streamlit
st.pyplot(fig)
max_percentage = age_attrition['Percentage_Verlaten'].max()
max_index = age_attrition['Percentage_Verlaten'].idxmax()  # Geeft de index (leeftijdsgroep) met het hoogste percentage

conclusion = f"De leeftijdsgroep **{max_index}** heeft het hoogste percentage vertrokken werknemers, " \
             f"wat kan wijzen op een verhoogde kans op attritie in deze groep."

st.write(conclusion)

#%%

# Zet voorbeeld dataset om in een DataFrame
combined_dataset = pd.DataFrame(combined_dataset)

# 1. Salary Slider voor filtering van data
st.write("### Selecteer Salarisbereik")
min_salary, max_salary = st.slider("Kies een salarisbereik", 
                                   int(combined_dataset['Salary'].min()), 
                                   int(combined_dataset['Salary'].max()), 
                                   (int(combined_dataset['Salary'].min()), int(combined_dataset['Salary'].max())))

# Filter data op basis van salaris
filtered_dataset = combined_dataset[(combined_dataset['Salary'] >= min_salary) & 
                                    (combined_dataset['Salary'] <= max_salary)]

# 2. Dropdown voor filtering van Department
st.write("### Selecteer Afdeling")
departments = combined_dataset['Department'].unique()
selected_department = st.selectbox("Selecteer Afdeling", departments)

# Filter data op basis van zowel salaris als afdeling
filtered_data = filtered_dataset[filtered_dataset['Department'] == selected_department]

# Toon gefilterde data in Streamlit
st.write(f"Gefilterde data op basis van Salaris: {min_salary} - {max_salary} en Afdeling: {selected_department}")
st.write(filtered_data) 

# Visualiseer de relatie tussen salaris en attritie
st.subheader('Effect van Salaris op werknemersattritie')
plt.figure(figsize=(10, 6))
sns.scatterplot(data=combined_dataset, x='Salary', y='Attrition', hue='Attrition', palette={0: 'green', 1: 'red'}, s=100)
# Grafiek labels
plt.title('Salaris versus Werknemersattritie')
plt.xlabel('Salaris')
plt.ylabel('Attritie')
plt.yticks([0, 1], ['Gebleven', 'Verlaten'])
plt.grid()
st.pyplot(plt)

mean_salary_stayed = combined_dataset[combined_dataset['Attrition'] == 0]['Salary'].mean()
mean_salary_left = combined_dataset[combined_dataset['Attrition'] == 1]['Salary'].mean()

if mean_salary_stayed > mean_salary_left:
    conclusion = f"Werknemers die zijn gebleven hebben een gemiddeld salaris van **€{mean_salary_stayed:.2f}**, terwijl " \
                 f"werknemers die zijn vertrokken een gemiddeld salaris van **€{mean_salary_left:.2f}** hebben. Dit kan erop wijzen " \
                 "dat hogere salarissen mogelijk bijdragen aan een lagere attritie."
else:
    conclusion = f"Werknemers die zijn vertrokken hebben een gemiddeld salaris van **€{mean_salary_left:.2f}**, terwijl " \
                 f"werknemers die zijn gebleven een gemiddeld salaris van **€{mean_salary_stayed:.2f}** hebben. Dit kan erop wijzen " \
                 "dat lagere salarissen bijdragen aan een hogere attritie."

st.write(conclusion)

#%%

# Visualiseer de relatie tussen werktevredenheid en attritie
st.subheader('Effect van werktevredenheid op werknemersattritie')
median_values = combined_dataset.groupby('Attrition')['JobSatisfaction'].median()

fig, ax = plt.subplots()
sns.violinplot(x='Attrition', y='JobSatisfaction', data=combined_dataset, ax=ax, hue='Attrition', palette={0: 'green', 1: 'red'}, dodge=True)

# Labels en titel instellen
ax.set_xlabel('Werknemersattritie')
ax.set_ylabel('Werktevredenheid')
plt.xticks([0, 1], ['Gebleven', 'Verlaten'])
# Voeg de medianen toe aan de grafiek
for i, median in enumerate(median_values):
    ax.annotate(f'Mediaan: {median}', xy=(i, median), xytext=(i, median + 0.5), 
                ha='center', color='black', fontsize=10, weight='bold')

st.pyplot(fig)

mean_job_satisfaction_stayed = combined_dataset[combined_dataset['Attrition'] == 0]['JobSatisfaction'].mean()
mean_job_satisfaction_left = combined_dataset[combined_dataset['Attrition'] == 1]['JobSatisfaction'].mean()

if mean_job_satisfaction_stayed > mean_job_satisfaction_left:
    conclusion = f"Werknemers die zijn gebleven hebben een gemiddelde werktevredenheid van **{mean_job_satisfaction_stayed:.2f}**, terwijl " \
                 f"werknemers die zijn vertrokken een gemiddelde werktevredenheid van **{mean_job_satisfaction_left:.2f}** hebben. Dit suggereert dat hogere werktevredenheid " \
                 "kan bijdragen aan een lagere attritie."
else:
    conclusion = f"Werknemers die zijn vertrokken hebben een gemiddelde werktevredenheid van **{mean_job_satisfaction_left:.2f}**, terwijl " \
                 f"werknemers die zijn gebleven een gemiddelde werktevredenheid van **{mean_job_satisfaction_stayed:.2f}** hebben. Dit kan erop wijzen dat lagere werktevredenheid " \
                 "bijdraagt aan een hogere attritie."

st.write(conclusion)

#%%

# Visualiseer de relatie tussen DistanceFromHome (KM) en attritie van werknemers
st.subheader('Effect van DistanceFromHome (KM) op werknemersattritie')
median_values = combined_dataset.groupby('Attrition')['DistanceFromHome (KM)'].median()

fig, ax = plt.subplots()
sns.boxplot(x='Attrition', y='DistanceFromHome (KM)', data=combined_dataset, ax=ax, 
            hue='Attrition', palette={0: 'green', 1: 'red'}, dodge=True)
plt.xticks(ticks=[0, 1], labels=['Gebleven', 'Verlaten'], rotation=0)
# Voeg de medianen toe aan de grafiek
for i, median in enumerate(median_values):
    ax.annotate(f'Mediaan: {median}', xy=(i, median), xytext=(i, median + 0.5), 
                ha='center', color='black', fontsize=10, weight='bold')

st.pyplot(fig)

#%%

st.subheader('Effect van Werk-privébalans op werknemersattritie')
# Slider voor filtering van DistanceFromHome
min_distance, max_distance = st.slider(
    'Selecteer Afstand van huis (in km)',
    min_value=int(combined_dataset['DistanceFromHome (KM)'].min()),
    max_value=int(combined_dataset['DistanceFromHome (KM)'].max()),
    value=(int(combined_dataset['DistanceFromHome (KM)'].min()), int(combined_dataset['DistanceFromHome (KM)'].max()))
)

# Filter de data op basis van geselecteerde afstand
filtered_data = combined_dataset[
    (combined_dataset['DistanceFromHome (KM)'] >= min_distance) &
    (combined_dataset['DistanceFromHome (KM)'] <= max_distance)
]

# Dropdown voor filtering van MaritalStatus
marital_status = combined_dataset['MaritalStatus'].unique()
selected_status = st.selectbox("Selecteer burgerlijke staat", marital_status)

# Filter de data op basis van burgerlijke staat
filtered_data = filtered_data[filtered_data['MaritalStatus'] == selected_status]
st.write(f"Gefilterde data op basis van Afstand: {min_distance} - {max_distance} km en Burgerlijke staat: {selected_status}")

# Visualiseer de relatie tussen werk-privébalans en attritie
fig, ax = plt.subplots()
sns.boxplot(x='Attrition', y='WorkLifeBalance', data=filtered_data, ax=ax,
            hue='Attrition', palette={0: 'green', 1: 'red'}, dodge=True)

plt.xticks(ticks=[0, 1], labels=['Gebleven', 'Verlaten'], rotation=0)

st.pyplot(fig)
mean_work_life_balance_stayed = filtered_data[filtered_data['Attrition'] == 0]['WorkLifeBalance'].mean()
mean_work_life_balance_left = filtered_data[filtered_data['Attrition'] == 1]['WorkLifeBalance'].mean()

if mean_work_life_balance_stayed > mean_work_life_balance_left:
    conclusion = f"Werknemers die zijn gebleven hebben een gemiddelde werk-privébalans van **{mean_work_life_balance_stayed:.2f}**, terwijl " \
                 f"werknemers die zijn vertrokken een gemiddelde werk-privébalans van **{mean_work_life_balance_left:.2f}** hebben. Dit suggereert dat een betere werk-privébalans " \
                 "kan bijdragen aan een lagere attritie."
else:
    conclusion = f"Werknemers die zijn vertrokken hebben een gemiddelde werk-privébalans van **{mean_work_life_balance_left:.2f}**, terwijl " \
                 f"werknemers die zijn gebleven een gemiddelde werk-privébalans van **{mean_work_life_balance_stayed:.2f}** hebben. Dit kan erop wijzen dat een slechtere werk-privébalans " \
                 "bijdraagt aan een hogere attritie."


#%%

st.subheader('Effect van PromotionFrequency op werknemersattritie')

# Maak de checkboxes met standaardwaarde 'Alle Afdelingen' aangevinkt
alle_afdelingen = st.checkbox("Alle Afdelingen", value=True, key="alle_afdelingen")
sales = st.checkbox("Sales", value=False, key="sales")
Technology = st.checkbox("Technology", value=False, key="Technology")
hr = st.checkbox("Human Resources", value=False, key="hr")

# Zorg ervoor dat slechts één checkbox tegelijkertijd kan worden geselecteerd
if alle_afdelingen:
    sales, Technology, hr = False, False, False
elif sales:
    alle_afdelingen, Technology, hr = False, False, False
elif Technology:
    alle_afdelingen, sales, hr = False, False, False
elif hr:
    alle_afdelingen, sales, Technology = False, False, False

# Filter de dataset op basis van de geselecteerde checkbox
if alle_afdelingen:
    filtered_data = combined_dataset  # Geen filtering toepassen
elif sales:
    filtered_data = combined_dataset[combined_dataset['Department'] == 'Sales']
elif Technology:
    filtered_data = combined_dataset[combined_dataset['Department'] == 'Technology']
elif hr:
    filtered_data = combined_dataset[combined_dataset['Department'] == 'Human Resources']

# Controleer of er gefilterde data beschikbaar is
if filtered_data.empty:
    st.write("Geen data beschikbaar voor de geselecteerde afdeling.")
else:
    # Maak de barplot voor de geselecteerde afdeling of alle afdelingen
    st.write(f"Promotie frequentie binnen {'alle afdelingen' if alle_afdelingen else 'de geselecteerde afdeling'}")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='PromotionFrequency', hue='Attrition', data=filtered_data, ax=ax,
                 palette={0: 'green', 1: 'red'})

    # Labels en titels voor de grafiek
    ax.set_title(f'Promotie Frequentie versus Werknemersattritie in {"alle afdelingen" if alle_afdelingen else "de geselecteerde afdeling"}')
    ax.set_xlabel('Promotion Frequency (Aantal promoties per werknemer)')
    ax.set_ylabel('Aantal werknemers')
    ax.legend(title="Attrition", labels=["Gebleven", "Vertrokken"])
    ax.grid(True)

    # Bereken de mediaanwaarden
    median_values = filtered_data.groupby('Attrition')['PromotionFrequency'].median()

    # Voeg de mediaanwaarden toe aan de grafiek
    for i, median in enumerate(median_values):
        ax.annotate(f'Mediaan: {median}', 
                    xy=(i, median),  # xy is de positie op de x- en y-as
                    xytext=(i, median + 1),  # Verplaats de tekst hoger of lager
                    ha='center', va='bottom',  # ha is horizontal alignment, va is vertical alignment
                    color='black', fontsize=10, weight='bold',
                    arrowprops=dict(facecolor='black', shrink=0.05))  # Optionele pijl toevoegen

    # Toon de grafiek in Streamlit
    st.pyplot(fig)

    # Gemiddelde promotiefrequentie berekenen
    mean_promotion_frequency_stayed = filtered_data[filtered_data['Attrition'] == 0]['PromotionFrequency'].mean()
    mean_promotion_frequency_left = filtered_data[filtered_data['Attrition'] == 1]['PromotionFrequency'].mean()

    if mean_promotion_frequency_stayed > mean_promotion_frequency_left:
        conclusion = f"Werknemers die zijn gebleven hebben een gemiddelde promotiefrequentie van **{mean_promotion_frequency_stayed:.2f}**, terwijl " \
                     f"werknemers die zijn vertrokken een gemiddelde promotiefrequentie van **{mean_promotion_frequency_left:.2f}** hebben. Dit suggereert dat een hogere promotiefrequentie " \
                     "kan bijdragen aan een lagere attritie."
    else:
        conclusion = f"Werknemers die zijn vertrokken hebben een gemiddelde promotiefrequentie van **{mean_promotion_frequency_left:.2f}**, terwijl " \
                     f"werknemers die zijn gebleven een gemiddelde promotiefrequentie van **{mean_promotion_frequency_stayed:.2f}** hebben. Dit kan erop wijzen dat lagere promotiefrequentie " \
                     "bijdraagt aan een hogere attritie."

    st.write(conclusion)


#%%

# Functie om de kans op promotie te berekenen
def calculate_promotion_chance(
    YearsAtCompany,
    YearsInMostRecentRole,
    YearsSinceLastPromotion,
    TrainingOpportunitiesWithinYear,
    SelfRating,
    ManagerRating,
):
    # Gewichten toekennen
    w1 = 0.2  # Jaren bij het bedrijf
    w2 = 0.3  # Jaren in de huidige rol
    w3 = -0.1  # Jaren sinds de laatste promotie (negatieve impact)
    w4 = 0.2  # Training kansen
    w5 = 0.1  # Zelfbeoordeling
    w6 = 0.1  # Beoordeling door manager

    # Bereken kans op promotie
    chance = (
        w1 * YearsAtCompany
        + w2 * YearsInMostRecentRole
        + w3 * YearsSinceLastPromotion
        + w4 * TrainingOpportunitiesWithinYear
        + w5 * SelfRating
        + w6 * ManagerRating
    ) * 100  # Omzetten naar percentage

    return max(
        0.0, min(100.0, chance)
    )  # Zorg ervoor dat de kans tussen 0.0 en 100.0 blijft


# Streamlit interface
st.title("Kans op Promotie Calculator")
st.subheader("Bereken je kans op promotie op basis van je invoer.")

# Form voor invoer
with st.form(key="promotion_form"):
    # Plaats invoervelden in rijen van twee met behulp van columns
    col1, col2 = st.columns(2)
    with col1:
        YearsAtCompany = st.number_input(
            "Jaren bij het bedrijf", min_value=0, max_value=30, value=5
        )
    with col2:
        YearsInMostRecentRole = st.number_input(
            "Jaren in de huidige rol", min_value=0, max_value=30, value=2
        )

    col3, col4 = st.columns(2)
    with col3:
        YearsSinceLastPromotion = st.number_input(
            "Jaren sinds de laatste promotie",
            min_value=0,
            max_value=30,
            value=1,
        )
    with col4:
        TrainingOpportunitiesWithinYear = st.number_input(
            "Training kansen dit jaar", min_value=0, max_value=5, value=2
        )

    col5, col6 = st.columns(2)
    with col5:
        SelfRating = st.number_input(
            "Zelfbeoordeling (1-10)", min_value=1, max_value=10, value=5
        )
    with col6:
        ManagerRating = st.number_input(
            "Beoordeling door manager (1-10)",
            min_value=1,
            max_value=10,
            value=5,
        )

    # Knop voor berekening
    submit_button = st.form_submit_button("Bereken Kans op Promotie")

# Bereken de kans alleen als de knop is ingedrukt
if submit_button:
    promotion_chance = calculate_promotion_chance(
        YearsAtCompany,
        YearsInMostRecentRole,
        YearsSinceLastPromotion,
        TrainingOpportunitiesWithinYear,
        SelfRating,
        ManagerRating,
    )

    # Toon het resultaat
    st.write(f"Je kans op promotie is: {promotion_chance:.2f}%")

# Knop om resetten
if st.button("Reset Invoer"):
    st.experimental_rerun()

#%%

import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Voorbeeld dataset laden (vervang dit met jouw eigen dataset)
# combined_dataset = pd.read_csv('jouw_dataset.csv')

# Titel en beschrijving van de voorspelling
st.title("Voorspelling van werknemersattritie")
st.write("""
In deze sectie gebruiken we een Random Forest Classifier om te voorspellen of een werknemer het bedrijf zal verlaten 
(werknemersattritie) op basis van verschillende factoren zoals leeftijd, werktevredenheid, balans tussen werk en privé, enzovoort.
""")

# Select relevant features for prediction
features = ['JobSatisfaction', 'WorkLifeBalance', 'Age', 'YearsAtCompany', 
            'YearsSinceLastPromotion', 'DistanceFromHome (KM)', 'Salary', 'PromotionFrequency']
X = combined_dataset[features]
y = combined_dataset['Attrition']

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and evaluate a Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Random Forest metrics
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_report = classification_report(y_test, y_pred_rf, output_dict=True)

# Weergeven van de accuracy score
st.subheader("Modelresultaten")
st.write(f"De nauwkeurigheid van het Random Forest-model is: **{rf_accuracy:.2f}**")

# Weergeven van het classification report
st.write("Hieronder vind je het classificatierapport, dat de prestaties van het model per categorie (verbleven of vertrokken) weergeeft:")

# Converteer het classification report naar een DataFrame voor overzichtelijke weergave
rf_report_df = pd.DataFrame(rf_report).transpose()
st.dataframe(rf_report_df)

# Optioneel: Extra uitleg van het classificatierapport
st.write("""
Het classificatierapport toont de prestaties van het model in termen van precision, recall en F1-score voor beide klassen (werknemers die blijven en werknemers die vertrekken). 
Een hogere F1-score betekent dat het model beter presteert in het voorspellen van die klasse.
""")












