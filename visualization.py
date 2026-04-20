import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load dataset
df = pd.read_csv('../data/netflix_titles.csv')

# Basic cleaning
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
df['year_added'] = df['date_added'].dt.year

# Create output folder
import os
os.makedirs('../outputs', exist_ok=True)

# -------------------------------
# 1. BASIC PLOTS
# -------------------------------

# Count of Movies vs TV Shows
type_counts = df['type'].value_counts()

plt.figure()
type_counts.plot(kind='bar')
plt.title("Movies vs TV Shows")
plt.xlabel("Type")
plt.ylabel("Count")
plt.savefig('../outputs/bar_type.png')

# -------------------------------
# 2. HISTOGRAM
# -------------------------------

plt.figure()
df['release_year'].hist(bins=30)
plt.title("Distribution of Release Year")
plt.savefig('../outputs/hist_release_year.png')

# -------------------------------
# 3. PIE CHART
# -------------------------------

plt.figure()
type_counts.plot(kind='pie', autopct='%1.1f%%')
plt.title("Content Type Distribution")
plt.ylabel('')
plt.savefig('../outputs/pie_chart.png')

# -------------------------------
# 4. LINE PLOT
# -------------------------------

year_data = df['year_added'].value_counts().sort_index()

plt.figure()
plt.plot(year_data.index, year_data.values)
plt.title("Content Added Over Years")
plt.xlabel("Year")
plt.ylabel("Count")
plt.savefig('../outputs/line_plot.png')

# -------------------------------
# 5. TOP COUNTRIES
# -------------------------------

top_countries = df['country'].value_counts().head(10)

plt.figure()
top_countries.plot(kind='barh')
plt.title("Top 10 Countries")
plt.savefig('../outputs/top_countries.png')

# -------------------------------
# 6. ADVANCED - HEATMAP
# -------------------------------

# Prepare data
df['month_added'] = df['date_added'].dt.month
pivot = df.pivot_table(index='month_added', columns='year_added', aggfunc='size')

plt.figure(figsize=(12,6))
sns.heatmap(pivot, cmap='coolwarm')
plt.title("Heatmap of Content Added")
plt.savefig('../outputs/heatmap.png')

# -------------------------------
# 7. BOX PLOT
# -------------------------------

plt.figure()
sns.boxplot(x='type', y='release_year', data=df)
plt.title("Boxplot of Release Year by Type")
plt.savefig('../outputs/boxplot.png')

# -------------------------------
# 8. SCATTER PLOT
# -------------------------------

plt.figure()
plt.scatter(df['release_year'], df['year_added'], alpha=0.3)
plt.title("Release Year vs Added Year")
plt.xlabel("Release Year")
plt.ylabel("Year Added")
plt.savefig('../outputs/scatter.png')

# -------------------------------
# 9. STACKED BAR (ADVANCED)
# -------------------------------

genre_data = df['listed_in'].str.split(',', expand=True).stack().value_counts().head(10)

plt.figure()
genre_data.plot(kind='bar')
plt.title("Top Genres")
plt.savefig('../outputs/genres.png')

# Top 10 Directors
top_directors = df['director'].dropna().str.split(',', expand=True).stack().value_counts().head(10)

plt.figure(figsize=(10,6))
top_directors.plot(kind='bar')
plt.title("Top 10 Directors on Netflix")
plt.xlabel("Director")
plt.ylabel("Number of Titles")
plt.savefig('../outputs/top10directors.png')


type_year = df.groupby(['year_added', 'type']).size().unstack().fillna(0)

plt.figure(figsize=(10,6))
type_year.plot.area()
plt.title("Growth of Movies vs TV Shows Over Time")
plt.xlabel("Year")
plt.ylabel("Count")
plt.savefig('../outputs/Growth of Movies vs TV Shows Over Time.png')

# Extract numeric duration
df['duration_int'] = df['duration'].str.extract('(\d+)').astype(float)

plt.figure(figsize=(8,5))
sns.histplot(data=df, x='duration_int', hue='type', bins=30)
plt.title("Duration Distribution")
plt.savefig('../outputs/Durations Distribution.png')

monthly = df['month_added'].value_counts().sort_index()

plt.figure()
plt.plot(monthly.index, monthly.values, marker='o')
plt.title("Monthly Content Addition Trend")
plt.xlabel("Month")
plt.ylabel("Count")
plt.savefig('../outputs/Monthly Content Addition Trend.png')

from wordcloud import WordCloud

text = " ".join(df['title'].dropna())

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Title WordCloud")
plt.savefig('../outputs/Title WordCloud.png')

plt.figure()
sns.kdeplot(df['release_year'], fill=True)
plt.title("Density of Release Years")
plt.savefig('../outputs/Density of Release Years.png')

print("✅ All plots generated and saved in outputs folder!")