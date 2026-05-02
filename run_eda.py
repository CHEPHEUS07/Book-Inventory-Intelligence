"""
📊 CodeAlpha Internship — Task 2: Exploratory Data Analysis
Dataset: scrapped_books.csv — scraped from books.toscrape.com
Tools: pandas, seaborn, matplotlib
This script runs all EDA cells and saves plots + cleaned CSV.
Uses non-interactive matplotlib backend so it runs without a GUI.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['figure.figsize'] = (10, 5)
print('✅ Libraries loaded!')

# ===========================================
# 2. Load the Dataset
# ===========================================
df = pd.read_csv('scrapped_books.csv')

# Clean column names: lowercase, strip spaces, replace special chars
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(' ', '_')
    .str.replace('(', '', regex=False)
    .str.replace(')', '', regex=False)
)

print('Columns:', df.columns.tolist())
print(f'Shape  : {df.shape[0]} rows x {df.shape[1]} columns')
print(df.head())

# ===========================================
# 3. Data Overview
# ===========================================
print('\n=== DATA TYPES ===')
print(df.dtypes)
print('\n=== INFO ===')
df.info()
print('\n=== STATISTICAL SUMMARY ===')
print(df.describe(include='all').T)

# ===========================================
# 4. Data Cleaning
# ===========================================
# Missing values
missing = df.isnull().sum()
print('\n=== MISSING VALUES ===')
print(missing[missing > 0] if missing.any() else 'No missing values ✅')

# Duplicates
dupes = df.duplicated().sum()
print(f'\nDuplicate rows: {dupes}')
if dupes > 0:
    df.drop_duplicates(inplace=True)
    print(f'Dropped {dupes} duplicates.')

# Fix numeric columns — strip £ and  symbol if present
for col in ['price', 'price_excl._tax', 'price_incl._tax', 'tax']:
    if col in df.columns:
        df[col] = (
            df[col].astype(str)
            .str.replace('£', '', regex=False)
            .str.replace('Â', '', regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Rename columns for convenience
rename_map = {}
if 'price_excl._tax' in df.columns:
    rename_map['price_excl._tax'] = 'price_ex_tax'
if 'price_incl._tax' in df.columns:
    rename_map['price_incl._tax'] = 'price_in_tax'
if 'rating_out_of_5' in df.columns:
    rename_map['rating_out_of_5'] = 'rating'
if 'number_of_reviews' in df.columns:
    rename_map['number_of_reviews'] = 'no_of_reviews'
df.rename(columns=rename_map, inplace=True)

# Rating — ensure numeric
df['rating'] = pd.to_numeric(
    df['rating'].astype(str).str.extract(r'(\d+\.?\d*)')[0],
    errors='coerce'
)

# no_of_reviews → integer
if 'no_of_reviews' in df.columns:
    df['no_of_reviews'] = pd.to_numeric(df['no_of_reviews'], errors='coerce').fillna(0).astype(int)
 
# Derived columns
df['description_length'] = df['description'].fillna('').apply(len)
df['in_stock'] = df['availability'].astype(str).str.lower().str.contains('in stock')
df['price_band'] = pd.cut(
    df['price'],
    bins=[0, 10, 20, 30, 60],
    labels=['Budget (<£10)', 'Mid (£10-20)', 'Premium (£20-30)', 'Luxury (£30+)']
)

print('✅ Cleaning done. Shape:', df.shape)
print(df[['title', 'rating', 'price', 'category', 'in_stock', 'no_of_reviews']].head(8))

# ===========================================
# 5. Univariate Analysis
# ===========================================

# 5.1 Rating Distribution
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
rating_counts = df['rating'].value_counts().sort_index()

axes[0].bar(rating_counts.index.astype(str), rating_counts.values,
            color=sns.color_palette('Blues_d', len(rating_counts)), edgecolor='white')
axes[0].set_title('Distribution of Star Ratings', fontweight='bold')
axes[0].set_xlabel('Star Rating (out of 5)')
axes[0].set_ylabel('Number of Books')
for x, v in zip(rating_counts.index, rating_counts.values):
    axes[0].text(str(x), v + 0.5, str(v), ha='center', fontweight='bold')

axes[1].pie(rating_counts.values,
            labels=[f'{int(i)} Star' for i in rating_counts.index],
            autopct='%1.1f%%',
            colors=sns.color_palette('Blues', len(rating_counts)),
            startangle=140,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
axes[1].set_title('Rating Share (%)', fontweight='bold')

plt.suptitle('Star Rating Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot1_rating_distribution.png', bbox_inches='tight')
plt.close()
print('💡 Most common rating:', int(rating_counts.idxmax()), 'stars —', rating_counts.max(), 'books')
print('✅ Saved plot1_rating_distribution.png')

# 5.2 Price Distribution
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sns.histplot(df['price'].dropna(), bins=20, kde=True,
             color='steelblue', ax=axes[0], edgecolor='white')
axes[0].axvline(df['price'].mean(),   color='crimson', linestyle='--',
                label=f"Mean: £{df['price'].mean():.2f}")
axes[0].axvline(df['price'].median(), color='orange',  linestyle='--',
                label=f"Median: £{df['price'].median():.2f}")
axes[0].legend()
axes[0].set_title('Price Distribution', fontweight='bold')
axes[0].set_xlabel('Price (£)')

band_counts = df['price_band'].value_counts()
axes[1].bar(band_counts.index.astype(str), band_counts.values,
            color=sns.color_palette('Greens_d', len(band_counts)), edgecolor='white')
axes[1].set_title('Books per Price Band', fontweight='bold')
axes[1].set_xlabel('Price Band')
axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=15)

plt.suptitle('Price Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot2_price_distribution.png', bbox_inches='tight')
plt.close()
print(f'💡 Price range: £{df["price"].min():.2f} – £{df["price"].max():.2f}')
print('✅ Saved plot2_price_distribution.png')

# 5.3 Top Categories
top_cats = df['category'].value_counts().head(15)

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(top_cats.index[::-1], top_cats.values[::-1],
               color=sns.color_palette('viridis', len(top_cats)))
ax.set_title('Top 15 Categories by Book Count', fontweight='bold', fontsize=13)
ax.set_xlabel('Number of Books')
for bar, val in zip(bars, top_cats.values[::-1]):
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
            str(val), va='center', fontsize=9)
plt.tight_layout()
plt.savefig('plot3_top_categories.png', bbox_inches='tight')
plt.close()
print('💡 Most common category:', top_cats.index[0], '—', top_cats.iloc[0], 'books')
print('✅ Saved plot3_top_categories.png')

# 5.4 Stock Availability
stock_counts = df['in_stock'].value_counts()
fig, ax = plt.subplots(figsize=(6, 5))
ax.pie(stock_counts.values,
       labels=['In Stock', 'Out of Stock'] if len(stock_counts) > 1 else ['In Stock'],
       autopct='%1.1f%%',
       colors=['#4CAF50', '#F44336'][:len(stock_counts)],
       wedgeprops={'edgecolor': 'white', 'linewidth': 2},
       startangle=90)
ax.set_title('Stock Availability', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('plot4_availability.png', bbox_inches='tight')
plt.close()
pct = df['in_stock'].mean() * 100
print(f'💡 {pct:.1f}% of books are in stock.')
print('✅ Saved plot4_availability.png')

# ===========================================
# 6. Bivariate Analysis
# ===========================================

# 6.1 Rating vs Price
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(data=df, x='rating', y='price', palette='coolwarm', ax=axes[0])
axes[0].set_title('Price by Star Rating', fontweight='bold')
axes[0].set_xlabel('Star Rating')
axes[0].set_ylabel('Price (£)')

sns.regplot(data=df, x='rating', y='price',
            scatter_kws={'alpha': 0.4, 'color': 'steelblue'},
            line_kws={'color': 'crimson'}, ax=axes[1])
axes[1].set_title('Rating vs Price (Trend Line)', fontweight='bold')
axes[1].set_xlabel('Star Rating')
axes[1].set_ylabel('Price (£)')

corr = df[['rating', 'price']].corr().iloc[0, 1]
plt.suptitle('Rating vs Price Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot5_rating_vs_price.png', bbox_inches='tight')
plt.close()
print(f'💡 Pearson correlation (rating vs price): {corr:.3f}')
print('✅ Saved plot5_rating_vs_price.png')

# 6.2 Average Rating by Category
cat_rating = (
    df.groupby('category')['rating']
    .agg(['mean', 'count'])
    .query('count >= 2')
    .sort_values('mean', ascending=False)
    .head(15)
    .reset_index()
)

if len(cat_rating) > 0:
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(cat_rating['category'][::-1], cat_rating['mean'][::-1],
                   color=sns.color_palette('RdYlGn', len(cat_rating)))
    ax.axvline(df['rating'].mean(), color='navy', linestyle='--', alpha=0.7,
               label=f'Overall avg: {df["rating"].mean():.2f}')
    ax.set_xlim(0, 5.8)
    ax.set_title('Average Rating by Category (min 2 books)', fontweight='bold')
    ax.set_xlabel('Average Star Rating')
    ax.legend()
    for bar, val in zip(bars, cat_rating['mean'][::-1]):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig('plot6_category_avg_rating.png', bbox_inches='tight')
    plt.close()
    print('💡 Top rated category:', cat_rating.iloc[0]['category'],
          '—', round(cat_rating.iloc[0]['mean'], 2), 'stars')
    print('✅ Saved plot6_category_avg_rating.png')

# 6.3 Price (ex tax) vs Price (inc tax) coloured by Rating
if 'price_ex_tax' in df.columns and 'price_in_tax' in df.columns:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(data=df, x='price_ex_tax', y='price_in_tax',
                    hue='rating', palette='viridis', alpha=0.7, s=60, ax=ax)
    ax.set_title('Price ex Tax vs Price inc Tax — coloured by Rating', fontweight='bold')
    ax.set_xlabel('Price ex Tax (£)')
    ax.set_ylabel('Price inc Tax (£)')
    ax.legend(title='Rating', bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('plot7_tax_scatter.png', bbox_inches='tight')
    plt.close()
    print('✅ Saved plot7_tax_scatter.png')

# 6.4 Number of Reviews vs Rating
if 'no_of_reviews' in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    avg_reviews = df.groupby('rating')['no_of_reviews'].mean()
    axes[0].bar(avg_reviews.index.astype(str), avg_reviews.values,
                color=sns.color_palette('Purples_d', len(avg_reviews)), edgecolor='white')
    axes[0].set_title('Avg Number of Reviews per Rating', fontweight='bold')
    axes[0].set_xlabel('Star Rating')
    axes[0].set_ylabel('Avg Reviews')

    sns.stripplot(data=df, x='rating', y='no_of_reviews',
                  palette='Set2', alpha=0.5, jitter=True, ax=axes[1])
    axes[1].set_title('Reviews Distribution by Rating', fontweight='bold')
    axes[1].set_xlabel('Star Rating')
    axes[1].set_ylabel('Number of Reviews')

    plt.suptitle('Reviews Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plot8_reviews_vs_rating.png', bbox_inches='tight')
    plt.close()
    print('✅ Saved plot8_reviews_vs_rating.png')

# ===========================================
# 7. Multivariate Analysis
# ===========================================

# 7.1 Category × Rating Heatmap
top10 = df['category'].value_counts().head(10).index
pivot = (
    df[df['category'].isin(top10)]
    .pivot_table(index='category', columns='rating',
                 values='title', aggfunc='count', fill_value=0)
)

fig, ax = plt.subplots(figsize=(11, 6))
sns.heatmap(pivot, annot=True, fmt='d', cmap='YlOrRd',
            linewidths=0.5, cbar_kws={'label': 'Number of Books'}, ax=ax)
ax.set_title('Category × Star Rating Heatmap (Top 10 Categories)',
             fontweight='bold', fontsize=13)
ax.set_xlabel('Star Rating')
ax.set_ylabel('Category')
plt.tight_layout()
plt.savefig('plot9_category_rating_heatmap.png', bbox_inches='tight')
plt.close()
print('✅ Saved plot9_category_rating_heatmap.png')

# 7.2 Correlation Heatmap — All Numeric Columns
num_cols = ['rating', 'price', 'price_ex_tax', 'price_in_tax',
            'tax', 'no_of_reviews', 'description_length']
num_cols = [c for c in num_cols if c in df.columns]
corr_matrix = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            mask=mask, vmin=-1, vmax=1,
            linewidths=0.8, square=True, ax=ax,
            cbar_kws={'shrink': 0.7})
ax.set_title('Correlation Matrix — All Numeric Columns', fontweight='bold')
plt.tight_layout()
plt.savefig('plot10_correlation.png', bbox_inches='tight')
plt.close()
print('✅ Saved plot10_correlation.png')

# ===========================================
# 8. Outlier Detection
# ===========================================
check_cols = [c for c in ['price', 'tax', 'description_length', 'no_of_reviews'] if c in df.columns]
fig, axes = plt.subplots(1, len(check_cols), figsize=(5 * len(check_cols), 5))
if len(check_cols) == 1:
    axes = [axes]

colors = ['steelblue', 'darkorange', 'mediumseagreen', 'mediumpurple']
for ax, col, color in zip(axes, check_cols, colors):
    sns.boxplot(y=df[col].dropna(), color=color, ax=ax, width=0.4)
    ax.set_title(f'Outliers: {col}', fontweight='bold')
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    n_out = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
    ax.text(0.05, 0.95, f'Outliers: {n_out}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', color='crimson')

plt.suptitle('Outlier Detection (IQR Method)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plot11_outliers.png', bbox_inches='tight')
plt.close()
print('✅ Saved plot11_outliers.png')

# ===========================================
# 9. Summary of Key Findings
# ===========================================
print()
print('=' * 60)
print('   📋 EDA SUMMARY — KEY FINDINGS')
print('=' * 60)
print(f'Total books analysed        : {len(df)}')
print(f'Unique categories           : {df["category"].nunique()}')
print(f'Average star rating         : {df["rating"].mean():.2f} / 5')
print(f'Most common rating          : {int(df["rating"].mode()[0])} stars')
print(f'Average price               : £{df["price"].mean():.2f}')
print(f'Price range                 : £{df["price"].min():.2f} – £{df["price"].max():.2f}')
print(f'Most popular category       : {df["category"].value_counts().index[0]}')
if len(cat_rating) > 0:
    print(f'Top rated category          : {cat_rating.iloc[0]["category"]} ({cat_rating.iloc[0]["mean"]:.2f} stars)')
print(f'Rating–Price correlation    : {df[["rating", "price"]].corr().iloc[0,1]:.3f}')
if 'no_of_reviews' in df.columns:
    print(f'Total reviews in dataset    : {df["no_of_reviews"].sum()}')
print(f'% Books in stock            : {df["in_stock"].mean()*100:.1f}%')
print('=' * 60)
print('✅ 11 plots saved as PNG files in the same folder.')
print('✅ Cleaned CSV ready for Task 3 (Tableau) & Task 4 (Sentiment).')

# ===========================================
# 10. Export Cleaned Data for Tasks 3 & 4
# ===========================================
df.to_csv('books_cleaned.csv', index=False)
print('\n✅ Saved: books_cleaned.csv')
print('   → Task 3 : import into Tableau for your dashboard')
print('   → Task 4 : run sentiment analysis on the description column')
