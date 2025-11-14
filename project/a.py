import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ---- Load and normalize ----
df = pd.read_csv('hhs_breach_data.csv')

df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(' ', '_')
)

print("Columns:", df.columns.tolist())

# ---- Date handling ----
df['breach_submission_date'] = pd.to_datetime(df['breach_submission_date'], errors='coerce')

# If you already have a YearMonth column in string form, you can skip this.
# Create a yearmonth Period then convert to a timestamp (first day of month) for plotting.
df['yearmonth'] = df['breach_submission_date'].dt.to_period('M')
# Convert Period to Timestamp (Seaborn/Matplotlib prefer datetime objects)
df['yearmonth_ts'] = df['yearmonth'].dt.to_timestamp()

# ---- Clean individuals_affected robustly ----
# Inspect a few unique problem-looking values (optional debug)
# print(df['individuals_affected'].unique()[:50])

# Convert to string, remove common non-digit characters, extract first numeric group
def parse_number(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    # remove commas and whitespace
    s = s.replace(',', '').replace(' ', '')
    # sometimes there are ">1000", "1000+", "approx. 1200", etc.
    # extract the first continuous digit group
    m = re.search(r'(\d{1,})', s)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return np.nan
    return np.nan

df['individuals_affected_clean'] = df['individuals_affected'].apply(parse_number)

# Drop rows that couldn't be parsed
n_before = len(df)
df = df.dropna(subset=['individuals_affected_clean', 'breach_submission_date'])
n_after = len(df)
print(f"Dropped {n_before - n_after} rows with invalid individuals_affected or date")

# Filter out non-positive if desired
df = df[df['individuals_affected_clean'] > 0]

# Convert to numeric type explicitly
df['individuals_affected_clean'] = df['individuals_affected_clean'].astype(float)

# Optionally create log column for plotting scale
df['individuals_affected_log'] = np.log10(df['individuals_affected_clean'] + 1)

# ---- Agg for monthly plotting ----
monthly = (
    df.groupby(['yearmonth_ts', 'state', 'type_of_breach'])['individuals_affected_clean']
      .sum()
      .reset_index()
)

# Ensure types are friendly for plotting
monthly['state'] = monthly['state'].astype(str)
monthly['type_of_breach'] = monthly['type_of_breach'].astype(str)

# ---- Visualization 1: Violin (distribution) ----
plt.figure(figsize=(12, 6))
sns.violinplot(
    data=df,
    x='type_of_breach',
    y='individuals_affected_clean',
    hue='covered_entity_type',
    split=True,
    scale='width',
    cut=0
)
plt.yscale('log')
plt.title('Distribution of Individuals Affected by Breach Type and Entity Type')
plt.xlabel('Type of Breach')
plt.ylabel('Individuals Affected (log scale)')
plt.legend(title='Entity Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ---- Visualization 2: FacetGrid bubble chart ----
sns.set(style='whitegrid')

# If too many states, limit to top N by count for readability (adjust as needed)
top_states = (monthly.groupby('state')['individuals_affected_clean']
                        .sum()
                        .sort_values(ascending=False)
                        .head(12)
                        .index.tolist())
monthly_small = monthly[monthly['state'].isin(top_states)]

g = sns.FacetGrid(monthly_small, col='state', col_wrap=4, height=3, sharey=False)
g.map_dataframe(
    sns.scatterplot,
    x='yearmonth_ts',
    y='individuals_affected_clean',
    hue='type_of_breach',
    size='individuals_affected_clean',
    sizes=(20, 200),
    alpha=0.7,
    legend=False  # add custom legend outside
)

# Add a legend for hue and sizes manually
for ax in g.axes.flatten():
    ax.tick_params(axis='x', rotation=45)
g.set_axis_labels('Month', 'Individuals Affected')
g.set_titles(col_template='{col_name}')

# Create a separate legend for breach types
handles, labels = g.axes.flatten()[0].get_legend_handles_labels()
# The above may not always work consistently; create a color legend using seaborn color palette
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Monthly Breaches by State and Breach Type\n(Point size = Individuals Affected)', fontsize=14)
plt.show()
