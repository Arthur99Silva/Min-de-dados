import pandas as pd
import glob

# COlunas
column_mapping = {
    'Country': 'Country',
    'Country or region': 'Country',
    'Region': 'Region',
    'Happiness Rank': 'Happiness Rank',
    'Overall rank': 'Happiness Rank',
    'Happiness.Rank': 'Happiness Rank',
    'Happiness Score': 'Happiness Score',
    'Score': 'Happiness Score',
    'Happiness.Score': 'Happiness Score',
    'Economy (GDP per Capita)': 'GDP per capita',
    'Economy..GDP.per.Capita.': 'GDP per capita',
    'GDP per capita': 'GDP per capita',
    'Family': 'Social support',
    'Social support': 'Social support',
    'Health (Life Expectancy)': 'Healthy life expectancy',
    'Health..Life.Expectancy.': 'Healthy life expectancy',
    'Healthy life expectancy': 'Healthy life expectancy',
    'Freedom': 'Freedom',
    'Freedom to make life choices': 'Freedom',
    'Generosity': 'Generosity',
    'Trust (Government Corruption)': 'Perceptions of corruption',
    'Trust..Government.Corruption.': 'Perceptions of corruption',
    'Perceptions of corruption': 'Perceptions of corruption',
    'Dystopia Residual': 'Dystopia Residual',
    'Whisker.high': 'Whisker high',
    'Whisker.low': 'Whisker low'
}

def process_file(file_path):
    year = file_path.split('/')[-1].split('.')[0]
    df = pd.read_csv(file_path)

    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    df['Year'] = int(year)
    
    expected_columns = ['Country', 'Region', 'Year', 'Happiness Rank', 'Happiness Score',
                       'GDP per capita', 'Social support', 'Healthy life expectancy',
                       'Freedom', 'Generosity', 'Perceptions of corruption', 'Dystopia Residual']
    
    for col in expected_columns:
        if col not in df.columns:
            df[col] = None
    
    return df[expected_columns]

files = glob.glob('/home/arthurantunes/Min-de-dados/IA2/archive/201*.csv')

merged_df = pd.concat([process_file(f) for f in files], ignore_index=True)

country_corrections = {
    'Taiwan Province of China': 'Taiwan',
    'Hong Kong S.A.R., China': 'Hong Kong',
    'Trinidad & Tobago': 'Trinidad and Tobago'
}

merged_df['Country'] = merged_df['Country'].replace(country_corrections)

merged_df.to_csv('/home/arthurantunes/Min-de-dados/IA2/archive/world_happiness_merged.csv', index=False)

print("Dataset consolidado salvo em:")
print("/home/arthurantunes/Min-de-dados/IA2/archive/world_happiness_merged.csv")