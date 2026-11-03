import glob, pandas as pd
for f in glob.glob('plots_current/**/wasserstein_distances.csv', recursive=True):
    print(f, pd.read_csv(f)['tag'].unique())
