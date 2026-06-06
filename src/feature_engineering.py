import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── Load ──────────────────────────────────────────────────────────────────
df = pd.read_csv('data/listings_clean.csv')
print(f"Loaded: {df.shape}")

# ── Clean ─────────────────────────────────────────────────────────────────
df = df.dropna(subset=['rent_monthly', 'area_sqft', 'bhk_type', 'furnishing', 'city_zone'])
df = df[df['city_zone'] != 'Other']
df = df[df['area_sqft'] > 0]
print(f"After cleaning: {len(df)} rows")

# ── BHK ordinal ───────────────────────────────────────────────────────────
bhk_map = {'1 RK': 0, '1 BHK': 1, '2 BHK': 2, '3 BHK': 3, '4 BHK': 4, '4+ BHK': 5}
df['bhk_encoded'] = df['bhk_type'].map(bhk_map)
df = df[df['bhk_encoded'].notna()]

# ── Furnishing ordinal ────────────────────────────────────────────────────
furnish_map = {'Unfurnished': 0, 'Semi-Furnished': 1, 'Furnished': 2}
df['furnishing_encoded'] = df['furnishing'].map(furnish_map)
df = df[df['furnishing_encoded'].notna()]

# ── Log sqft ─────────────────────────────────────────────────────────────
df['log_sqft'] = np.log1p(df['area_sqft'])

# ── Locality listing count ────────────────────────────────────────────────
locality_counts = df['locality'].value_counts().rename('locality_count')
df = df.join(locality_counts, on='locality')

# ── Zone target encoding (EDA only — leak-safe encoding inside CV) ────────
zone_mean = df.groupby('city_zone')['rent_monthly'].mean()
df['zone_target_enc'] = df['city_zone'].map(zone_mean)
print(f"\nZone mean rents:\n{zone_mean.sort_values(ascending=False)}")

# ── Interaction term ──────────────────────────────────────────────────────
df['zone_x_furnishing'] = df['zone_target_enc'] * df['furnishing_encoded']

# ── Feature matrix ────────────────────────────────────────────────────────
FEATURES = [
    'bhk_encoded',
    'furnishing_encoded',
    'log_sqft',
    'locality_count',
    'zone_target_enc',
    'zone_x_furnishing',
]
TARGET = 'rent_monthly'

X = df[FEATURES].copy()
y = df[TARGET].copy()

print(f"\nFeature matrix: {X.shape}")
print(f"\nCorrelations with rent:")
print(X.corrwith(y).sort_values(ascending=False))

# ── Save ──────────────────────────────────────────────────────────────────
df_out = df[FEATURES + [TARGET, 'locality', 'city_zone']].copy()
df_out.to_csv('data/features.csv', index=False)
print(f"\nSaved → data/features.csv")
