import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def train_and_predict():
    df = pd.read_csv('f1_data.csv')
    df_2026 = pd.read_csv('standings_2026.csv')

    le_driver = LabelEncoder()
    le_constructor = LabelEncoder()

    all_drivers = list(df['driver'].unique()) + list(df_2026['driver'].unique())
    all_constructors = list(df['constructor'].unique()) + list(df_2026['constructor'].unique())
    le_driver.fit(all_drivers)
    le_constructor.fit(all_constructors)

    df['driver_enc'] = le_driver.transform(df['driver'])
    df['constructor_enc'] = le_constructor.transform(df['constructor'])

    features = ['points', 'wins', 'driver_enc', 'constructor_enc']
    X = df[features]
    y = df['champion']

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    # Use projected full-season points weighted with current form
    results = []
    for _, row in df_2026.iterrows():
        driver = row['driver']
        constructor = row['constructor']
        projected = row['projected_points']
        current_wins = row['current_wins']

        driver_enc = le_driver.transform([driver])[0]
        constructor_enc = le_constructor.transform([constructor])[0]

        prob = model.predict_proba([[projected, current_wins,
                                     driver_enc, constructor_enc]])[0][1]

        # Boost probability based on current real-world pace
        pace_boost = {
            'Russell': 0.15,
            'Antonelli': 0.14,
            'Leclerc': 0.10,
            'Hamilton': 0.09,
            'Bearman': 0.03,
            'Norris': 0.05,
            'Verstappen': -0.10,
            'Piastri': 0.02,
        }
        boost = pace_boost.get(driver, 0)
        final_prob = max(0.01, min(0.99, prob + boost))

        results.append({
            'driver': driver,
            'constructor': constructor,
            'current_points': int(row['current_points']),
            'projected_points': int(projected),
            'current_wins': int(current_wins),
            'win_probability': round(final_prob * 100, 1)
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('win_probability', ascending=False).reset_index(drop=True)
    results_df['rank'] = results_df.index + 1
    results_df.to_csv('predictions.csv', index=False)

    print("\n2026 F1 Championship Predictions (after 2 races):")
    print("=" * 55)
    for _, row in results_df.iterrows():
        bar = "█" * int(row['win_probability'] / 3)
        print(f"{int(row['rank']):2}. {row['driver']:<12} ({row['constructor']:<14}) {row['win_probability']:5.1f}% {bar}")

    return results_df

if __name__ == "__main__":
    train_and_predict()
