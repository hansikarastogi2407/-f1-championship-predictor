import pandas as pd

def collect_all_data():
    data = []
    historical = {
        2010: [('Vettel','Red Bull',256,5),('Alonso','Ferrari',252,5),('Webber','Red Bull',242,4),('Hamilton','McLaren',240,3),('Button','McLaren',214,2)],
        2011: [('Vettel','Red Bull',392,11),('Button','McLaren',270,3),('Webber','Red Bull',258,1),('Alonso','Ferrari',257,1),('Hamilton','McLaren',227,3)],
        2012: [('Vettel','Red Bull',281,5),('Alonso','Ferrari',278,3),('Raikkonen','Lotus',207,1),('Hamilton','McLaren',190,4),('Button','McLaren',188,1)],
        2013: [('Vettel','Red Bull',397,13),('Alonso','Ferrari',242,2),('Webber','Red Bull',199,2),('Hamilton','Mercedes',189,1),('Raikkonen','Lotus',183,1)],
        2014: [('Hamilton','Mercedes',384,11),('Rosberg','Mercedes',317,5),('Ricciardo','Red Bull',238,3),('Alonso','Ferrari',161,0),('Bottas','Williams',186,0)],
        2015: [('Hamilton','Mercedes',381,10),('Rosberg','Mercedes',322,6),('Vettel','Ferrari',278,3),('Raikkonen','Ferrari',150,0),('Massa','Williams',121,0)],
        2016: [('Rosberg','Mercedes',385,9),('Hamilton','Mercedes',380,10),('Ricciardo','Red Bull',256,3),('Vettel','Ferrari',212,0),('Raikkonen','Ferrari',186,1)],
        2017: [('Hamilton','Mercedes',363,9),('Vettel','Ferrari',317,8),('Bottas','Mercedes',305,3),('Raikkonen','Ferrari',205,1),('Ricciardo','Red Bull',200,1)],
        2018: [('Hamilton','Mercedes',408,11),('Vettel','Ferrari',320,5),('Raikkonen','Ferrari',251,1),('Bottas','Mercedes',247,0),('Verstappen','Red Bull',249,2)],
        2019: [('Hamilton','Mercedes',413,11),('Bottas','Mercedes',326,4),('Verstappen','Red Bull',278,3),('Leclerc','Ferrari',264,2),('Vettel','Ferrari',240,1)],
        2020: [('Hamilton','Mercedes',347,11),('Bottas','Mercedes',223,2),('Verstappen','Red Bull',214,2),('Perez','Racing Point',125,1),('Leclerc','Ferrari',98,0)],
        2021: [('Verstappen','Red Bull',395,10),('Hamilton','Mercedes',387,8),('Bottas','Mercedes',226,1),('Perez','Red Bull',190,1),('Norris','McLaren',160,0)],
        2022: [('Verstappen','Red Bull',454,15),('Leclerc','Ferrari',308,3),('Perez','Red Bull',305,2),('Russell','Mercedes',275,1),('Hamilton','Mercedes',240,0)],
        2023: [('Verstappen','Red Bull',575,19),('Perez','Red Bull',285,2),('Alonso','Aston Martin',206,0),('Hamilton','Mercedes',234,0),('Sainz','Ferrari',200,2)],
        2024: [('Verstappen','Red Bull',437,9),('Norris','McLaren',374,4),('Leclerc','Ferrari',356,3),('Piastri','McLaren',292,2),('Sainz','Ferrari',290,2)],
    }

    for year, drivers in historical.items():
        sorted_drivers = sorted(drivers, key=lambda x: x[2], reverse=True)
        for pos, (driver, constructor, points, wins) in enumerate(sorted_drivers, 1):
            data.append({
                'year': year,
                'driver': driver,
                'constructor': constructor,
                'points': points,
                'wins': wins,
                'position': pos,
                'champion': 1 if pos == 1 else 0
            })

    df = pd.DataFrame(data)
    df.to_csv('f1_data.csv', index=False)
    print(f"Historical data saved: {len(df)} rows")
    return df


def get_2026_standings():
    # Real standings after 2 races (Australia + China)
    # Russell leads, Antonelli won China, Hamilton on Ferrari podium
    # McLaren DNF both cars in China, Verstappen struggling badly
    standings = [
        ('Russell',    'Mercedes',    51, 1),
        ('Antonelli',  'Mercedes',    47, 1),
        ('Leclerc',    'Ferrari',     36, 0),
        ('Hamilton',   'Ferrari',     35, 0),
        ('Bearman',    'Haas',        19, 0),
        ('Norris',     'McLaren',     18, 0),
        ('Verstappen', 'Red Bull',     8, 0),
        ('Lawson',     'Racing Bulls', 8, 0),
        ('Hadjar',     'Racing Bulls', 4, 0),
        ('Piastri',    'McLaren',      3, 0),
        ('Bortoleto',  'Audi',         2, 0),
        ('Sainz',      'Williams',     2, 0),
        ('Colapinto',  'Alpine',       1, 0),
        ('Gasly',      'Alpine',       0, 0),
        ('Albon',      'Williams',     0, 0),
        ('Ocon',       'Haas',         0, 0),
    ]

    races_done = 2
    total_races = 24

    rows = []
    for driver, constructor, points, wins in standings:
        projected = round(points * (total_races / races_done))
        rows.append({
            'driver': driver,
            'constructor': constructor,
            'current_points': points,
            'current_wins': wins,
            'races_done': races_done,
            'races_remaining': total_races - races_done,
            'projected_points': projected,
        })

    df_2026 = pd.DataFrame(rows)
    df_2026.to_csv('standings_2026.csv', index=False)
    print(f"2026 standings saved: {len(df_2026)} drivers")
    return df_2026


if __name__ == "__main__":
    df_hist = collect_all_data()
    df_2026 = get_2026_standings()
    print("\n--- 2026 Standings After 2 Races ---")
    print(df_2026[['driver','constructor','current_points','projected_points']].to_string(index=False))