import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="F1 2026 Championship Predictor", page_icon="🏎️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;800&family=Barlow:wght@400;500;600&display=swap');
* { font-family: 'Barlow', sans-serif; }
[data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #0a0a0a 0%, #1a0500 40%, #0d0a00 70%, #0a0a0a 100%) !important;
}
[data-testid="stMain"] {
    background: transparent !important;
}
.main .block-container {
    background: transparent !important;
}
[data-testid="stHeader"] { background: transparent !important; }
section[data-testid="stSidebar"] { background: #0a0000 !important; }
h1 {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 3.5rem !important; font-weight: 800 !important;
    color: #ff2200 !important;
    letter-spacing: 3px;
}
h2, h3 {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 700 !important; color: #ff4444 !important;
    letter-spacing: 2px; text-transform: uppercase; font-size: 1.3rem !important;
}
[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(255,50,0,0.15), rgba(255,100,0,0.05)) !important;
    border: 1px solid rgba(255,80,0,0.4) !important;
    border-radius: 14px !important; padding: 18px !important;
}
[data-testid="stMetricLabel"] {
    color: #ff9966 !important; font-size: 0.75rem !important;
    text-transform: uppercase !important; letter-spacing: 1.5px !important;
}
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 1.9rem !important; font-weight: 700 !important;
}
[data-testid="stDivider"] { border-color: rgba(255,80,0,0.3) !important; }
div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# Racing animation
st.markdown("""
<div style="width:100%;height:55px;position:relative;overflow:hidden;margin-bottom:8px;
background:linear-gradient(90deg,transparent,rgba(255,60,0,0.05),transparent);
border-bottom:1px solid rgba(255,60,0,0.2);">
<div style="position:absolute;bottom:10px;left:0;right:0;height:1px;
background:repeating-linear-gradient(90deg,rgba(255,255,255,0.1) 0px,rgba(255,255,255,0.1) 20px,transparent 20px,transparent 40px)"></div>
<div style="position:absolute;bottom:4px;left:0;right:0;height:2px;
background:linear-gradient(90deg,transparent,rgba(255,60,0,0.5),transparent)"></div>
<style>
@keyframes f1race{0%{left:-100px}100%{left:110%}}
@keyframes f1race2{0%{left:-100px}100%{left:110%}}
.car1{position:absolute;top:5px;font-size:2rem;animation:f1race 3.5s linear infinite;filter:drop-shadow(0 0 8px rgba(255,80,0,0.9));}
.car2{position:absolute;top:8px;font-size:1.5rem;animation:f1race2 3.5s linear infinite 1.8s;filter:drop-shadow(0 0 5px rgba(255,200,0,0.7));opacity:0.8;}
</style>
<div class="car1">🏎️</div>
<div class="car2">🏎️</div>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
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
    data = []
    for year, drivers in historical.items():
        sorted_drivers = sorted(drivers, key=lambda x: x[2], reverse=True)
        for pos, (driver, constructor, points, wins) in enumerate(sorted_drivers, 1):
            data.append({'year': year, 'driver': driver, 'constructor': constructor,
                         'points': points, 'wins': wins, 'position': pos,
                         'champion': 1 if pos == 1 else 0})
    return pd.DataFrame(data)

@st.cache_data
def get_2026():
    standings = [
        ('Russell','Mercedes',51,1),('Antonelli','Mercedes',47,1),
        ('Leclerc','Ferrari',36,0),('Hamilton','Ferrari',35,0),
        ('Bearman','Haas',19,0),('Norris','McLaren',18,0),
        ('Verstappen','Red Bull',8,0),('Lawson','Racing Bulls',8,0),
        ('Hadjar','Racing Bulls',4,0),('Piastri','McLaren',3,0),
        ('Bortoleto','Audi',2,0),('Sainz','Williams',2,0),
        ('Colapinto','Alpine',1,0),('Gasly','Alpine',0,0),
        ('Albon','Williams',0,0),('Ocon','Haas',0,0),
    ]
    rows = []
    for driver, constructor, points, wins in standings:
        rows.append({'driver': driver, 'constructor': constructor,
                     'current_points': points, 'current_wins': wins,
                     'projected_points': round(points * (24/2))})
    return pd.DataFrame(rows)

@st.cache_data
def run_model(hist_df, df_2026):
    le_driver, le_constructor = LabelEncoder(), LabelEncoder()
    all_drivers = list(hist_df['driver'].unique()) + list(df_2026['driver'].unique())
    all_constructors = list(hist_df['constructor'].unique()) + list(df_2026['constructor'].unique())
    le_driver.fit(all_drivers)
    le_constructor.fit(all_constructors)
    hist_df['driver_enc'] = le_driver.transform(hist_df['driver'])
    hist_df['constructor_enc'] = le_constructor.transform(hist_df['constructor'])
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(hist_df[['points','wins','driver_enc','constructor_enc']], hist_df['champion'])
    pace_boost = {'Russell':0.15,'Antonelli':0.14,'Leclerc':0.10,'Hamilton':0.09,
                  'Bearman':0.03,'Norris':0.05,'Verstappen':-0.10,'Piastri':0.02}
    results = []
    for _, row in df_2026.iterrows():
        driver, constructor = row['driver'], row['constructor']
        d_enc = le_driver.transform([driver])[0] if driver in le_driver.classes_ else 0
        c_enc = le_constructor.transform([constructor])[0] if constructor in le_constructor.classes_ else 0
        prob = model.predict_proba([[row['projected_points'], row['current_wins'], d_enc, c_enc]])[0][1]
        final_prob = max(0.01, min(0.99, prob + pace_boost.get(driver, 0)))
        results.append({'driver': driver, 'constructor': constructor,
                        'current_points': int(row['current_points']),
                        'projected_points': int(row['projected_points']),
                        'current_wins': int(row['current_wins']),
                        'win_probability': round(final_prob * 100, 1)})
    df = pd.DataFrame(results).sort_values('win_probability', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1
    return df

COLORS = {'Mercedes':'#00D2BE','Ferrari':'#E8002D','McLaren':'#FF8000','Red Bull':'#3671C6',
          'Haas':'#B6BABD','Racing Bulls':'#6692FF','Alpine':'#FF87BC','Audi':'#C0C0C0',
          'Williams':'#64C4FF','Aston Martin':'#358C75'}

hist = load_data()
df_2026 = get_2026()
df = run_model(hist.copy(), df_2026.copy())

st.title("F1 2026 CHAMPIONSHIP PREDICTOR")
st.markdown("*Machine learning predictions · 15 seasons of data · Updated after Round 2 — Chinese GP*")
st.divider()

col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("🏆 Predicted Champion", df.iloc[0]['driver'])
with col2: st.metric("📊 Win Probability", f"{df.iloc[0]['win_probability']}%")
with col3: st.metric("🏁 Season Progress", "2 of 24 Races")
with col4: st.metric("⚡ Points Leader", f"{df_2026.iloc[0]['driver']} — {df_2026.iloc[0]['current_points']}pts")

st.divider()
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🎯 Championship Win Probability")
    top10 = df.head(10).sort_values('win_probability')
    colors = ['#e10600' if w > 30 else '#ff6b00' if w > 10 else '#555555' for w in top10['win_probability']]
    fig1 = px.bar(top10, x='win_probability', y='driver', orientation='h',
                  text=top10['win_probability'].apply(lambda x: f"{x}%"),
                  labels={'win_probability': 'Win Probability (%)', 'driver': ''})
    fig1.update_traces(marker_color=colors, textposition='outside', textfont=dict(color='white', size=13))
    fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                       font_color='white', height=420,
                       xaxis=dict(gridcolor='rgba(255,255,255,0.08)', range=[0,65], color='#aaa'),
                       yaxis=dict(gridcolor='rgba(255,255,255,0.08)', color='white', tickfont=dict(size=13)),
                       margin=dict(l=10,r=60,t=20,b=20))
    st.plotly_chart(fig1, use_container_width=True)

with col_right:
    st.subheader("📍 Current 2026 Standings")
    df_sorted = df_2026.sort_values('current_points', ascending=True).tail(10)
    fig2 = px.bar(df_sorted, x='current_points', y='driver', orientation='h',
                  text='current_points', color='constructor',
                  labels={'current_points': 'Points', 'driver': ''}, color_discrete_map=COLORS)
    fig2.update_traces(textposition='outside', textfont=dict(color='white', size=12))
    fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                       font_color='white', height=420,
                       xaxis=dict(gridcolor='rgba(255,255,255,0.08)', color='#aaa'),
                       yaxis=dict(gridcolor='rgba(255,255,255,0.08)', color='white'),
                       legend=dict(bgcolor='rgba(0,0,0,0.4)', font_color='white'),
                       margin=dict(l=10,r=60,t=20,b=20))
    st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.subheader("📈 Projected Full Season vs Current Points")
merged = df_2026.merge(df[['driver','win_probability']], on='driver', how='left')
fig3 = px.scatter(merged, x='current_points', y='projected_points', size='win_probability',
                  color='constructor', hover_name='driver', text='driver', size_max=60,
                  labels={'current_points':'Points After 2 Races','projected_points':'Projected Final Points'},
                  color_discrete_map=COLORS)
fig3.update_traces(textposition='top center', textfont=dict(color='white', size=11))
fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                   font_color='white', height=430,
                   xaxis=dict(gridcolor='rgba(255,255,255,0.08)', color='#aaa'),
                   yaxis=dict(gridcolor='rgba(255,255,255,0.08)', color='#aaa'),
                   legend=dict(bgcolor='rgba(0,0,0,0.4)', font_color='white'))
st.plotly_chart(fig3, use_container_width=True)

st.divider()
st.subheader("🏅 Historical Champions 2010–2024")
champs = hist[hist['champion']==1][['year','driver','constructor','points','wins']]
fig4 = px.bar(champs.sort_values('year'), x='year', y='points', color='driver', text='driver',
              labels={'points':'Championship Points','year':'Season'})
fig4.update_traces(textangle=0, textposition='outside', textfont=dict(size=10, color='white'))
fig4.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                   font_color='white', showlegend=False, height=380,
                   xaxis=dict(gridcolor='rgba(255,255,255,0.08)', color='#aaa', tickmode='linear'),
                   yaxis=dict(gridcolor='rgba(255,255,255,0.08)', color='#aaa'))
st.plotly_chart(fig4, use_container_width=True)

st.divider()
st.subheader("📋 Full Predictions Table")

display_df = df[['rank','driver','constructor','current_points','projected_points','current_wins','win_probability']].copy()
display_df.columns = ['Rank','Driver','Constructor','Current Points','Projected Points','Wins','Win Probability %']
display_df['Win Probability %'] = display_df['Win Probability %'].apply(lambda x: f"{x}%")

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Rank": st.column_config.NumberColumn(width="small"),
        "Driver": st.column_config.TextColumn(width="medium"),
        "Constructor": st.column_config.TextColumn(width="medium"),
        "Current Points": st.column_config.NumberColumn(width="medium"),
        "Projected Points": st.column_config.NumberColumn(width="medium"),
        "Wins": st.column_config.NumberColumn(width="small"),
        "Win Probability %": st.column_config.TextColumn(width="medium"),
    }
)

st.caption("⚡ Model: Random Forest Classifier · Trained on 15 seasons of F1 data (2010–2026) · Pace-weighted using 2026 car performance data")
