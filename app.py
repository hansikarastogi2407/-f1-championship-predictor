import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="F1 2026 Championship Predictor", page_icon="🏎️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;800&family=Barlow:wght@400;500;600&display=swap');

* { font-family: 'Barlow', sans-serif; }
[data-testid="stAppViewContainer"] { background: linear-gradient(135deg, #1a0000 0%, #2d0000 30%, #1a0a00 70%, #0d0d0d 100%); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: #1a0000; }

h1 { font-family: 'Barlow Condensed', sans-serif !important; font-size: 3.5rem !important; font-weight: 800 !important;
     background: linear-gradient(90deg, #ff0000, #ff6b00, #ffffff); -webkit-background-clip: text;
     -webkit-text-fill-color: transparent; letter-spacing: 2px; }
h2, h3 { font-family: 'Barlow Condensed', sans-serif !important; font-weight: 600 !important;
          color: #ff4444 !important; letter-spacing: 1px; text-transform: uppercase; }

[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(255,0,0,0.15), rgba(255,60,0,0.05));
    border: 1px solid rgba(255,0,0,0.3);
    border-radius: 12px; padding: 20px;
    box-shadow: 0 4px 20px rgba(255,0,0,0.1);
}
[data-testid="stMetricLabel"] { color: #ff8888 !important; font-size: 0.8rem !important; text-transform: uppercase; letter-spacing: 1px; }
[data-testid="stMetricValue"] { color: #ffffff !important; font-family: 'Barlow Condensed', sans-serif !important;
                                  font-size: 2rem !important; font-weight: 700 !important; }

.stDataFrame { border: 1px solid rgba(255,0,0,0.2); border-radius: 10px; }
[data-testid="stDivider"] { border-color: rgba(255,0,0,0.3) !important; }

.race-badge {
    display: inline-block; background: linear-gradient(90deg, #e10600, #ff6b00);
    color: white; padding: 4px 16px; border-radius: 20px;
    font-size: 0.85rem; font-weight: 600; letter-spacing: 1px; margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

df = pd.read_csv('predictions.csv')
df_2026 = pd.read_csv('standings_2026.csv')
hist = pd.read_csv('f1_data.csv')

st.markdown('<div class="race-badge">🏁 LIVE SEASON · 2026</div>', unsafe_allow_html=True)
st.title("F1 2026 Championship Predictor")
st.markdown("*Machine learning predictions · 2010–2026 historical data · Updated after Round 2 — Chinese GP, Shanghai*")
st.divider()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🏆 Predicted Champion", df.iloc[0]['driver'])
with col2:
    st.metric("📊 Win Probability", f"{df.iloc[0]['win_probability']}%")
with col3:
    st.metric("🏁 Season Progress", "2 of 24 Races")
with col4:
    st.metric("⚡ Points Leader", f"{df_2026.iloc[0]['driver']} — {df_2026.iloc[0]['current_points']}pts")

st.divider()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🎯 Championship Win Probability")
    top10 = df.head(10).sort_values('win_probability')
    colors = ['#e10600' if w > 30 else '#ff6b00' if w > 10 else '#555555' for w in top10['win_probability']]
    fig1 = px.bar(top10, x='win_probability', y='driver', orientation='h',
                  text=top10['win_probability'].apply(lambda x: f"{x}%"),
                  labels={'win_probability': 'Win Probability (%)', 'driver': ''})
    fig1.update_traces(marker_color=colors, textposition='outside',
                       textfont=dict(color='white', size=13))
    fig1.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_color='white', height=420,
        xaxis=dict(gridcolor='rgba(255,255,255,0.08)', range=[0, 65], color='#aaa'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.08)', color='white', tickfont=dict(size=13)),
        margin=dict(l=10, r=60, t=20, b=20)
    )
    st.plotly_chart(fig1, use_container_width=True)

with col_right:
    st.subheader("📍 Current 2026 Standings")
    df_sorted = df_2026.sort_values('current_points', ascending=True).tail(10)
    fig2 = px.bar(df_sorted, x='current_points', y='driver', orientation='h',
                  text='current_points', color='constructor',
                  labels={'current_points': 'Points', 'driver': ''},
                  color_discrete_map={
                      'Mercedes': '#00D2BE', 'Ferrari': '#E8002D',
                      'McLaren': '#FF8000', 'Red Bull': '#3671C6',
                      'Haas': '#B6BABD', 'Racing Bulls': '#6692FF',
                      'Alpine': '#FF87BC', 'Audi': '#C0C0C0',
                      'Williams': '#64C4FF', 'Aston Martin': '#358C75'})
    fig2.update_traces(textposition='outside', textfont=dict(color='white', size=12))
    fig2.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_color='white', height=420,
        xaxis=dict(gridcolor='rgba(255,255,255,0.08)', color='#aaa'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.08)', color='white', tickfont=dict(size=13)),
        legend=dict(bgcolor='rgba(0,0,0,0.4)', font_color='white', bordercolor='rgba(255,0,0,0.2)', borderwidth=1),
        margin=dict(l=10, r=60, t=20, b=20)
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.subheader("📈 Projected Full Season vs Current Points")
merged = df_2026.merge(df[['driver', 'win_probability']], on='driver', how='left')
fig3 = px.scatter(merged, x='current_points', y='projected_points',
                  size='win_probability', color='constructor',
                  hover_name='driver', text='driver',
                  size_max=60,
                  labels={'current_points': 'Points After 2 Races', 'projected_points': 'Projected Final Points'},
                  color_discrete_map={
                      'Mercedes': '#00D2BE', 'Ferrari': '#E8002D',
                      'McLaren': '#FF8000', 'Red Bull': '#3671C6',
                      'Haas': '#B6BABD', 'Racing Bulls': '#6692FF',
                      'Alpine': '#FF87BC', 'Audi': '#C0C0C0',
                      'Williams': '#64C4FF', 'Aston Martin': '#358C75'})
fig3.update_traces(textposition='top center', textfont=dict(color='white', size=11))
fig3.update_layout(
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    font_color='white', height=430,
    xaxis=dict(gridcolor='rgba(255,255,255,0.08)', color='#aaa'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.08)', color='#aaa'),
    legend=dict(bgcolor='rgba(0,0,0,0.4)', font_color='white', bordercolor='rgba(255,0,0,0.2)', borderwidth=1),
    margin=dict(l=20, r=20, t=20, b=20)
)
st.plotly_chart(fig3, use_container_width=True)

st.divider()
st.subheader("🏅 Historical Champions 2010–2024")
champs = hist[hist['champion'] == 1][['year', 'driver', 'constructor', 'points', 'wins']]
fig4 = px.bar(champs.sort_values('year'), x='year', y='points',
              color='driver', text='driver',
              labels={'points': 'Championship Points', 'year': 'Season'})
fig4.update_traces(textangle=0, textposition='outside', textfont=dict(size=10, color='white'))
fig4.update_layout(
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    font_color='white', showlegend=False, height=380,
    xaxis=dict(gridcolor='rgba(255,255,255,0.08)', color='#aaa', tickmode='linear'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.08)', color='#aaa'),
    margin=dict(l=20, r=20, t=30, b=20)
)
st.plotly_chart(fig4, use_container_width=True)

st.divider()
st.subheader("📋 Full Predictions Table")
st.dataframe(
    df[['rank', 'driver', 'constructor', 'current_points', 'projected_points', 'current_wins', 'win_probability']],
    use_container_width=True, hide_index=True
)

st.markdown("---")
st.caption("⚡ Model: Random Forest Classifier · Trained on 15 seasons of F1 data (2010–2026) · Pace-weighted using 2026 car performance data")
