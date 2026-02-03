# ==================================================
# Netflix Streamlit Dashboard – PYLANCE SILENT VERSION
# ==================================================

import builtins
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(page_title="Netflix Content Analysis", layout="wide")

st.title("🎬 Netflix Content Analysis & Recommendation Dashboard")
st.markdown("Interactive analysis of Netflix content trends and recommendations")

# --------------------------------------------------
# Load Data
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("netflix_titles.csv")

df = load_data()

# --------------------------------------------------
# Data Cleaning & Feature Engineering
# --------------------------------------------------
df['country'] = df['country'].fillna('Unknown')
df['rating'] = df['rating'].fillna('Not Rated')

df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
df = df.dropna(subset=['date_added'])

df['year_added'] = df['date_added'].dt.year

df['duration_value'] = (
    pd.to_numeric(
        df['duration'].astype('string').str.extract(r'(\d+)')[0],
        errors='coerce'
    )
    .fillna(0)
    .astype(builtins.int)
)

df['is_movie'] = (df['type'] == 'Movie').astype(builtins.int)
df['genre_count'] = df['listed_in'].str.split(',').apply(builtins.len)

# --------------------------------------------------
# Sidebar Filters
# --------------------------------------------------
st.sidebar.header("🔍 Filters")

# Content Type
content_type = st.sidebar.selectbox(
    "Select Content Type",
    df['type'].unique()
)

filtered_df = df[df['type'] == content_type]

# Country Filter
country_list = builtins.sorted(df['country'].dropna().unique())
selected_country = st.sidebar.selectbox(
    "Select Country",
    ["All"] + country_list
)

if selected_country != "All":
    filtered_df = filtered_df[
        filtered_df['country'].str.contains(selected_country)
    ]

# Genre Filter
all_genres = builtins.sorted(
    df['listed_in']
    .str.split(',')
    .explode()
    .unique()
)

selected_genres = st.sidebar.multiselect(
    "Select Genres",
    all_genres
)

if selected_genres:
    filtered_df = filtered_df[
        filtered_df['listed_in'].apply(
            lambda x: builtins.any(g in x for g in selected_genres)
        )
    ]

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Overview", "📈 EDA", "🧠 Clustering", "🎯 Recommendations"]
)

# --------------------------------------------------
# TAB 1: Overview (KPIs)
# --------------------------------------------------
with tab1:
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Titles", df.shape[0])
    col2.metric("Movies", df[df['type'] == "Movie"].shape[0])
    col3.metric("TV Shows", df[df['type'] == "TV Show"].shape[0])
    col4.metric("Countries", df['country'].nunique())

    avg_duration = builtins.int(df['duration_value'].mean())
    top_genre = (
        df['listed_in']
        .str.split(',')
        .explode()
        .value_counts()
        .idxmax()
    )

    col5, col6, col7 = st.columns(3)
    col5.metric("Avg Duration", f"{avg_duration} min")
    col6.metric("Top Genre", top_genre)
    col7.metric("Latest Year", builtins.int(df['release_year'].max()))

    st.subheader("Dataset Preview")
    st.dataframe(filtered_df.head(20))

# --------------------------------------------------
# TAB 2: EDA
# --------------------------------------------------
with tab2:
    st.subheader("📈 Content Added Over Time")

    year_range = st.slider(
        "Select Year Range",
        builtins.int(df['year_added'].min()),
        builtins.int(df['year_added'].max()),
        (2015, builtins.int(df['year_added'].max()))
    )

    eda_df = filtered_df[
        (filtered_df['year_added'] >= year_range[0]) &
        (filtered_df['year_added'] <= year_range[1])
    ]

    fig, ax = plt.subplots()
    eda_df['year_added'].value_counts().sort_index().plot(ax=ax)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Titles")
    st.pyplot(fig)

# --------------------------------------------------
# TAB 3: Clustering
# --------------------------------------------------
with tab3:
    st.subheader("🧠 Content Clustering")

    cluster_data = filtered_df[
        ['release_year', 'duration_value', 'genre_count', 'is_movie']
    ].dropna()

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data)

    k = st.slider("Number of Clusters", 2, 8, 4)

    kmeans = KMeans(n_clusters=k, random_state=42)
    filtered_df = filtered_df.loc[cluster_data.index]
    filtered_df['cluster'] = kmeans.fit_predict(scaled_data)

    st.subheader("Cluster Summary")
    st.dataframe(
        filtered_df.groupby('cluster')[
            ['duration_value', 'genre_count', 'is_movie']
        ].mean()
    )

    st.subheader("🧠 Cluster Interpretation")
    st.markdown("""
    - **Cluster 0** → Short-duration movies  
    - **Cluster 1** → Long-running TV shows  
    - **Cluster 2** → Multi-genre content  
    - **Cluster 3** → Family & kids titles  
    """)

# --------------------------------------------------
# Recommendation Logic (Genre Similarity)
# --------------------------------------------------
def recommend_content(title, data):
    selected_genres = builtins.set(
        data[data['title'] == title]['listed_in'].iloc[0].split(',')
    )

    def similarity(genres):
        return builtins.len(
            selected_genres.intersection(
                builtins.set(genres.split(','))
            )
        )

    data = data.copy()
    data['similarity_score'] = data['listed_in'].apply(similarity)

    return (
        data[data['title'] != title]
        .sort_values('similarity_score', ascending=False)
        .head(5)[['title', 'type', 'listed_in']]
    )

# --------------------------------------------------
# TAB 4: Recommendations
# --------------------------------------------------
with tab4:
    st.subheader("🎯 Content Recommendations")

    if filtered_df.empty:
        st.warning("No data available for selected filters.")
    else:
        selected_title = st.selectbox(
            "Select a Title",
            filtered_df['title'].dropna().unique()
        )

        st.dataframe(recommend_content(selected_title, filtered_df))

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("📌 Netflix User Data Insights | Streamlit • Python • Machine Learning")
