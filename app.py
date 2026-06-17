import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# Page setup
# --------------------------------------------------

st.set_page_config(
    page_title="Diamond Dynamics Dashboard",
    layout="wide"
)


# --------------------------------------------------
# Dashboard styling
# --------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #07111f;
        color: #f8fafc;
    }

    section[data-testid="stSidebar"] {
        background-color: #020617;
    }

    section[data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }

    h1, h2, h3 {
        color: #f8fafc;
    }

    .hero-box {
        background: linear-gradient(135deg, #0f766e, #0369a1);
        padding: 30px;
        border-radius: 18px;
        margin-bottom: 25px;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.35);
    }

    .hero-title {
        font-size: 42px;
        font-weight: 900;
        color: white;
    }

    .hero-subtitle {
        font-size: 18px;
        color: #e0f2fe;
        margin-top: 8px;
    }

    .white-card {
        background-color: #ffffff;
        color: #111827;
        padding: 24px;
        border-radius: 16px;
        margin-top: 18px;
        margin-bottom: 22px;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.28);
    }

    .white-card h1,
    .white-card h2,
    .white-card h3,
    .white-card p,
    .white-card div {
        color: #111827;
    }

    div[data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 18px;
        border-radius: 14px;
        border-left: 6px solid #22d3ee;
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] div {
        color: #111827 !important;
    }

    .prediction-box {
        background: linear-gradient(135deg, #0e7490, #0f766e);
        padding: 30px;
        border-radius: 18px;
        text-align: center;
        margin-top: 25px;
        color: white;
        box-shadow: 0 14px 38px rgba(0, 0, 0, 0.35);
    }

    .prediction-text {
        font-size: 34px;
        font-weight: 900;
        color: white;
        line-height: 1.6;
    }

    .flow-step {
        background-color: #f8fafc;
        color: #111827;
        padding: 18px;
        border-radius: 14px;
        border-left: 6px solid #0ea5e9;
        margin-bottom: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

DATA_PATH = r"c:\Users\ADMIN\Downloads\diamonds.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


try:
    df = load_data()
except Exception as e:
    st.error("Dataset not found. Please check the diamonds.csv path.")
    st.write(e)
    st.stop()


# --------------------------------------------------
# Encoding maps
# --------------------------------------------------

cut_map = {
    "Fair": 1,
    "Good": 2,
    "Very Good": 3,
    "Premium": 4,
    "Ideal": 5
}

color_map = {
    "J": 1,
    "I": 2,
    "H": 3,
    "G": 4,
    "F": 5,
    "E": 6,
    "D": 7
}

clarity_map = {
    "I1": 1,
    "SI2": 2,
    "SI1": 3,
    "VS2": 4,
    "VS1": 5,
    "VVS2": 6,
    "VVS1": 7,
    "IF": 8
}


# --------------------------------------------------
# Data preprocessing
# --------------------------------------------------

@st.cache_data
def prepare_data(data):
    data = data.copy()

    data = data.drop_duplicates()
    data = data.dropna()

    data = data[
        (data["x"] > 0) &
        (data["y"] > 0) &
        (data["z"] > 0)
    ]

    data["volume"] = data["x"] * data["y"] * data["z"]

    data["cut_score"] = data["cut"].map(cut_map)
    data["color_score"] = data["color"].map(color_map)
    data["clarity_score"] = data["clarity"].map(clarity_map)

    data["quality_score"] = (
        data["cut_score"] +
        data["color_score"] +
        data["clarity_score"]
    )

    data["depth_table_ratio"] = data["depth"] / data["table"]

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()

    return data


df_clean = prepare_data(df)

features = [
    "carat",
    "depth",
    "table",
    "x",
    "y",
    "z",
    "volume",
    "cut_score",
    "color_score",
    "clarity_score",
    "quality_score",
    "depth_table_ratio"
]


# --------------------------------------------------
# Train prediction model
# --------------------------------------------------

@st.cache_resource
def train_price_model(data):
    X = data[features]
    y = data["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(
        n_estimators=80,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return model, scaler, mae, rmse, r2


model, scaler, mae, rmse, r2 = train_price_model(df_clean)


# --------------------------------------------------
# Model comparison
# --------------------------------------------------

@st.cache_data
def compare_models(data):
    sample_data = data.sample(min(12000, len(data)), random_state=42)

    X = sample_data[features]
    y = sample_data["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    scaler_local = StandardScaler()
    X_train_scaled = scaler_local.fit_transform(X_train)
    X_test_scaled = scaler_local.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=60,
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=60,
            random_state=42
        )
    }

    results = []

    for name, reg_model in models.items():
        reg_model.fit(X_train_scaled, y_train)
        pred = reg_model.predict(X_test_scaled)

        results.append({
            "Model": name,
            "MAE": round(mean_absolute_error(y_test, pred), 2),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, pred)), 2),
            "R2 Score": round(r2_score(y_test, pred), 4)
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("R2 Score", ascending=False)

    return results_df


comparison_df = compare_models(df_clean)


# --------------------------------------------------
# Market segmentation
# --------------------------------------------------

@st.cache_data
def create_segments(data):
    sample_data = data.sample(min(4000, len(data)), random_state=42).copy()

    cluster_features = [
        "carat",
        "price",
        "volume",
        "quality_score"
    ]

    X_cluster = sample_data[cluster_features]

    cluster_scaler = StandardScaler()
    X_scaled = cluster_scaler.fit_transform(X_cluster)

    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    sample_data["cluster"] = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)

    sample_data["PCA1"] = pca_result[:, 0]
    sample_data["PCA2"] = pca_result[:, 1]

    profile = sample_data.groupby("cluster")[cluster_features].mean().reset_index()
    profile = profile.sort_values("price").reset_index(drop=True)

    segment_names = [
        "Budget Diamonds",
        "Mid Range Diamonds",
        "Premium Diamonds",
        "Luxury Diamonds"
    ]

    profile["Segment Name"] = segment_names

    segment_map = dict(zip(profile["cluster"], profile["Segment Name"]))
    sample_data["Segment Name"] = sample_data["cluster"].map(segment_map)

    return sample_data, profile


segmented_df, segment_profile = create_segments(df_clean)


# --------------------------------------------------
# Category functions
# --------------------------------------------------

def get_price_category(price):
    if price < 1000:
        return "Budget Diamond"
    elif price < 5000:
        return "Mid Range Diamond"
    elif price < 10000:
        return "Premium Diamond"
    else:
        return "Luxury Diamond"


def get_carat_category(carat):
    if carat < 0.5:
        return "Small Diamond"
    elif carat < 1.0:
        return "Medium Diamond"
    elif carat < 2.0:
        return "Large Diamond"
    else:
        return "Very Large Diamond"


def get_quality_category(quality_score):
    if quality_score <= 8:
        return "Basic Quality"
    elif quality_score <= 12:
        return "Good Quality"
    elif quality_score <= 16:
        return "Premium Quality"
    else:
        return "Excellent Quality"


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">Diamond Dynamics Dashboard</div>
        <div class="hero-subtitle">
            Price prediction, category detection, market segmentation, model comparison,
            and complete project workflow in one interactive dashboard.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

page = st.sidebar.radio(
    "Dashboard Sections",
    [
        "Project Flow",
        "Data Overview",
        "EDA",
        "Price Prediction",
        "Market Segmentation",
        "Model Performance"
    ]
)


# --------------------------------------------------
# Project Flow
# --------------------------------------------------

if page == "Project Flow":

    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.subheader("Project Workflow")

    flow_steps = [
        "1. Load diamond dataset",
        "2. Clean missing, duplicate, and invalid values",
        "3. Perform exploratory data analysis",
        "4. Create new features such as volume and quality score",
        "5. Encode categorical features",
        "6. Scale numerical features",
        "7. Train machine learning regression models",
        "8. Predict diamond price",
        "9. Detect price category",
        "10. Segment diamonds into market groups",
        "11. Compare model performance",
        "12. Present final dashboard insights"
    ]

    for step in flow_steps:
        st.markdown(
            f"""
            <div class="flow-step">{step}</div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------
# Data Overview
# --------------------------------------------------

elif page == "Data Overview":

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", f"{len(df_clean):,}")
    col2.metric("Average Price", f"${df_clean['price'].mean():,.0f}")
    col3.metric("Average Carat", f"{df_clean['carat'].mean():.2f}")
    col4.metric("Model R2 Score", f"{r2:.4f}")

    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.subheader("Dataset Preview")
    st.dataframe(df_clean.head(12), use_container_width=True)

    st.subheader("Available Columns")
    st.write(df_clean.columns.tolist())
    st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------
# EDA
# --------------------------------------------------

elif page == "EDA":

    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.subheader("Exploratory Data Analysis")

    sample_df = df_clean.sample(min(5000, len(df_clean)), random_state=42)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df_clean,
            x="price",
            nbins=50,
            title="Price Distribution",
            color_discrete_sequence=["#0ea5e9"]
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            df_clean,
            x="carat",
            nbins=50,
            title="Carat Distribution",
            color_discrete_sequence=["#f59e0b"]
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig = px.scatter(
            sample_df,
            x="carat",
            y="price",
            color="cut",
            title="Price vs Carat"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        avg_cut = df_clean.groupby("cut", as_index=False)["price"].mean()

        fig = px.bar(
            avg_cut,
            x="cut",
            y="price",
            color="cut",
            title="Average Price by Cut"
        )
        st.plotly_chart(fig, use_container_width=True)

    corr = df_clean[
        ["carat", "depth", "table", "price", "x", "y", "z", "volume", "quality_score"]
    ].corr()

    fig = px.imshow(
        corr,
        title="Correlation Heatmap",
        color_continuous_scale="RdBu_r"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------
# Price Prediction
# --------------------------------------------------

elif page == "Price Prediction":

    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.subheader("Diamond Price Prediction")

    with st.form("prediction_form"):

        col1, col2, col3 = st.columns(3)

        with col1:
            carat = st.number_input("Carat", 0.10, 5.00, 1.00, 0.01)
            cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
            color = st.selectbox("Color", ["J", "I", "H", "G", "F", "E", "D"])

        with col2:
            clarity = st.selectbox(
                "Clarity",
                ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
            )
            depth = st.number_input("Depth", 40.0, 80.0, 61.5, 0.1)
            table = st.number_input("Table", 40.0, 80.0, 57.0, 0.1)

        with col3:
            x = st.number_input("Length x", 1.0, 15.0, 6.0, 0.1)
            y = st.number_input("Width y", 1.0, 15.0, 6.0, 0.1)
            z = st.number_input("Depth z", 1.0, 10.0, 3.8, 0.1)

        submitted = st.form_submit_button("Predict Price")

    if submitted:

        volume = x * y * z

        cut_score = cut_map[cut]
        color_score = color_map[color]
        clarity_score = clarity_map[clarity]

        quality_score = cut_score + color_score + clarity_score
        depth_table_ratio = depth / table

        input_data = pd.DataFrame([{
            "carat": carat,
            "depth": depth,
            "table": table,
            "x": x,
            "y": y,
            "z": z,
            "volume": volume,
            "cut_score": cut_score,
            "color_score": color_score,
            "clarity_score": clarity_score,
            "quality_score": quality_score,
            "depth_table_ratio": depth_table_ratio
        }])

        input_scaled = scaler.transform(input_data)
        predicted_price = model.predict(input_scaled)[0]

        price_category = get_price_category(predicted_price)
        carat_category = get_carat_category(carat)
        quality_category = get_quality_category(quality_score)

        st.markdown(
            f"""
            <div class="prediction-box">
                <div class="prediction-text">
                    Predicted Price: ${predicted_price:,.2f}<br>
                    Category: {price_category}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader("Detailed Category Detection")

        c1, c2, c3 = st.columns(3)

        c1.metric("Price Category", price_category)
        c2.metric("Carat Category", carat_category)
        c3.metric("Quality Category", quality_category)

        st.subheader("Input Used for Prediction")
        st.dataframe(input_data, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------
# Market Segmentation
# --------------------------------------------------

elif page == "Market Segmentation":

    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.subheader("Market Segmentation")

    st.write("KMeans clustering is used to divide diamonds into business-friendly segments.")

    st.dataframe(segment_profile, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(
            segmented_df,
            x="PCA1",
            y="PCA2",
            color="Segment Name",
            title="PCA Visualization of Diamond Segments"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        count_df = segmented_df["Segment Name"].value_counts().reset_index()
        count_df.columns = ["Segment Name", "Count"]

        fig = px.bar(
            count_df,
            x="Segment Name",
            y="Count",
            color="Segment Name",
            title="Segment Count"
        )
        st.plotly_chart(fig, use_container_width=True)

    fig = px.box(
        segmented_df,
        x="Segment Name",
        y="price",
        color="Segment Name",
        title="Price Distribution by Segment"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------
# Model Performance
# --------------------------------------------------

elif page == "Model Performance":

    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", f"{mae:,.2f}")
    col2.metric("RMSE", f"{rmse:,.2f}")
    col3.metric("R2 Score", f"{r2:.4f}")

    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.subheader("Regression Model Comparison")

    st.dataframe(comparison_df, use_container_width=True)

    fig = px.bar(
        comparison_df,
        x="Model",
        y="R2 Score",
        color="Model",
        title="Model Comparison by R2 Score"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(
        comparison_df,
        x="Model",
        y="RMSE",
        color="Model",
        title="Model Comparison by RMSE"
    )
    st.plotly_chart(fig, use_container_width=True)

    best_model = comparison_df.iloc[0]

    st.success(
        f"Best model is {best_model['Model']} with R2 Score {best_model['R2 Score']}."
    )

    st.markdown("</div>", unsafe_allow_html=True)
