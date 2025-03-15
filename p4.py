import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import calendar
from datetime import datetime

st.set_page_config(
    page_title="Cornell Tech Weather Dashboard",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Cornell Tech Weather Data Explorer")
st.markdown(
    """
Welcome to the **Cornell Tech Weather Data Explorer**! This interactive dashboard allows you to explore 
temperature trends at Cornell Tech. Upload your weather data file to begin exploring historical and 
seasonal weather patterns.
"""
)

# File upload section
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload weather.csv file", type=["csv", "txt"])

# Variable to store our data
data = None
available_years = []

if uploaded_file is not None:
    try:
        # Read the uploaded file
        data = pd.read_csv(uploaded_file)

        # Process the data
        data["time"] = pd.to_datetime(data["time"])
        data["year"] = data["time"].dt.year
        data["month"] = data["time"].dt.month
        data["day"] = data["time"].dt.day
        data["month_name"] = data["time"].dt.strftime("%b")
        data["season"] = pd.cut(
            data["month"],
            bins=[0, 3, 6, 9, 12],
            labels=["Winter", "Spring", "Summer", "Fall"],
            include_lowest=True,
        )

        # Convert temperature from Kelvin to Fahrenheit and Celsius
        data["Ftemp"] = (data["Ktemp"] - 273.15) * (9 / 5) + 32
        data["Ctemp"] = data["Ktemp"] - 273.15

        available_years = sorted(data["year"].unique())
        st.success(
            f"✅ Data successfully loaded! The dataset spans from **{min(available_years)}** to **{max(available_years)}**."
        )
    except Exception as e:
        st.error(f"⚠️ Error processing uploaded file: {e}")
        st.info(
            "Please ensure your file contains columns for 'time' and 'Ktemp' (temperature in Kelvin)."
        )
else:
    st.info(
        "👆 Please upload your weather data file (CSV format) using the sidebar upload button."
    )
    st.markdown(
        """
    ### Expected Data Format
    Your CSV file should include at least these columns:
    - `time`: Date and time of the measurement (YYYY-MM-DD format)
    - `Ktemp`: Temperature in Kelvin
    
    Optional columns:
    - `longitude`: Geographic longitude
    - `latitude`: Geographic latitude
    
    Example:
    ```
    time,longitude,latitude,Ktemp
    1950-01-01 9:00:00,286,40.75,274.39734
    1950-01-02 9:00:00,286,40.75,277.07593
    ...
    ```
    """
    )

# Only show visualization options if data is loaded
if data is not None:
    # --- Sidebar with Navigation ---
    st.sidebar.header("Dashboard Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Monthly Temperature Analysis",
            "When Will Cornell Tech Be Warm?",
            "Creative Temperature Visualization",
        ],
    )

    # --- Shared controls in sidebar ---
    st.sidebar.header("Visualization Controls")

    temp_unit = st.sidebar.radio("Temperature Unit", ["Fahrenheit", "Celsius"])
    temp_col = "Ftemp" if temp_unit == "Fahrenheit" else "Ctemp"
    temp_symbol = "°F" if temp_unit == "Fahrenheit" else "°C"

    if page == "Monthly Temperature Analysis":
        # --- MONTHLY TEMPERATURE ANALYSIS PAGE ---

        # Year selection with slider
        selected_years = st.sidebar.slider(
            "Select Year Range",
            min_value=min(available_years),
            max_value=max(available_years),
            value=(
                min(available_years),
                min(available_years) + min(2, len(available_years) - 1),
            ),
            step=1,
        )

        # Additional display options
        st.sidebar.subheader("Display Options")
        show_minmax = st.sidebar.checkbox("Show Min/Max Temperatures", value=False)
        show_stddev = st.sidebar.checkbox("Show Standard Deviation", value=False)
        add_trend_line = st.sidebar.checkbox("Show Trend Line", value=True)

        # Extra analysis options
        st.sidebar.subheader("Additional Analysis")
        show_seasonal = st.sidebar.checkbox("Show Seasonal Comparison", value=True)
        show_yearly_comparison = st.sidebar.checkbox(
            "Show Year-over-Year Comparison", value=True
        )

        # --- Data processing for visualization ---
        # Filter by selected years
        filtered_data = data[
            (data["year"] >= selected_years[0]) & (data["year"] <= selected_years[1])
        ]

        # Calculate monthly statistics
        def get_monthly_stats(df, temp_col):
            monthly_stats = (
                df.groupby(["year", "month", "month_name"])[temp_col]
                .agg([("mean", "mean"), ("min", "min"), ("max", "max"), ("std", "std")])
                .reset_index()
            )

            month_order = {
                month: i
                for i, month in enumerate(
                    [
                        "Jan",
                        "Feb",
                        "Mar",
                        "Apr",
                        "May",
                        "Jun",
                        "Jul",
                        "Aug",
                        "Sep",
                        "Oct",
                        "Nov",
                        "Dec",
                    ]
                )
            }
            monthly_stats["month_order"] = monthly_stats["month_name"].map(month_order)
            monthly_stats = monthly_stats.sort_values(["year", "month_order"])

            return monthly_stats

        # Get monthly statistics
        monthly_stats = get_monthly_stats(filtered_data, temp_col)

        # --- Main visualization ---
        st.header(f"Monthly Average Temperatures ({temp_unit})")

        # Create main monthly temperature plot
        fig = px.line(
            monthly_stats,
            x="month",
            y="mean",
            color="year",
            labels={"mean": f"Temperature ({temp_symbol})", "month": "Month"},
            markers=True,
            title=f"Monthly Average Temperatures from {selected_years[0]} to {selected_years[1]}",
            line_shape="linear",
        )

        # Customize x-axis to show month names
        fig.update_xaxes(
            tickvals=list(range(1, 13)),
            ticktext=[
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ],
        )

        if show_minmax:
            for year in monthly_stats["year"].unique():
                year_data = monthly_stats[monthly_stats["year"] == year]

                fig.add_trace(
                    go.Scatter(
                        x=year_data["month"],
                        y=year_data["min"],
                        fill=None,
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=year_data["month"],
                        y=year_data["max"],
                        fill="tonexty",
                        mode="lines",
                        line=dict(width=0),
                        name=f"{year} Range",
                        hoverinfo="text",
                        hovertext=[
                            f"Month: {m}<br>Min: {min:.1f}{temp_symbol}<br>Max: {max:.1f}{temp_symbol}"
                            for m, min, max in zip(
                                year_data["month_name"],
                                year_data["min"],
                                year_data["max"],
                            )
                        ],
                        fillcolor=f"rgba(100, 100, 255, 0.2)",
                    )
                )

        if show_stddev:
            for year in monthly_stats["year"].unique():
                year_data = monthly_stats[monthly_stats["year"] == year]

                fig.add_trace(
                    go.Scatter(
                        x=year_data["month"],
                        y=year_data["mean"] - year_data["std"],
                        fill=None,
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=year_data["month"],
                        y=year_data["mean"] + year_data["std"],
                        fill="tonexty",
                        mode="lines",
                        line=dict(width=0),
                        name=f"{year} ±1σ",
                        hoverinfo="text",
                        hovertext=[
                            f"Month: {m}<br>Mean: {mean:.1f}{temp_symbol}<br>StdDev: ±{std:.1f}{temp_symbol}"
                            for m, mean, std in zip(
                                year_data["month_name"],
                                year_data["mean"],
                                year_data["std"],
                            )
                        ],
                        fillcolor=f"rgba(100, 100, 255, 0.1)",
                    )
                )

        fig.update_layout(
            height=600,
            hovermode="x unified",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- Additional Visualizations ---
        if show_seasonal or show_yearly_comparison:
            col1, col2 = st.columns(2)

            # Seasonal comparison
            if show_seasonal:
                with col1:
                    st.subheader("Seasonal Temperature Comparison")

                    seasonal_data = (
                        filtered_data.groupby(["year", "season"])[temp_col]
                        .mean()
                        .reset_index()
                    )

                    season_fig = px.bar(
                        seasonal_data,
                        x="season",
                        y=temp_col,
                        color="year",
                        barmode="group",
                        title=f"Seasonal Average Temperatures by Year",
                        labels={temp_col: f"Average Temperature ({temp_symbol})"},
                    )

                    season_fig.update_xaxes(
                        categoryorder="array",
                        categoryarray=["Winter", "Spring", "Summer", "Fall"],
                    )

                    st.plotly_chart(season_fig, use_container_width=True)

            # Year-over-Year comparison
            if show_yearly_comparison:
                with col2:
                    st.subheader("Year-over-Year Temperature Trends")

                    yearly_data = (
                        filtered_data.groupby("year")[temp_col]
                        .agg(["mean", "min", "max"])
                        .reset_index()
                    )

                    trend_fig = go.Figure()

                    trend_fig.add_trace(
                        go.Scatter(
                            x=yearly_data["year"],
                            y=yearly_data["mean"],
                            mode="lines+markers",
                            name="Average",
                            line=dict(color="blue"),
                        )
                    )

                    if show_minmax:
                        trend_fig.add_trace(
                            go.Scatter(
                                x=yearly_data["year"],
                                y=yearly_data["min"],
                                mode="lines+markers",
                                name="Minimum",
                                line=dict(color="green", dash="dash"),
                            )
                        )

                        trend_fig.add_trace(
                            go.Scatter(
                                x=yearly_data["year"],
                                y=yearly_data["max"],
                                mode="lines+markers",
                                name="Maximum",
                                line=dict(color="red", dash="dash"),
                            )
                        )

                    if add_trend_line and len(yearly_data) > 1:
                        years_array = yearly_data["year"].values
                        temp_array = yearly_data["mean"].values

                        z = np.polyfit(years_array, temp_array, 1)
                        p = np.poly1d(z)

                        trend_fig.add_trace(
                            go.Scatter(
                                x=yearly_data["year"],
                                y=p(yearly_data["year"]),
                                mode="lines",
                                name=f"Trend: {z[0]:.2f}°/year",
                                line=dict(color="black", dash="dot"),
                            )
                        )

                    trend_fig.update_layout(
                        title=f"Average Temperature Trend ({selected_years[0]}-{selected_years[1]})",
                        xaxis_title="Year",
                        yaxis_title=f"Temperature ({temp_symbol})",
                    )

                    st.plotly_chart(trend_fig, use_container_width=True)

        # --- Data Table ---
        st.header("Monthly Temperature Data")
        st.dataframe(
            monthly_stats[["year", "month_name", "mean", "min", "max", "std"]].rename(
                columns={
                    "month_name": "Month",
                    "mean": f"Mean ({temp_symbol})",
                    "min": f"Min ({temp_symbol})",
                    "max": f"Max ({temp_symbol})",
                    "std": f"Std Dev ({temp_symbol})",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    elif page == "When Will Cornell Tech Be Warm?":
        # --- WHEN WILL CORNELL TECH BE WARM? PAGE ---
        st.header("When Will Cornell Tech Be Warm?")
        st.subheader("Finding the First Year with Average Temperature Above 55°F")

        st.markdown(
            """
        In this analysis, we'll identify the first year when the annual average temperature at Cornell Tech 
        exceeds 55 degrees Fahrenheit - answering the question of when Cornell Tech will finally be "warm".
        """
        )

        yearly_avg = data.groupby("year")["Ftemp"].mean().reset_index()

        # Find the first year with average temperature > 55°F
        warm_threshold = 55.0  # in Fahrenheit
        warm_years = yearly_avg[yearly_avg["Ftemp"] > warm_threshold]

        if len(warm_years) > 0:
            first_warm_year = warm_years.iloc[0]["year"]
            first_warm_temp = warm_years.iloc[0]["Ftemp"]

            highlight_point = dict(
                x=[first_warm_year],
                y=[first_warm_temp],
                text=["First year > 55°F"],
                mode="markers+text",
                marker=dict(color="red", size=12),
                textposition="top center",
                textfont=dict(color="red", size=12),
                showlegend=False,
            )

            st.success(
                f"🌡️ The first year with an average temperature above 55°F is: **{int(first_warm_year)}** (Average: {first_warm_temp:.2f}°F)"
            )
        else:
            st.warning(
                "No years in the dataset have an average temperature above 55°F."
            )
            first_warm_year = None
            first_warm_temp = None
            highlight_point = None

        fig = px.line(
            yearly_avg,
            x="year",
            y="Ftemp",
            labels={"Ftemp": "Average Temperature (°F)", "year": "Year"},
            title="Yearly Average Temperatures at Cornell Tech Location",
            markers=True,
        )

        fig.add_hline(
            y=warm_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text="55°F Threshold",
            annotation_position="bottom right",
        )

        if highlight_point:
            fig.add_trace(
                go.Scatter(
                    x=highlight_point["x"],
                    y=highlight_point["y"],
                    mode=highlight_point["mode"],
                    marker=highlight_point["marker"],
                    text=highlight_point["text"],
                    textposition=highlight_point["textposition"],
                    textfont=highlight_point["textfont"],
                    showlegend=highlight_point["showlegend"],
                )
            )

        min_temp = max(30, yearly_avg["Ftemp"].min() - 5)
        max_temp = min(80, yearly_avg["Ftemp"].max() + 5)
        fig.update_yaxes(range=[min_temp, max_temp])

        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Detailed Temperature Analysis")
        st.markdown("### Years Approaching the 55°F Threshold")

        sorted_years = yearly_avg.sort_values("Ftemp", ascending=False).head(10)

        sorted_years["Distance from 55°F"] = sorted_years["Ftemp"] - warm_threshold
        display_df = sorted_years.rename(
            columns={"year": "Year", "Ftemp": "Avg. Temp (°F)"}
        )

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.subheader("Temperature Trend Analysis")

        yearly_avg["5yr_moving_avg"] = (
            yearly_avg["Ftemp"].rolling(window=5, center=True).mean()
        )

        x = yearly_avg["year"].values
        y = yearly_avg["Ftemp"].values
        mask = ~np.isnan(y)  # Remove NaN values
        x = x[mask]
        y = y[mask]

        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)

            trend_rate = z[0]

            current_year = datetime.now().year
            future_years = np.arange(min(available_years), current_year + 30)
            predicted_temps = p(future_years)

            # Find when we'll hit 55°F based on trend
            if trend_rate > 0:
                years_to_threshold = (warm_threshold - p(current_year)) / trend_rate
                threshold_year = current_year + years_to_threshold

                st.info(
                    f"Based on the observed warming trend of {trend_rate:.3f}°F per year, Cornell Tech will reach an average temperature of 55°F around the year **{int(threshold_year)}**."
                )
            else:
                st.warning(
                    f"Based on the current trend of {trend_rate:.3f}°F per year, Cornell Tech is not projected to reach an average temperature of 55°F in the foreseeable future."
                )

            fig_pred = go.Figure()

            fig_pred.add_trace(
                go.Scatter(
                    x=yearly_avg["year"],
                    y=yearly_avg["Ftemp"],
                    mode="markers",
                    name="Historical Data",
                    marker=dict(color="blue", size=8),
                )
            )

            fig_pred.add_trace(
                go.Scatter(
                    x=yearly_avg["year"],
                    y=yearly_avg["5yr_moving_avg"],
                    mode="lines",
                    name="5-Year Moving Average",
                    line=dict(color="green", width=2),
                )
            )

            fig_pred.add_trace(
                go.Scatter(
                    x=future_years,
                    y=predicted_temps,
                    mode="lines",
                    name=f"Trend Line ({trend_rate:.3f}°F/year)",
                    line=dict(color="red", dash="dash"),
                )
            )

            fig_pred.add_hline(
                y=warm_threshold,
                line_dash="dot",
                line_color="orange",
                annotation_text="55°F Threshold",
                annotation_position="bottom right",
            )

            fig_pred.update_layout(
                title="Temperature Trend and Projection",
                xaxis_title="Year",
                yaxis_title="Average Temperature (°F)",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
            )

            st.plotly_chart(fig_pred, use_container_width=True)

    elif page == "Creative Temperature Visualization":
        # --- CREATIVE TEMPERATURE VISUALIZATION PAGE ---
        st.header("Creative Temperature Visualization")
        st.subheader("Exploring Temperature Patterns Through Time")

        st.markdown(
            """
        This visualization explores the rhythms and patterns of temperature change at Cornell Tech 
        through creative and alternative perspectives. We'll look at temperature data in circular visualizations,
        polar plots, and other non-standard approaches to reveal insights about seasonal cycles, climate trends,
        and the relationship between time and temperature.
        """
        )

        st.sidebar.subheader("Creative Visualization Options")
        visual_type = st.sidebar.selectbox(
            "Visualization Type",
            ["Annual Temperature Cycle", "Temperature Heatmap", "Temperature Spiral"],
        )

        monthly_all_years = (
            data.groupby(["month", "month_name"])[temp_col]
            .agg(["mean", "min", "max", "std"])
            .reset_index()
        )
        month_order = {
            month: i
            for i, month in enumerate(
                [
                    "Jan",
                    "Feb",
                    "Mar",
                    "Apr",
                    "May",
                    "Jun",
                    "Jul",
                    "Aug",
                    "Sep",
                    "Oct",
                    "Nov",
                    "Dec",
                ]
            )
        }
        monthly_all_years["month_order"] = monthly_all_years["month_name"].map(
            month_order
        )
        monthly_all_years = monthly_all_years.sort_values("month_order")

        yearly_monthly_avg = (
            data.groupby(["year", "month"])[temp_col].mean().reset_index()
        )

        if visual_type == "Annual Temperature Cycle":
            st.markdown("### Annual Temperature Cycle (Polar Plot)")
            st.markdown(
                """
            This circular visualization displays the annual temperature cycle. Temperature values are mapped 
            radially, with each month positioned around the circle. This highlights the cyclical nature of 
            seasons and temperature patterns throughout the year.
            """
            )

            monthly_all_years["angle"] = monthly_all_years["month"] * (2 * np.pi / 12)

            fig = go.Figure()

            fig.add_trace(
                go.Scatterpolar(
                    r=monthly_all_years["mean"],
                    theta=monthly_all_years["month_name"],
                    mode="lines+markers",
                    marker=dict(
                        size=12, color=monthly_all_years["mean"], colorscale="thermal"
                    ),
                    line=dict(width=3),
                    fill="toself",
                    name=f"Average Temperature ({temp_symbol})",
                )
            )

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[
                            min(0, monthly_all_years["mean"].min() - 10),
                            monthly_all_years["mean"].max() + 10,
                        ],
                    )
                ),
                showlegend=False,
                title=f"Annual Temperature Cycle ({temp_symbol})",
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                """
            **Interpretation:** This polar visualization clearly shows how temperatures peak during summer months 
            (forming the widest part of the shape) and reach their minimum during winter (the narrowest part).
            The smooth transition between seasons is visible in the gradual expansion and contraction of the shape.
            """
            )

        elif visual_type == "Temperature Heatmap":
            st.markdown("### Temperature Heatmap")
            st.markdown(
                """
            This heatmap visualization shows temperature patterns across years (vertical axis) and months 
            (horizontal axis). Warmer colors represent higher temperatures, cooler colors represent lower temperatures.
            This helps identify seasonal patterns, yearly variations, and long-term temperature trends.
            """
            )

            heatmap_data = data.pivot_table(
                index="year", columns="month", values=temp_col, aggfunc="mean"
            )

            fig = px.imshow(
                heatmap_data,
                labels=dict(x="Month", y="Year", color=f"Temperature ({temp_symbol})"),
                x=[
                    "Jan",
                    "Feb",
                    "Mar",
                    "Apr",
                    "May",
                    "Jun",
                    "Jul",
                    "Aug",
                    "Sep",
                    "Oct",
                    "Nov",
                    "Dec",
                ],
                y=heatmap_data.index,
                color_continuous_scale="thermal",
                aspect="auto",
                title=f"Temperature Heatmap by Year and Month ({temp_symbol})",
            )

            fig.update_layout(
                height=800,
                xaxis_title="Month",
                yaxis_title="Year",
                coloraxis_colorbar=dict(title=f"Temp ({temp_symbol})"),
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                """
            **Interpretation:** This heatmap reveals how temperatures vary consistently by season (horizontally) and shows 
            any long-term warming or cooling trends (vertically). Areas of similar color grouped together indicate periods 
            of consistent temperature, while gradual shifts in color across years suggest climate change patterns.
            """
            )

        elif visual_type == "Temperature Spiral":
            st.markdown("### Temperature Spiral")
            st.markdown(
                """
            This spiral visualization shows how temperatures have changed over time in a novel way. 
            Each year forms one complete revolution around the spiral, with months positioned at regular intervals.
            The distance from the center represents the temperature value. This helps visualize both 
            seasonal patterns and long-term trends in a compact, visually striking way.
            """
            )

            spiral_data = yearly_monthly_avg.copy()

            spiral_data["angle"] = spiral_data["month"] * (2 * np.pi / 12)

            years_range = spiral_data["year"].max() - spiral_data["year"].min() + 1
            spiral_data["radius_increment"] = (
                spiral_data["year"] - spiral_data["year"].min()
            ) / years_range

            scaling_factor = 0.25
            spiral_data["r"] = 1 + spiral_data["radius_increment"] * scaling_factor

            spiral_data["x"] = spiral_data["r"] * np.cos(spiral_data["angle"])
            spiral_data["y"] = spiral_data["r"] * np.sin(spiral_data["angle"])

            fig = px.scatter(
                spiral_data,
                x="x",
                y="y",
                color=temp_col,
                color_continuous_scale="thermal",
                hover_data=["year", "month", temp_col],
                labels={temp_col: f"Temperature ({temp_symbol})"},
                title=f"Temperature Spiral ({temp_symbol})",
            )

            for year in spiral_data["year"].unique():
                year_data = spiral_data[spiral_data["year"] == year].sort_values(
                    "month"
                )

                fig.add_trace(
                    go.Scatter(
                        x=year_data["x"],
                        y=year_data["y"],
                        mode="lines",
                        line=dict(width=1, color="rgba(100,100,100,0.3)"),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

            outer_radius = spiral_data["r"].max() + 0.05
            month_labels = []
            month_positions_x = []
            month_positions_y = []

            for month in range(1, 13):
                angle = month * (2 * np.pi / 12)
                x = outer_radius * np.cos(angle)
                y = outer_radius * np.sin(angle)
                month_labels.append(calendar.month_abbr[month])
                month_positions_x.append(x)
                month_positions_y.append(y)

            fig.add_trace(
                go.Scatter(
                    x=month_positions_x,
                    y=month_positions_y,
                    mode="text",
                    text=month_labels,
                    textposition="middle center",
                    textfont=dict(size=10, color="black"),
                    showlegend=False,
                )
            )

            fig.update_layout(
                height=700,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    scaleanchor="x",
                    scaleratio=1,
                ),
                plot_bgcolor="rgba(240,240,240,0.9)",
            )

            st.plotly_chart(fig, use_container_width=True)

            # Add a small explanation
            st.markdown(
                """
            **Interpretation:** This spiral visualization shows how temperatures cycle through the seasons while 
            simultaneously revealing long-term trends. Each complete circle represents one year, with months 
            marked around the perimeter. As you move outward from the center, you progress through the years.
            
            Color changes along the spiral reveal temperature variations, with warmer colors representing higher temperatures.
            Look for patterns such as consistent color bands (stable climate) or gradual shifts in color as you move 
            outward (warming or cooling trends over time).
            """
            )
            
        # Temperature Correlation Analysis section - only show if data is loaded
        if data is not None:
            st.header("Temperature Correlation Analysis")
            st.markdown(
                """
            This section examines potential correlations between temperature and other factors.
            Let's look at how monthly temperature variations might correlate with seasonal patterns.
            """
            )

            if "monthly_all_years" not in locals():
                monthly_all_years = (
                    data.groupby(["month", "month_name"])[temp_col]
                    .agg(["mean", "min", "max", "std"])
                    .reset_index()
                )
                month_order = {
                    month: i
                    for i, month in enumerate(
                        [
                            "Jan",
                            "Feb",
                            "Mar",
                            "Apr",
                            "May",
                            "Jun",
                            "Jul",
                            "Aug",
                            "Sep",
                            "Oct",
                            "Nov",
                            "Dec",
                        ]
                    )
                }
                monthly_all_years["month_order"] = monthly_all_years["month_name"].map(
                    month_order
                )
                monthly_all_years = monthly_all_years.sort_values("month_order")

            # Create a simple solar radiation model (simulated data)
            months = range(1, 13)
            # Solar radiation follows roughly a sinusoidal pattern throughout the year
            solar_radiation = [
                500 + 300 * np.sin((month - 1.5) * np.pi / 6) for month in months
            ]

            # Get monthly temperature averages
            monthly_temp = monthly_all_years["mean"].values

            # Calculate correlation
            corr = np.corrcoef(solar_radiation, monthly_temp)[0, 1]

            # Create a scatter plot with regression line
            corr_fig = px.scatter(
                x=solar_radiation,
                y=monthly_temp,
                trendline="ols",
                labels={
                    "x": "Solar Radiation (simulated, W/m²)",
                    "y": f"Average Temperature ({temp_symbol})",
                },
                title=f"Correlation between Temperature and Solar Radiation (r = {corr:.2f})",
            )

            # Add month labels to points
            for i, month in enumerate(monthly_all_years["month_name"]):
                corr_fig.add_annotation(
                    x=solar_radiation[i],
                    y=monthly_temp[i],
                    text=month,
                    showarrow=False,
                    font=dict(size=10),
                )

            # Show the correlation plot
            st.plotly_chart(corr_fig, use_container_width=True)

# --- Footer ---
st.markdown("---")
st.markdown("Cornell Tech Weather Data Visualization | Created with Streamlit")
