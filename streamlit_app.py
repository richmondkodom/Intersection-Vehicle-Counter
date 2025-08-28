# ===================== CUSTOM COLOR PALETTES =====================
COLOR_MAP = {
    "Left → Right": "#4CAF50",   # Green
    "Right → Left": "#2196F3",   # Blue
    "Up → Down": "#FF9800",      # Orange
    "Down → Up": "#9C27B0",      # Purple
}

CLASS_COLORS = px.colors.qualitative.Set2  # nice preset for vehicle classes

# Vehicles by Class Chart
if class_totals:
    df_classes = pd.DataFrame(list(class_totals.items()), columns=["Class", "Count"])
    fig_classes = px.bar(
        df_classes,
        x="Class",
        y="Count",
        color="Class",
        color_discrete_sequence=CLASS_COLORS,
        title="🚘 Vehicles by Class",
    )
    st.plotly_chart(fig_classes, use_container_width=True)

# Direction Share Chart
df_directions = pd.DataFrame([
    ["Left → Right", direction_counts["left_to_right"]],
    ["Right → Left", direction_counts["right_to_left"]],
    ["Up → Down", direction_counts["up_to_down"]],
    ["Down → Up", direction_counts["down_to_up"]],
], columns=["Direction", "Count"])

fig_directions = px.pie(
    df_directions,
    names="Direction",
    values="Count",
    title="🧭 Direction Share",
    color="Direction",
    color_discrete_map=COLOR_MAP
)
st.plotly_chart(fig_directions, use_container_width=True)

# Total Vehicles Over Time (line chart)
if events:
    df_events = pd.DataFrame(events, columns=["track_id","direction","class","frame","timestamp"])
    df_events["cum_total"] = range(1, len(df_events)+1)
    fig_time = px.line(
        df_events,
        x="timestamp",
        y="cum_total",
        title="📈 Total Vehicles Over Time",
        markers=True,
        line_shape="spline"
    )
    st.plotly_chart(fig_time, use_container_width=True)
