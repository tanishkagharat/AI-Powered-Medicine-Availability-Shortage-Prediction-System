import streamlit as st
import pandas as pd
import base64
import datetime
import time
import easyocr
import numpy as np
from PIL import Image
from rapidfuzz import fuzz

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(layout="wide")

# -------------------------
# BACKGROUND
# -------------------------
def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()
img = get_base64("notebook/background image.png")
# -------------------------
# DATA
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("notebook/medicine dataset.csv")
    df.columns = df.columns.str.strip()
    return df  

df = load_data()

# -------------------------
# TITLE
# -------------------------
st.title("AI-Powered Medicine Availability & Shortage Prediction System")

# -------------------------
# TABS
# -------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Predictor",
    "🚨 Emergency",
    "📸 Prescription",
    "⏰ Reminder",
    "🚚 Delivery"
])

# =====================================================
# 🔮 TAB 1
# =====================================================
with tab1:

    st.header("🔮 Medicine Predictor")

    col1, col2 = st.columns(2)

    with col1:
        med1 = st.selectbox("💊 Medicine", df['Medicine_Name'].unique())

    with col2:
        area1 = st.selectbox("📍 Area", df['Area'].unique())

    if st.button("Check Availability"):

        result = df[
            (df['Medicine_Name'] == med1) &
            (df['Area'] == area1)
        ]

        if result.empty:
            st.error("❌ Not available")
        else:
            row = result.iloc[0]
            status = row['Availability']

            if status == "Available":
                color = "#00c9a7"; emoji = "🟢"
            elif status == "Low Stock":
                color = "#ffc107"; emoji = "🟡"
            else:
                color = "#ff4d4d"; emoji = "🔴"

            st.markdown(f"""
            <div style="background:rgba(0,0,0,0.7);padding:20px;border-radius:15px;border-left:6px solid {color}">
                <h2 style="color:{color};">{emoji} {status}</h2>
                <p>🏥 {row['Pharmacy_Name']}</p>
                <p>📍 {row['Area']}</p>
                <p>📞 {row['Contact_Number']}</p>
                <p>🚚 Delivery: {row['Home_Delivery']}</p>
                <p>⏱️ {row['Delivery_Time_Minutes']} mins</p>
            </div>
            """, unsafe_allow_html=True)

# =====================================================
# 🚨 TAB 2
# =====================================================
with tab2:

    st.header("🚨 Emergency Finder")

    col1, col2 = st.columns(2)

    with col1:
        med2 = st.selectbox("💊 Medicine", df['Medicine_Name'].unique(), key="em")

    with col2:
        area2 = st.selectbox("📍 Your Area", df['Area'].unique(), key="em_area")

    if st.button("Find Fastest"):

        available = df[
            (df['Medicine_Name'] == med2) &
            (df['Availability'] == "Available")
        ].copy()

        if available.empty:
            st.error("❌ Not available anywhere")
        else:
            available["Time"] = available["Delivery_Time_Minutes"]
            available = available.sort_values(by="Time")

            for _, r in available.head(3).iterrows():

                pharmacy = r['Pharmacy_Name']
                area = r['Area']

                origin = area2.replace(" ", "+")
                destination = f"{pharmacy} {area}".replace(" ", "+")

                maps_url = f"https://www.google.com/maps/dir/?api=1&origin={origin}&destination={destination}"

                st.markdown(f"""
                <a href="{maps_url}" target="_blank">
                    <div style="background:rgba(0,0,0,0.7);padding:18px;border-radius:12px;margin-bottom:10px;border-left:5px solid #00c9a7">
                        <h4 style="color:white;">🏥 {pharmacy} ({area})</h4>
                        <p style="color:white;">⏱️ {r['Time']} mins</p>
                        <p style="color:#00c9a7;">👉 Directions</p>
                    </div>
                </a>
                """, unsafe_allow_html=True)

with tab3:

    st.header("📸 Upload Prescription")

    file = st.file_uploader(
        "Upload Prescription Image",
        type=["png", "jpg", "jpeg"],
        key="upload"
    )

    if file:
        st.image(file, width=300)
        st.success("Uploaded successfully")

        image = Image.open(file)

        reader = easyocr.Reader(["en"])

        with st.spinner("Reading prescription..."):
            results = reader.readtext(
                np.array(image),
                detail=0
            )

        st.subheader("💊 Medicines Detected")

        medicine_db = df["Medicine_Name"].dropna().unique()

        detected = []

        for text in results:
            best_match = None
            best_score = 0

            for med in medicine_db:
                score = fuzz.ratio(str(med).lower(), text.lower())

                if score > best_score:
                    best_score = score
                    best_match = med

            if best_score >= 50:
                detected.append(best_match)

        detected = list(dict.fromkeys(detected))

        if len(detected) == 0:
            st.warning("No medicine detected clearly. Try uploading a clearer image.")

        else:
            if "selected_med" not in st.session_state:
                st.session_state.selected_med = None

            for i, med in enumerate(detected):
                if st.button(f"✔ {med}", key=f"med_{i}_{med}"):
                    st.session_state.selected_med = med

            if st.session_state.selected_med:

                selected = st.session_state.selected_med

                st.subheader(f"🏥 Availability for {selected}")

                matches = df[df["Medicine_Name"] == selected]

                if matches.empty:
                    st.error("No pharmacy found in dataset.")

                else:
                    available = matches[matches["Availability"] == "Available"]
                    low_stock = matches[matches["Availability"] == "Low Stock"]
                    out_stock = matches[matches["Availability"] == "Out of Stock"]

                    if not available.empty:
                        st.success(f"✅ {selected} is Available")

                        st.write("### 📍 Available at:")

                        for _, shop in available.iterrows():
                            st.write(
                                f"✅ **{shop['Pharmacy_Name']}** - {shop['Area']} "
                                f"| 📞 {shop['Contact_Number']} "
                                f"| 🚚 Delivery: {shop['Home_Delivery']}"
                            )

                    elif not low_stock.empty:
                        st.warning(f"⚠️ {selected} is Low Stock")

                        st.write("### 📍 Low stock at:")

                        for _, shop in low_stock.iterrows():
                            st.write(
                                f"⚠️ **{shop['Pharmacy_Name']}** - {shop['Area']} "
                                f"| 📞 {shop['Contact_Number']}"
                            )

                    else:
                        st.error(f"❌ {selected} is Out of Stock")
# =====================================================
# ⏰ TAB 4 (FINAL HYBRID CLOCK + AM/PM)
# =====================================================
with tab4:

    st.header("⏰ Medicine Reminder")

    # ✅ SESSION INIT (IMPORTANT)
    if "reminders" not in st.session_state:
        st.session_state["reminders"] = []

    # -------------------------
    # INPUT SECTION
    # -------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        med_name = st.selectbox(
            "💊 Medicine",
            df['Medicine_Name'].unique(),
            key="rem_med"
        )

    with col2:
        clock_time = st.time_input(
            "⏰ Select Time",
            value=datetime.datetime.now().time(),
            key="rem_clock"
        )

    with col3:
        am_pm = st.selectbox(
            "AM / PM",
            ["AM", "PM"],
            key="rem_ampm"
        )

    # -------------------------
    # ADD REMINDER
    # -------------------------
    if st.button("➕ Add Reminder"):

        hour = clock_time.hour
        minute = clock_time.minute

        # 🔥 AM/PM FIX
        if am_pm == "PM" and hour < 12:
            hour += 12
        elif am_pm == "AM" and hour >= 12:
            hour -= 12

        final_time = datetime.time(hour, minute)

        now = datetime.datetime.now()

        reminder_datetime = datetime.datetime.combine(
            now.date(),
            final_time
        )

        # 🔥 if time passed → next day
        if reminder_datetime <= now:
            reminder_datetime += datetime.timedelta(days=1)

        st.session_state["reminders"].append({
            "medicine": med_name,
            "time": reminder_datetime,
            "taken": False
        })

        st.success(
            f"✅ Reminder set for {med_name} at {reminder_datetime.strftime('%I:%M %p')}"
        )

    st.divider()

    # -------------------------
    # DISPLAY REMINDERS
    # -------------------------
    st.subheader("📋 Your Reminders")

    current_time = datetime.datetime.now()

    if len(st.session_state["reminders"]) == 0:
        st.info("No reminders set")

    else:
        for i in range(len(st.session_state["reminders"])):

            r = st.session_state["reminders"][i]

            med = r["medicine"]
            time_val = r["time"]
            taken = r["taken"]

            time_diff = (time_val - current_time).total_seconds()

            # 🔥 STATUS LOGIC
            if not taken:
                if time_diff <= 0:
                    status = "🔔 DUE NOW"
                    color = "#ff4d4d"
                elif time_diff <= 300:
                    status = "⏳ Due Soon"
                    color = "#ffc107"
                else:
                    status = "⏳ Upcoming"
                    color = "#00c9a7"
            else:
                status = "✅ Taken"
                color = "#28a745"

            # -------------------------
            # CARD UI
            # -------------------------
            st.markdown(f"""
            <div style="background:rgba(0,0,0,0.7);padding:15px;border-radius:10px;margin-bottom:10px;border-left:5px solid {color}">
                <h4 style="color:white;">💊 {med}</h4>
                <p style="color:white;">⏰ {time_val.strftime('%I:%M %p')}</p>
                <p style="color:{color};"><b>{status}</b></p>
            </div>
            """, unsafe_allow_html=True)

            # -------------------------
            # BUTTONS
            # -------------------------
            colA, colB = st.columns(2)

            with colA:
                if not taken:
                    if st.button("✔ Mark Taken", key=f"taken_{i}"):
                        st.session_state["reminders"][i]["taken"] = True
                        st.rerun()

            with colB:
                if st.button("🗑 Delete", key=f"delete_{i}"):
                    st.session_state["reminders"].pop(i)
                    st.rerun()
# =====================================================
# 🚚 TAB 5 (ADVANCED + ORDER)
# =====================================================
with tab5:

    st.header("🚚 Smart Delivery Finder")

    col1, col2 = st.columns(2)

    with col1:
        med3 = st.selectbox("💊 Medicine", df['Medicine_Name'].unique(), key="del")

    with col2:
        area3 = st.selectbox("📍 Your Area", df['Area'].unique(), key="del_area")

    if st.button("Find Delivery Options"):

        delivery_df = df[
            (df['Medicine_Name'] == med3) &
            (df['Home_Delivery'] == "Yes")
        ].copy()

        if delivery_df.empty:
            st.error("❌ No delivery available")
        else:

            def estimate_time(row):
                return row['Delivery_Time_Minutes'] if row['Area'] == area3 else row['Delivery_Time_Minutes'] + 10

            delivery_df["Time"] = delivery_df.apply(estimate_time, axis=1)
            delivery_df["Score"] = 100 - delivery_df["Time"]
            delivery_df = delivery_df.sort_values(by="Score", ascending=False)

            for i, row in delivery_df.head(5).iterrows():

                if i == delivery_df.index[0]:
                    badge = "🏆 Best Choice"; color = "#00c9a7"
                elif row["Time"] <= 15:
                    badge = "⚡ Fast"; color = "#28a745"
                else:
                    badge = "📦 Available"; color = "#ffc107"

                st.markdown(f"""
                <div style="background:rgba(0,0,0,0.75);padding:18px;border-radius:12px;margin-bottom:10px;border-left:5px solid {color}">
                    <h4 style="color:white;">🏥 {row['Pharmacy_Name']} ({row['Area']})</h4>
                    <p style="color:white;">⏱️ {row['Time']} mins</p>
                    <p style="color:{color};"><b>{badge}</b></p>
                </div>
                """, unsafe_allow_html=True)

                colA, colB = st.columns(2)

                with colA:
                    location = f"{row['Pharmacy_Name']} {row['Area']}".replace(" ", "+")
                    st.link_button("📍 View Location", f"https://www.google.com/maps/search/?api=1&query={location}")

                with colB:
                    if st.button(f"🛒 Order Now - {row['Pharmacy_Name']}", key=f"order_{i}"):
                        st.success(f"✅ Order placed from {row['Pharmacy_Name']}")
                        st.info("💊 Your medicine will be delivered soon!")

# =====================================================
# 🎨 UI
# =====================================================
st.markdown(f"""
<style>

.stApp {{
    background: url("data:image/png;base64,{img}") no-repeat center;
    background-size: cover;
}}

.block-container {{
    background: rgba(0,0,0,0.6);
    padding: 20px;
    border-radius: 15px;
}}

.stButton > button {{
    background: linear-gradient(145deg, #00c9a7, #009e83);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
    box-shadow: 5px 5px 12px rgba(0,0,0,0.6),
                -3px -3px 8px rgba(255,255,255,0.1);
}}

h1,h2,h3,p,label {{
    color:white;
}}

</style>
""", unsafe_allow_html=True)
st.markdown(f"""
<style>

/* REMOVE TOP SPACE */
html, body, [data-testid="stAppViewContainer"] {{
    margin: 0;
    padding: 0;
}}

/* REMOVE HEADER GAP */
header {{
    visibility: hidden;
    height: 0px;
}}

/* REMOVE EXTRA TOP PADDING */
.block-container {{
    padding-top: 0rem !important;
}}

/* FULL BACKGROUND FIX */
.stApp {{
    background: url("data:image/png;base64,{img}") no-repeat center center fixed;
    background-size: cover;
}}

</style>
""", unsafe_allow_html=True)
st.markdown(f"""
<style>

/* REMOVE LINK UNDERLINE */
a {{
    text-decoration: none !important;
    color: white !important;
}}

/* OPTIONAL: hover pe bhi underline na aaye */
a:hover {{
    text-decoration: none !important;
    color: #00c9a7 !important;
}}

</style>
""", unsafe_allow_html=True)