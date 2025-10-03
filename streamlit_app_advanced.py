"""
Digital Farm Biosecurity Portal — Advanced Prototype (single-file)
Features:
- Farm registration + profile (Pig/Poultry)
- Multilingual UI (English / Hindi / Telugu)
- Risk assessment (questionnaire) + AI predictor (trained on synthetic data)
- Geospatial map (pydeck) showing farm + simulated outbreak hotspots
- Compliance tracker + training modules + gamification (points & badges)
- Immutable assessment hash (SHA256) for provenance demo
- Simple Farmer-Vet messaging (local)
- Downloadable certificate when safety badge achieved
Note: This prototype uses local JSON for demonstration. Replace with a DB in production.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json, os, uuid, hashlib
from datetime import datetime, date
import altair as alt
import pydeck as pdk
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import base64

DATA_FILE = "farm_data_advanced.json"

# ---------- Utilities ----------
def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return {"farms": [], "assessments": [], "compliance": [], "alerts": [], "messages": [], "users": []}

def save_data(d):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, default=str)

data = load_data()

def hash_record(record: dict):
    s = json.dumps(record, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(s).hexdigest()

def generate_certificate_text(farm):
    txt = f"""Certificate of Biosecurity Progress

Farm: {farm['farm']}
Owner: {farm['owner']}
Species: {farm['species']}
Location: {farm['location']}
Date: {date.today().isoformat()}

This is to certify that the farm has attained the 'Biosecure Starter' badge in the Digital Farm Biosecurity Portal prototype.
"""
    return txt

# ---------- Multilingual strings ----------
LANGS = {
    "en": {
        "title": "Digital Farm Biosecurity Portal (Advanced Prototype)",
        "language": "Language",
        "register": "Register Farm",
        "farmer_name": "Farmer / Owner name",
        "farm_name": "Farm name",
        "location": "Location (village/district)",
        "species": "Species",
        "choose_farm": "Select farm",
        "risk_assessment": "Risk Assessment",
        "start_assessment": "Start Assessment",
        "score": "Risk score",
        "recommendations": "Recommendations",
        "training": "Training Modules",
        "compliance": "Compliance Tracker",
        "add_task": "Add Compliance Task",
        "task_name": "Task name",
        "deadline": "Deadline",
        "mark_done": "Mark done",
        "dashboard": "Dashboard & Alerts",
        "simulate_alert": "Simulate Biosecurity Breach",
        "alerts": "Alerts",
        "map": "Farm Map & Nearby Outbreaks",
        "predictor": "AI Risk Predictor",
        "probability": "Predicted probability",
        "badges": "Badges & Points",
        "messages": "Farmer–Vet Messages",
        "send": "Send Message",
        "certificate": "Download Certificate"
    },
    "hi": {
        "title": "डिजिटल फार्म बायोसेक्युरिटी पोर्टल (प्रोटोटाइप - उन्नत)",
        "language": "भाषा",
        "register": "फार्म रजिस्टर करें",
        "farmer_name": "किसान / स्वामी का नाम",
        "farm_name": "फार्म का नाम",
        "location": "स्थान (गाँव/जिला)",
        "species": "प्रजाति",
        "choose_farm": "फार्म चुनें",
        "risk_assessment": "जोखिम मूल्यांकन",
        "start_assessment": "मूल्यांकन शुरू करें",
        "score": "जोखिम स्कोर",
        "recommendations": "अनुशंसाएँ",
        "training": "प्रशिक्षण मॉड्यूल",
        "compliance": "अनुपालन ट्रैकर",
        "add_task": "अनुपालन कार्य जोड़ें",
        "task_name": "कार्य का नाम",
        "deadline": "समय सीमा",
        "mark_done": "पूरा हुआ",
        "dashboard": "डैशबोर्ड और अलर्ट",
        "simulate_alert": "बायोसेक्युरिटी उल्लंघन सिमुलेट करें",
        "alerts": "अलर्ट",
        "map": "फार्म नक्शा और निकटवर्ती संक्रमण",
        "predictor": "AI जोखिम भविष्यवक्ता",
        "probability": "पूर्वानुमानित संभावना",
        "badges": "बैज और अंक",
        "messages": "किसान–पशु चिकित्सक संदेश",
        "send": "संदेश भेजें",
        "certificate": "प्रमाणपत्र डाउनलोड करें"
    },
    "te": {
        "title": "డిజిటల్ ఫార్మ్ బయోసెక్యూరిటీ పోర్టల్ (ప్రోటోటైప్)",
        "language": "భాష",
        "register": "ఫార్మ్ నమోదు చేయండి",
        "farmer_name": "వ్యవసాయి / యజమాని పేరు",
        "farm_name": "ఫార్మ్ పేరు",
        "location": "స్థానం (గ్రామం/జిల్లా)",
        "species": "జాతి",
        "choose_farm": "ఫార్మ్ ఎంచుకోండి",
        "risk_assessment": "రిస్క్ ఆంకలనం",
        "start_assessment": "ఆంకలనం ప్రారంభించండి",
        "score": "రిస్క్ స్కోరు",
        "recommendations": "సిఫార్సులు",
        "training": "శిక్షణ మాడ్యూల్స్",
        "compliance": "అనుగుణత ట్రాకర్",
        "add_task": "అనుగుణత కార్యాన్ని జోడించండి",
        "task_name": "కార్య పేరు",
        "deadline": "గడువు తేదీ",
        "mark_done": "పూర్తయింది",
        "dashboard": "డ్యాష్‌బోర్డ్ & అలర్ట్లు",
        "simulate_alert": "బ్రచ్‌ని అనుకరించండి",
        "alerts": "అలర్ట్ల",
        "map": "ఫార్మ్ నకశా మరియు పరిసర వ్యాధి కేంద్రాలు",
        "predictor": "AI రిస్క్ ప్రిడిక్టర్",
        "probability": "గుర్తించిన సంభావ్యత",
        "badges": "బాడ్జీలు & పాయింట్లు",
        "messages": "వ్యవసాయి–వెటర్ సందేశాలు",
        "send": "సందేశం పంపండి",
        "certificate": "సర్టిఫికెట్ డౌన్లోడ్ చేయండి"
    }
}

# ---------- Train a simple AI model on synthetic data ----------
@st.cache_resource
def train_synthetic_model(random_seed=42):
    # features: visitor_policy(0-2), isolation(0-2), feed_proof(0-2), footbath(0-2), nearby_outbreak(0/1)
    rng = np.random.RandomState(random_seed)
    n = 1200
    X = rng.randint(0,3,size=(n,4))
    nearby = rng.binomial(1, 0.05, size=(n,1))
    X = np.hstack([X, nearby])
    # create label: high risk if many bad practices or nearby outbreak
    scores = X[:,0] + (2 - X[:,1]) + (2 - X[:,2]) + (2 - X[:,3]) + 4*X[:,4]
    y = (scores > 5).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)
    clf = RandomForestClassifier(n_estimators=80, random_state=random_seed)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return clf, acc

model, model_acc = train_synthetic_model()

# ---------- App layout ----------
st.set_page_config(page_title="Farm Biosecurity (Advanced)", layout="wide")
if "lang" not in st.session_state:
    st.session_state.lang = "en"
t = LANGS[st.session_state.lang]

# Top bar
col1, col2 = st.columns([8,1])
with col1:
    st.title(t["title"])
with col2:
    lang = st.selectbox(t["language"], options=["English","हिन्दी","తెలుగు"])
    if lang.startswith("ह"):
        st.session_state.lang = "hi"
    elif lang.startswith("త"):
        st.session_state.lang = "te"
    else:
        st.session_state.lang = "en"
    t = LANGS[st.session_state.lang]

left, right = st.columns([2.5,7.5])

# ---------- Left: Registration & selection ----------
with left:
    st.header(t["register"])
    with st.form("reg"):
        owner = st.text_input(t["farmer_name"])
        farm = st.text_input(t["farm_name"])
        loc = st.text_input(t["location"])
        species = st.selectbox(t["species"], ["Poultry","Pig"])
        lat = st.number_input("Latitude (optional)", value=0.0, format="%.6f")
        lon = st.number_input("Longitude (optional)", value=0.0, format="%.6f")
        submit = st.form_submit_button(t["register"])
        if submit:
            fid = str(uuid.uuid4())
            new = {"id": fid, "owner": owner, "farm": farm, "location": loc, "species": species,
                   "lat": float(lat), "lon": float(lon), "created": str(datetime.utcnow()), "points": 0, "badges": []}
            data["farms"].append(new)
            save_data(data)
            st.success(f"Registered: {farm}")
    farm_options = {f"{f['farm']} ({f['owner']})": f["id"] for f in data["farms"]}
    if farm_options:
        sel = st.selectbox(t["choose_farm"], options=list(farm_options.keys()))
        current_id = farm_options[sel]
        current_farm = next((x for x in data["farms"] if x["id"]==current_id), None)
        st.markdown(f"**{current_farm['farm']}** — {current_farm['owner']}")
    else:
        current_farm = None
        current_id = None
        st.info("No farms yet")

# ---------- Right: Tabs ----------
tabs = right.tabs([t["risk_assessment"], t["training"], t["compliance"], t["dashboard"]])

# ---- Risk assessment tab ----
with tabs[0]:
    st.subheader(t["risk_assessment"])
    if not current_farm:
        st.info("Register/select a farm first.")
    else:
        st.markdown(f"**Farm:** {current_farm['farm']} | **Species:** {current_farm['species']} | **Location:** {current_farm['location']}")
        st.markdown("---")
        with st.form("assess"):
            v = st.selectbox("Visitors allowed without disinfection?", ["No","Sometimes","Yes"])
            iso = st.selectbox("Are sick animals isolated promptly?", ["Yes","Sometimes","No"])
            feed = st.selectbox("Is feed/water rodent/insect-proof?", ["Yes","Sometimes","No"])
            foot = st.selectbox("Is there footbath/vehicle disinfection at entry?", ["Yes","Sometimes","No"])
            recent = st.selectbox("Recent nearby outbreak (within 50 km)?", ["No","Yes"])
            sub = st.form_submit_button(t["start_assessment"])
        if sub:
            # map answers to numeric features for model: visitor_policy: No=0, Sometimes=1, Yes=2
            map_v = {"No":0,"Sometimes":1,"Yes":2}
            map_iso = {"Yes":2,"Sometimes":1,"No":0}
            x = np.array([[map_v[v], map_iso[iso], map_iso[feed], map_iso[foot], 1 if recent=="Yes" else 0]])
            prob = model.predict_proba(x)[0][1]  # probability of high-risk
            percent = int(prob*100)
            level = "High" if prob>0.6 else ("Moderate" if prob>0.3 else "Low")
            st.metric(t["score"], f"{percent} / 100 ({level})")
            # recommendations
            recs = []
            if prob > 0.6:
                recs = [
                    "Immediate containment: isolate stock and call vet.",
                    "Suspend incoming animals and visitors; disinfect entry.",
                    "Increase cleaning & PPE usage."
                ]
            elif prob > 0.3:
                recs = [
                    "Strengthen disinfection checks and staff training.",
                    "Review feed storage and rodent control."
                ]
            else:
                recs = ["Maintain current biosecurity protocols; continue monitoring."]
            st.markdown("**" + t["recommendations"] + "**")
            for r in recs:
                st.write("- " + r)
            # save assessment with hash & update points/badges
            assess = {
                "id": str(uuid.uuid4()),
                "farm_id": current_id,
                "timestamp": str(datetime.utcnow()),
                "answers": {"visitors":v,"isolation":iso,"feed_proof":feed,"footbath":foot,"recent_outbreak":recent},
                "probability": float(prob),
                "level": level
            }
            assess["hash"] = hash_record(assess)
            data["assessments"].append(assess)
            # award points (example)
            pts = int((1-prob)*10) + 2  # safer farms get more points
            # find farm record and update
            for f in data["farms"]:
                if f["id"] == current_id:
                    f["points"] = f.get("points",0) + pts
                    # assign badges by thresholds
                    if f["points"] >= 25 and "Biosecure Bronze" not in f["badges"]:
                        f["badges"].append("Biosecure Bronze")
                    if f["points"] >= 50 and "Biosecure Silver" not in f["badges"]:
                        f["badges"].append("Biosecure Silver")
                    if f["points"] >= 80 and "Biosecure Gold" not in f["badges"]:
                        f["badges"].append("Biosecure Gold")
            save_data(data)
            st.success("Assessment saved. Hash: " + assess["hash"][:12] + "...")
            # show AI model accuracy (for demo transparency)
            st.caption(f"(Demo AI model accuracy on synthetic data: {model_acc:.2f})")

# ---- Training tab ----
with tabs[1]:
    st.subheader(t["training"])
    st.markdown("Complete short modules to earn points & badges.")
    module = st.selectbox("Choose module", ["Biosecurity Basics","Cleaning & Disinfection","Rodent Control","Emergency Response"])
    if module == "Biosecurity Basics":
        st.markdown("""
- Why biosecurity matters  
- Visitor control and entry protocols  
- Daily checklist for staff
""")
        if st.button("Complete module & Earn points"):
            if current_farm:
                for f in data["farms"]:
                    if f["id"] == current_id:
                        f["points"] = f.get("points",0) + 8
                        if "Biosecure Bronze" not in f["badges"] and f["points"] >= 25:
                            f["badges"].append("Biosecure Bronze")
                save_data(data)
                st.success("Module completed. Points awarded.")
            else:
                st.info("Select a farm first.")
    elif module == "Cleaning & Disinfection":
        st.markdown("- Disinfectants, contact times, PPE, cleaning order")
    elif module == "Rodent Control":
        st.markdown("- Traps, baiting schedules, and feed hygiene")
    else:
        st.markdown("- Emergency steps when suspecting infection; vet contact")

    # certificate download if bronze or above
    if current_farm:
        f = current_farm
        st.markdown("---")
        st.markdown(t["badges"] + ": " + ", ".join(f.get("badges",[])))
        if "Biosecure Bronze" in f.get("badges",[]):
            cert_text = generate_certificate_text(f)
            b64 = base64.b64encode(cert_text.encode()).decode()
            href = f'<a download="certificate_{f["farm"]}.txt" href="data:text/plain;base64,{b64}">📄 {t["certificate"]}</a>'
            st.markdown(href, unsafe_allow_html=True)

# ---- Compliance tab ----
with tabs[2]:
    st.subheader(t["compliance"])
    if not current_farm:
        st.info("Select a farm to manage compliance.")
    else:
        with st.form("add_task"):
            task = st.text_input(t["task_name"])
            dl = st.date_input(t["deadline"])
            add = st.form_submit_button(t["add_task"])
            if add:
                item = {"id": str(uuid.uuid4()), "farm_id": current_id, "task": task, "deadline": str(dl), "done": False}
                data["compliance"].append(item)
                # small points for adding tasks (encourage record keeping)
                for f in data["farms"]:
                    if f["id"] == current_id:
                        f["points"] = f.get("points",0) + 1
                save_data(data)
                st.success("Task added.")

        df_tasks = pd.DataFrame([t for t in data["compliance"] if t["farm_id"]==current_id])
        if not df_tasks.empty:
            st.table(df_tasks[["task","deadline","done"]])
            to_mark = st.selectbox(t["mark_done"], options=["--"] + list(df_tasks["task"]))
            if to_mark != "--":
                row = df_tasks[df_tasks["task"]==to_mark].iloc[0]
                idx = next((i for i,x in enumerate(data["compliance"]) if x["id"]==row["id"]), None)
                if idx is not None:
                    data["compliance"][idx]["done"] = True
                    # reward points for completion
                    for f in data["farms"]:
                        if f["id"] == current_id:
                            f["points"] = f.get("points",0) + 5
                    save_data(data)
                    st.experimental_rerun()
        else:
            st.info("No compliance tasks for this farm.")

# ---- Dashboard tab ----
with tabs[3]:
    st.subheader(t["dashboard"])
    colA, colB = st.columns([3,1])
    with colA:
        st.markdown("**Overview**")
        st.metric("Farms registered", len(data["farms"]))
        assessments_df = pd.DataFrame(data["assessments"])
        if not assessments_df.empty:
            st.markdown("Recent assessments")
            st.dataframe(assessments_df[["farm_id","timestamp","probability","level"]].sort_values("timestamp", ascending=False).head(8))
            # probability histogram
            chart_df = assessments_df.copy()
            chart_df["prob"] = chart_df["probability"].astype(float)
            hist = alt.Chart(chart_df).mark_bar().encode(
                alt.X("prob:Q", bin=alt.Bin(maxbins=10), title="Risk probability"),
                y='count()'
            ).properties(height=220)
            st.altair_chart(hist, use_container_width=True)
    with colB:
        st.markdown("**" + t["alerts"] + "**")
        if st.button(t["simulate_alert"]):
            a = {"id": str(uuid.uuid4()), "timestamp": str(datetime.utcnow()), "farm_id": current_id,
                 "title": "Simulated: Nearby outbreak reported", "severity": "High",
                 "message": "Simulated outbreak within 10 km. Increase monitoring."}
            data["alerts"].append(a)
            save_data(data)
            st.success("Alert created.")
        alerts_df = pd.DataFrame(data["alerts"])
        if not alerts_df.empty:
            for _,r in alerts_df.sort_values("timestamp",ascending=False).head(5).iterrows():
                st.warning(f"[{r['severity']}] {r['title']} — {r['timestamp']}")
                st.write(r["message"])
        else:
            st.info("No alerts")

    st.markdown("---")
    st.subheader(t["map"])
    # Map: show current farm and random nearby outbreak points
    if current_farm and current_farm.get("lat",0) and current_farm.get("lon",0):
        latc = current_farm["lat"]
        lonc = current_farm["lon"]
        st.markdown(f"Location: {latc:.6f}, {lonc:.6f}")
        # create random outbreak points within ~0.2 degrees
        rng = np.random.RandomState(int(uuid.UUID(current_id).int % 100000))
        n_hot = rng.randint(3,8)
        lats = latc + (rng.rand(n_hot)-0.5)*0.4
        lons = lonc + (rng.rand(n_hot)-0.5)*0.4
        df_map = pd.DataFrame({"lat": np.concatenate(([latc], lats)), "lon": np.concatenate(([lonc], lons)), "type": ["Farm"] + ["Outbreak"]*n_hot})
        # pydeck
        layer = pdk.Layer(
            "ScatterplotLayer",
            df_map,
            get_position='[lon, lat]',
            get_fill_color='[200, 30, 0, 160] if type=="Outbreak" else [30, 160, 200, 200]',
            get_radius=200,
            pickable=True
        )
        view = pdk.ViewState(latitude=latc, longitude=lonc, zoom=9, pitch=0)
        r = pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"text":"{type}"})
        st.pydeck_chart(r)
    else:
        st.info("Set farm latitude & longitude when registering to view map visualization.")

    # messages (farmer-vet)
    st.markdown("---")
    st.subheader(t["messages"])
    if current_farm:
        msgs = [m for m in data["messages"] if m["farm_id"]==current_id]
        for m in sorted(msgs, key=lambda x: x["timestamp"]):
            who = m["from"]
            st.write(f"**{who}** — {m['timestamp']}")
            st.write(m["text"])
        with st.form("msgform"):
            name = st.text_input("Your name")
            text = st.text_area("Message / Question")
            send = st.form_submit_button(t["send"])
            if send:
                msg = {"id": str(uuid.uuid4()), "farm_id": current_id, "from": name, "text": text, "timestamp": str(datetime.utcnow())}
                data["messages"].append(msg)
                save_data(data)
                st.success("Message sent (demo). A real system would notify vets/extension workers.")
    else:
        st.info("Select a farm to view messages.")

# ---------- Footer notes ----------
st.sidebar.markdown("### Notes & Next steps")
st.sidebar.write("""
- This prototype uses a local JSON store (`farm_data_advanced.json`). Use Supabase/Postgres/MongoDB for production.
- Replace the synthetic AI with a model trained on real epidemiological + farm-practice datasets.
- Integrate SMS/WhatsApp alerts (Twilio / local gateway), and a geofenced outbreak feed (OIE/FAO/Government APIs).
- For offline operation, build a PWA or Android app with local SQLite sync.
- For tamper-proof records, integrate a lightweight blockchain or append-only store; here we show SHA-256 hashes as a demo.
""")
