"""
Digital Farm Biosecurity Portal — Final SIH Demo Prototype
Features:
- Fully fitted background image (poultry farm)
- Dark headings and tab titles for visibility
- Multilingual: English, Hindi, Telugu
- Risk Assessment + AI predictor
- Training Modules, Compliance Tracker, Dashboard & Alerts
"""

import streamlit as st
import pandas as pd
import numpy as np
import json, os, uuid
from datetime import datetime, date
import base64
from sklearn.ensemble import RandomForestClassifier

# ---------- Data storage ----------
DATA_FILE = "farm_data_final.json"

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

# ---------- Background image ----------
def set_background(image_file):
    if not os.path.exists(image_file):
        return
    with open(image_file, "rb") as file:
        data_img = file.read()
    b64 = base64.b64encode(data_img).decode()
    page_bg = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{b64}");
        background-size: cover;
        background-position: center;   /* ensures the whole image is centered */
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

# ---------- Dark headings & tabs ----------
st.markdown("""
<style>
h1, h2, h3, h4, h5, h6 { color: black !important; }
.css-1hynsf0 { color: black !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ---------- Multilingual support ----------
if "lang" not in st.session_state:
    st.session_state.lang = "en"

LANGS = {
    "english": {"title":"Digital Farm Biosecurity Portal","language":"Language","register":"Register Farm",
           "farmer_name":"Farmer / Owner name","farm_name":"Farm name","location":"Location","species":"Species",
           "choose_farm":"Select farm","risk_assessment":"Risk Assessment","start_assessment":"Start Assessment",
           "score":"Risk score","recommendations":"Recommendations","training":"Training Modules",
           "compliance":"Compliance Tracker","dashboard":"Dashboard & Alerts"},
    "hindi": {"title":"डिजिटल फार्म बायोसेक्युरिटी पोर्टल","language":"भाषा","register":"फार्म रजिस्टर करें",
           "farmer_name":"किसान / स्वामी का नाम","farm_name":"फार्म का नाम","location":"स्थान","species":"प्रजाति",
           "choose_farm":"फार्म चुनें","risk_assessment":"जोखिम मूल्यांकन","start_assessment":"मूल्यांकन शुरू करें",
           "score":"जोखिम स्कोर","recommendations":"अनुशंसाएँ","training":"प्रशिक्षण मॉड्यूल",
           "compliance":"अनुपालन ट्रैकर","dashboard":"डैशबोर्ड और अलर्ट"},
    "telugu": {"title":"డిజిటల్ ఫార్మ్ బయోసెక్యూరిటీ పోర్టల్","language":"భాష","register":"ఫార్మ్ నమోదు చేయండి",
           "farmer_name":"వ్యవసాయి / యజమాని పేరు","farm_name":"ఫార్మ్ పేరు","location":"స్థానం","species":"జాతి",
           "choose_farm":"ఫార్మ్ ఎంచుకోండి","risk_assessment":"రిస్క్ ఆంకలనం","start_assessment":"ఆంకలనం ప్రారంభించండి",
           "score":"రిస్క్ స్కోరు","recommendations":"సిఫార్సులు","training":"శిక్షణ మాడ్యూల్స్",
           "compliance":"అనుగుణత ట్రాకర్","dashboard":"డ్యాష్‌బోర్డ్ & అలర్ట్లు"}
}

# ---------- Header & language selection ----------
col1, col2 = st.columns([8,1])
with col1:
    st.title("Digital Farm Biosecurity Portal")
with col2:
    lang = st.selectbox("Language", ["English","हिन्दी","తెలుగు"])
    if lang.startswith("ह"): st.session_state.lang="hi"
    elif lang.startswith("త"): st.session_state.lang="te"
    else: st.session_state.lang="en"
t = LANGS[st.session_state.lang]

# ---------- Set background ----------
set_background("poultry.png")  # PNG landscape background

# ---------- AI Risk Model ----------
@st.cache_resource
def train_model(seed=42):
    rng = np.random.RandomState(seed)
    n = 1200
    X = rng.randint(0,3,(n,4))
    nearby = rng.binomial(1,0.05,(n,1))
    X = np.hstack([X, nearby])
    scores = X[:,0]+(2-X[:,1])+(2-X[:,2])+(2-X[:,3])+4*X[:,4]
    y = (scores>5).astype(int)
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=80, random_state=seed)
    clf.fit(X,y)
    return clf

model = train_model()

# ---------- Layout ----------
left, right = st.columns([2.5,7.5])
with left:
    st.header(t["register"])
    with st.form("reg"):
        owner = st.text_input(t["farmer_name"])
        farm = st.text_input(t["farm_name"])
        loc = st.text_input(t["location"])
        species = st.selectbox(t["species"], ["Poultry","Pig"])
        lat = st.number_input("Latitude", 0.0, format="%.6f")
        lon = st.number_input("Longitude", 0.0, format="%.6f")
        submit = st.form_submit_button(t["register"])
        if submit:
            fid = str(uuid.uuid4())
            new = {"id":fid,"owner":owner,"farm":farm,"location":loc,"species":species,
                   "lat":float(lat),"lon":float(lon),"created":str(datetime.utcnow())}
            data["farms"].append(new)
            with open(DATA_FILE,"w") as f: json.dump(data,f,indent=2)
            st.success(f"Registered: {farm}")
    farm_options = {f"{f['farm']} ({f['owner']})": f["id"] for f in data["farms"]}
    if farm_options:
        sel = st.selectbox(t["choose_farm"], list(farm_options.keys()))
        current_id = farm_options[sel]
        current_farm = next((x for x in data["farms"] if x["id"]==current_id), None)
        st.markdown(f"**{current_farm['farm']}** — {current_farm['owner']}")
    else:
        current_farm = None

# Tabs
tabs = right.tabs([t["risk_assessment"], t["training"], t["compliance"], t["dashboard"]])

# --- Risk Assessment ---
with tabs[0]:
    st.subheader(t["risk_assessment"])
    if current_farm:
        with st.form("assess"):
            v = st.selectbox("Visitors allowed?", ["No","Sometimes","Yes"])
            iso = st.selectbox("Sick animals isolated?", ["Yes","Sometimes","No"])
            feed = st.selectbox("Feed rodent-proof?", ["Yes","Sometimes","No"])
            foot = st.selectbox("Footbath/Vehicle disinfection?", ["Yes","Sometimes","No"])
            recent = st.selectbox("Nearby outbreak?", ["No","Yes"])
            sub = st.form_submit_button("Submit")
        if sub:
            map_v = {"No":0,"Sometimes":1,"Yes":2}
            map_iso = {"Yes":2,"Sometimes":1,"No":0}
            x = np.array([[map_v[v], map_iso[iso], map_iso[feed], map_iso[foot], 1 if recent=="Yes" else 0]])
            prob = model.predict_proba(x)[0][1]
            level = "High" if prob>0.6 else ("Moderate" if prob>0.3 else "Low")
            st.metric("Risk Score", f"{int(prob*100)}/100 ({level})")
            recs = ["Maintain protocols"] if prob<=0.3 else ["Strengthen disinfection","Call vet"] if prob<=0.6 else ["Immediate containment","Suspend visitors","Increase cleaning"]
            st.markdown("**Recommendations:**")
            for r in recs: st.write("- "+r)

# --- Training Modules ---
with tabs[1]:
    st.subheader(t["training"])
    st.write("📚 Best Practices & Training Modules")
    st.write("- Hygiene management")
    st.write("- Visitor and staff protocols")
    st.write("- Feed and water safety")
    st.write("- Vaccination and sick animal isolation")

# --- Compliance Tracker ---
with tabs[2]:
    st.subheader(t["compliance"])
    st.write("✅ Track your compliance tasks")
    st.write("- Daily cleaning logs")
    st.write("- Footbath/vehicle disinfection")
    st.write("- Record vaccinations")
    st.write("- Update outbreak logs")

# --- Dashboard & Alerts ---
with tabs[3]:
    st.subheader(t["dashboard"])
    st.write("📊 Farm Overview & Alerts")
    st.write("- Nearby outbreaks: None")
    st.write("- Risk status: Moderate")
    st.write("- Compliance: 3/5 tasks completed")

