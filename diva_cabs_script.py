# %%
%pip install pandas
# %%
%pip -q install pandas openpyxl

# %%
import pandas as pd

# If you used the file picker in Colab:
path = "C:\\Users\\Keerthana AK\\Desktop\\Internship\\Diva Cabs.xlsx"  # or the file path you uploaded
df_raw = pd.read_excel(path, sheet_name="Sheet1")

# %%
df_raw.info()
# %%
df_raw.columns
# %%

import pandas as pd, numpy as np, re
from datetime import datetime, timedelta, time as dtime

INPUT_PATH = path   # <- change if needed
TARGET_DATE = "2025-08-26"              # <- change to any date present in your sheet

def norm_city_name(s: str):
    if s is None or (isinstance(s, float) and np.isnan(s)): return None
    x = str(s).strip()
    if not x: return None
    x = re.sub(r"\s+", " ", x).title()
    x = x.replace("Bengaluru","Bangalore").replace("Bangaluru","Bangalore")
    return x

def city_cluster(x: str):
    if x is None: return None
    s = str(x).strip().lower()
    if s.startswith("ranchi"):
        return "Ranchi"
    if s.startswith("tata") or s.startswith("jamshed") or s.startswith("jamesh"):
        return "Tata"
    if s in ("dhanbad","bokaro","dhanbad/bokaro"):
        return "Dhanbad/Bokaro"
    if s == "kolkata":
        return "Kolkata"
    return s.title()

def parse_dt(date_val, time_val):
    d = pd.to_datetime(date_val, errors='coerce')
    if isinstance(time_val, dtime):
        t = time_val
    elif pd.isna(time_val):
        t = None
    else:
        tstamp = pd.to_datetime(str(time_val), errors='coerce')
        t = None if pd.isna(tstamp) else tstamp.time()
    if pd.isna(d) and t is None:
        return pd.NaT
    if pd.isna(d) and t is not None:
        return pd.to_datetime(datetime.combine(datetime.today().date(), t))
    if t is None:
        return pd.to_datetime(d)
    return pd.to_datetime(datetime.combine(pd.to_datetime(d).date(), t))

# %%

df = pd.read_excel(INPUT_PATH, sheet_name="Sheet1")
df.head(10)

# %%
# Promoting first row to header
df_cols = df.iloc[0].to_list()
df1 = df.iloc[1:].copy()
df1.columns = df_cols

def dupcol(df, name, k):
    idxs = [i for i, c in enumerate(df.columns) if str(c).strip() == name]
    if len(idxs) <= k:
        # return an empty series with right index if missing
        return pd.Series([pd.NA] * len(df), index=df.index)
    return df.iloc[:, idxs[k]].reset_index(drop=True)

def to_time_series(s):
    s = pd.to_datetime(s, errors='coerce')
    # s is a Series here, so .dt is safe
    return s.dt.time

out = pd.DataFrame({
    'posting_dt'          : pd.to_datetime(dupcol(df1, 'Posting Dt', 0), errors='coerce'),
    'posting_time'        : to_time_series(dupcol(df1, 'Posting Time', 0)),
    'username'            : dupcol(df1, 'Username', 0),
    'number'              : dupcol(df1, 'Number', 0),
    'seeking_car'         : dupcol(df1, 'Car', 0),
    'seeking_date'        : dupcol(df1, 'Date', 0),
    'seeking_time'        : dupcol(df1, 'Time', 0),
    'seeking_location'    : dupcol(df1, 'Location', 0),
    'seeking_destination' : dupcol(df1, 'Destination', 0),
    'offering_car'        : dupcol(df1, 'Car', 1),
    'offering_date'       : dupcol(df1, 'Date', 1),
    'offering_time'       : dupcol(df1, 'Time', 1),
    'offering_location'   : dupcol(df1, 'Location', 1),
    'offering_destination': dupcol(df1, 'Destination', 1),
})


out['seek_dt'] = [parse_dt(d, t) for d, t in zip(out['seeking_date'], out['seeking_time'])]
out['offer_dt'] = [parse_dt(d, t) for d, t in zip(out['offering_date'], out['offering_time'])]

for c in ['seeking_location','seeking_destination','offering_location','offering_destination']:
    out[c] = out[c].apply(norm_city_name)

out.head(10)

# %%
from datetime import datetime, time as dtime
import pandas as pd

def parse_dt(date_val, time_val):
    d = pd.to_datetime(date_val, errors='coerce')
    if isinstance(time_val, dtime):
        t = time_val
    elif pd.isna(time_val):
        t = None
    else:
        tstamp = pd.to_datetime(str(time_val), errors='coerce')
        t = None if pd.isna(tstamp) else tstamp.time()
    if pd.isna(d) and t is None:
        return pd.NaT
    if pd.isna(d) and t is not None:
        return pd.to_datetime(datetime.combine(pd.Timestamp.today().date(), t))
    if t is None:
        return pd.to_datetime(d)
    return pd.to_datetime(datetime.combine(pd.to_datetime(d).date(), t))

out['seek_dt']  = [parse_dt(d, t) for d, t in zip(out['seeking_date'],  out['seeking_time'])]
out['offer_dt'] = [parse_dt(d, t) for d, t in zip(out['offering_date'], out['offering_time'])]

# %%
# See which headers are duplicated and how many times
pd.Series(df1.columns).value_counts().head(10)

# %%
legs = []
for _, r in out.iterrows():
    if r['seeking_location'] and r['seeking_destination']:
        legs.append({
            'type':'seek','operator':r['username'],'phone':r['number'],'car':r['seeking_car'],
            'depart_city':r['seeking_location'],'arrive_city':r['seeking_destination'],
            'depart_dt':r['seek_dt']
        })
    if r['offering_location'] and r['offering_destination']:
        legs.append({
            'type':'offer','operator':r['username'],'phone':r['number'],'car':r['offering_car'],
            'depart_city':r['offering_location'],'arrive_city':r['offering_destination'],
            'depart_dt':r['offer_dt']
        })

legs_df = pd.DataFrame(legs).dropna(subset=['depart_dt'])
legs_df['depart_cluster'] = legs_df['depart_city'].apply(city_cluster)
legs_df['arrive_cluster'] = legs_df['arrive_city'].apply(city_cluster)
legs_df['date'] = legs_df['depart_dt'].dt.date
legs_df = legs_df.sort_values('depart_dt').reset_index(drop=True)
legs_df.head(10)
# %%
#drive time matrix (in minutes) + buffer time
DUR = {}
def set_dur(a,b,mins):
    DUR[(a,b)] = mins
    DUR[(b,a)] = mins

# TODO: tune durations (minutes)
set_dur("Ranchi","Tata", 210)
set_dur("Ranchi","Ranchi", 45)
set_dur("Tata","Tata", 45)
set_dur("Ranchi","Dhanbad/Bokaro", 240)
set_dur("Tata","Dhanbad/Bokaro", 180)
set_dur("Ranchi","Kolkata", 420)
set_dur("Tata","Kolkata", 300)

BUFFER_MIN = 30

# %%
#Chain search for a target date (≥3 legs, prefer closed loops)

td = pd.to_datetime(TARGET_DATE).date()
legs_day = legs_df[legs_df['date']==td].reset_index(drop=True)

by_depart = {}
for i, r in legs_day.iterrows():
    by_depart.setdefault(r['depart_city'], []).append(i)
for k in by_depart:
    by_depart[k] = sorted(by_depart[k], key=lambda idx: legs_day.loc[idx,'depart_dt'])

def drive_minutes(a,b,default=240):
    return DUR.get((a,b), default)

adj = {i: [] for i in range(len(legs_day))}
for i, r in legs_day.iterrows():
    travel = drive_minutes(r['depart_cluster'], r['arrive_cluster'])
    earliest = r['depart_dt'] + pd.Timedelta(minutes=travel + BUFFER_MIN)
    cutoff = r['depart_dt'] + pd.Timedelta(hours=18)
    for j in by_depart.get(r['arrive_city'], []):
        if j == i: continue
        t2 = legs_day.loc[j,'depart_dt']
        if t2 >= earliest and t2 <= cutoff:
            adj[i].append(j)

def score(path):
    seeks = sum(1 for idx in path if legs_day.loc[idx,'type']=="seek")
    offers = len(path) - seeks
    startC = legs_day.loc[path[0],'depart_cluster']
    endC   = legs_day.loc[path[-1],'arrive_cluster']
    loop_bonus = 3 if startC == endC else 0
    return seeks*3 + offers*1 + loop_bonus

chains=[]
def dfs(path, max_legs=4):
    if len(path) >= 3:
        chains.append((path.copy(), score(path)))
    if len(path) >= max_legs:
        return
    last = path[-1]
    for nxt in adj[last]:
        if nxt in path: continue
        path.append(nxt)
        dfs(path, max_legs=max_legs)
        path.pop()

for i in range(len(legs_day)):
    dfs([i], max_legs=4)

chains_sorted = sorted(chains, key=lambda x: (x[1], len(x[0])), reverse=True)
len(chains_sorted), chains_sorted[:3]

# %%
#top chains
def chain_table(path):
    rows=[]
    for k, idx in enumerate(path, 1):
        r = legs_day.loc[idx]
        rows.append({
            "leg": k,
            "when": r['depart_dt'].strftime("%Y-%m-%d %H:%M"),
            "type": r['type'],
            "route": f"{r['depart_city']} → {r['arrive_city']}",
            "operator": r['operator'],
            "phone": r['phone'],
            "car": r['car'],
        })
    return pd.DataFrame(rows)

for k, (p, s) in enumerate(chains_sorted[:3], 1):
    print(f"Chain {k} — score={s}, len={len(p)}")
    display(chain_table(p))

# %%

if chains_sorted:
    best_path, best_score = chains_sorted[0]
    best_df = chain_table(best_path)
    best_df.to_csv("C://Users//Keerthana AK//Desktop//Internship//diva_best_chain.csv", index=False)
    contacts = (best_df[['operator','phone']].drop_duplicates())
    contacts.to_csv("C://Users//Keerthana AK//Desktop//Internship///diva_best_chain_contacts.csv", index=False)
    best_df.head()

# %%
%pip install fsspec

# %%
# Rough profit estimate for best chain (if all rides were paid)
DIST_KM = {("Ranchi","Tata"): 130, ("Tata","Ranchi"): 130}
BASE_FARE = 35      # set if there is a base
RATE_KM = 22        # ₹ per km (example)
COST_KM = 18       # fuel+ops per km (example)

def est_km(a,b):
    return DIST_KM.get((a,b), 130)

if chains_sorted:
    best_path, _ = chains_sorted[0]
    km_total = 0
    rev_total = 0
    cost_total = 0
    for idx in best_path:
        r = legs_day.loc[idx]
        a, b = r['depart_cluster'], r['arrive_cluster']
        km = est_km(a,b)
        km_total += km
        rev_total += BASE_FARE + RATE_KM * km
        cost_total += COST_KM * km
    print(f"KM total≈{km_total} | Rev≈₹{rev_total:.0f} | Cost≈₹{cost_total:.0f} | Profit≈₹{rev_total-cost_total:.0f}")

# %%

