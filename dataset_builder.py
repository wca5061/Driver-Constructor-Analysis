# driver & constructor names (static lists)
# Full calendars (~22 races/season) with plausible street/permanent mix
# outcomes (finish order, gaps, pits, DNFs, odds)

import os, json, math, random
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

# Config
SEED = 123
random.seed(SEED); np.random.seed(SEED)

YEARS = list(range(2018, 2026))
DATA_DIR = "data/synth_f1_2018_2025_realish"
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

GRID_SIZE = 20                 # 10 teams x 2
RACES_PER_SEASON = 22          # ~typical modern calendar
GENERATE_LAPS = True
LAPS_FRACTION = 0.35
FIA_POINTS = [25,18,15,12,10,8,6,4,2,1]

OVERROUND_PRE_RANGE = (1.06, 1.12)
OVERROUND_POST_RANGE = (1.03, 1.08)
HAZARD_ALPHA = 1.25

# ---------------- Static names ----------------
CONSTRUCTORS = [
    ("RED","Red Bull Racing"),
    ("MER","Mercedes-AMG Petronas"),
    ("FER","Ferrari"),
    ("MCL","McLaren"),
    ("AST","Aston Martin"),
    ("ALP","Alpine"),
    ("WIL","Williams"),
    ("HAA","Haas F1 Team"),
    ("RBM","RB (Visa Cash App)"),
    ("SAU","Kick Sauber"),
]

DRIVERS = [
    ("VER","Max Verstappen",   2015), ("PER","Sergio Pérez", 2011),
    ("HAM","Lewis Hamilton",   2007), ("RUS","George Russell", 2019),
    ("LEC","Charles Leclerc",  2018), ("SAI","Carlos Sainz",  2015),
    ("NOR","Lando Norris",     2019), ("PIA","Oscar Piastri", 2023),
    ("ALO","Fernando Alonso",  2001), ("STR","Lance Stroll",  2017),
    ("OCO","Esteban Ocon",     2016), ("GAS","Pierre Gasly",  2017),
    ("ALB","Alexander Albon",  2019), ("TSU","Yuki Tsunoda",  2021),
    ("HUL","Nico Hülkenberg",  2010), ("BOT","Valtteri Bottas", 2013),
    ("MAG","Kevin Magnussen",  2014), ("ZHO","Zhou Guanyu",    2022),
    ("SAR","Logan Sargeant",   2023), ("LAW","Liam Lawson",    2023),
    # bench reserves to allow swaps:
    ("DRI","Daniel Ricciardo", 2011), ("RIC","Robert Kubica",  2006),
    ("DEV","Nyck de Vries",    2022), ("MSC","Mick Schumacher",2021),
]

# Canonical track bank (name, country, street?, typical_laps, length_km)
TRACK_BANK = [
    ("Bahrain","Bahrain", False, 57, 5.412),
    ("Jeddah","Saudi Arabia", True, 50, 6.174),
    ("Melbourne","Australia", False, 58, 5.278),
    ("Suzuka","Japan", False, 53, 5.807),
    ("Shanghai","China", False, 56, 5.451),
    ("Miami","USA", True, 57, 5.412),
    ("Imola","Italy", False, 63, 4.909),
    ("Monaco","Monaco", True, 78, 3.337),
    ("Catalunya","Spain", False, 66, 4.657),
    ("Gilles Villeneuve","Canada", False, 70, 4.361),
    ("Red Bull Ring","Austria", False, 71, 4.318),
    ("Silverstone","UK", False, 52, 5.891),
    ("Hungaroring","Hungary", False, 70, 4.381),
    ("Spa-Francorchamps","Belgium", False, 44, 7.004),
    ("Zandvoort","Netherlands", False, 72, 4.259),
    ("Monza","Italy", False, 53, 5.793),
    ("Marina Bay","Singapore", True, 62, 4.940),
    ("Marina Circuit","Abu Dhabi", False, 58, 5.554),
    ("Austin","USA", False, 56, 5.513),
    ("Mexico City","Mexico", False, 71, 4.304),
    ("Interlagos","Brazil", False, 71, 4.309),
    ("Las Vegas","USA", True, 50, 6.201),
    ("Baku","Azerbaijan", True, 51, 6.003),
    ("Qatar","Qatar", False, 57, 5.380),
]

# ---------------- Utils ----------------
def softmax(x, temp=1.0):
    z = (x - np.max(x)) / temp
    e = np.exp(z)
    return e / e.sum()

def hazard_from_prob(p, alpha=HAZARD_ALPHA):
    p = float(np.clip(p, 1e-6, 1-1e-6))
    return float(alpha * np.log(p/(1-p)))

def sample_overround(lo, hi): return float(np.round(np.random.uniform(lo, hi), 4))

# ---------------- Calendar & roster synthesis ----------------
def build_calendar(year:int, n_rounds:int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # sample n_rounds unique tracks; keep season order stable-ish by shuffling with seed+year
    rng = np.random.default_rng(SEED + year)
    idx = rng.choice(len(TRACK_BANK), size=n_rounds, replace=False)
    selected = [TRACK_BANK[i] for i in idx]
    rows_cal, rows_tracks = [], []
    for rnd, (tname, country, street, laps, length) in enumerate(selected, start=1):
        track_id = f"{country[:3].upper()}_{tname[:6].upper()}"
        date = pd.Timestamp(year=year, month=max(3, (rnd%10)+3), day=min(28, (rnd*3)%28+1)).date()
        wet_prob = float(np.round(np.clip(np.random.beta(2,5), 0.02, 0.55), 2))
        sc_rate = float(np.round(np.random.uniform(1.0, 2.2), 2))
        overtake = float(np.round(np.random.uniform(0.35, 0.9), 2))
        altitude = int(np.random.choice([5,10,20,37,90,150,200,401,620,800,153,162,2285]))
        rows_cal.append([f"{year}_{rnd}", str(year), rnd, f"{tname} Grand Prix", track_id, date])
        rows_tracks.append([track_id, tname, country, bool(street), float(length), int(laps), wet_prob, sc_rate, overtake, altitude])
    cal = pd.DataFrame(rows_cal, columns=["event_id","season_id","round","grand_prix","track_id","date"])
    tracks = pd.DataFrame(rows_tracks, columns=["track_id","track_name","country","is_street","length_km","typical_laps","wet_prob","sc_rate","overtake_difficulty","altitude_m"])
    return cal, tracks

def build_roster(year:int) -> pd.DataFrame:
    # 10 teams × 2 drivers (stable core, light swaps)
    rng = np.random.default_rng(SEED + 10*year)
    teams = CONSTRUCTORS.copy()
    drivers = DRIVERS.copy()
    # pick 20 primary drivers (ensure stars present)
    primaries = [d for d in drivers if d[0] in {"VER","PER","HAM","RUS","LEC","SAI","NOR","PIA","ALO","STR","OCO","GAS","ALB","TSU","HUL","BOT","MAG","ZHO","SAR","LAW"}]
    driver_pool = primaries[:GRID_SIZE]

    # Pair drivers to teams (fixed for season start)
    assignments = []
    for i, (cid, cname) in enumerate(teams):
        d1 = driver_pool[(2*i) % GRID_SIZE]
        d2 = driver_pool[(2*i+1) % GRID_SIZE]
        assignments.append((cid, cname, d1, d2))

    # Build round-by-round roster; add occasional swaps (2–3 per season)
    swaps = rng.integers(low=2, high=4)
    swap_rounds = set(rng.choice(range(4, 20), size=swaps, replace=False))
    bench = [d for d in DRIVERS if d[0] not in {x[0] for x in driver_pool}]

    rows = []
    for rnd in range(1, RACES_PER_SEASON+1):
        for (cid, cname, d1, d2) in assignments:
            a1, a2 = d1, d2
            # occasional swap-in for this round
            if rnd in swap_rounds and bench:
                if rng.random() < 0.3:
                    a1 = rng.choice(bench)
                if rng.random() < 0.3:
                    a2 = rng.choice(bench)
            rows.append([str(year), rnd, a1[0], a1[1], cid, cname])
            rows.append([str(year), rnd, a2[0], a2[1], cid, cname])

    roster = pd.DataFrame(rows, columns=["season_id","round","driver_id","driver_name","constructor_id","constructor_name"])
    return roster

# ---------------- Latents with continuity ----------------
def assign_latents(drivers: List[str], teams: List[str], prev_drv: Dict[str,float], prev_tm: Dict[str,float]):
    drv = {d: (prev_drv[d] + np.random.normal(0,0.08)) if d in prev_drv else (1.0 + np.random.normal(0,0.35)) for d in drivers}
    tm  = {t: (prev_tm[t] + np.random.normal(0,0.06)) if t in prev_tm else (1.0 + np.random.normal(0,0.30)) for t in teams}
    return drv, tm

# ---------------- Race synthesis ----------------
def simulate_season(calendar_df, tracks_df, roster_df):
    seasons = []
    races = []
    race_entries = []
    pitstops = []
    sc_rows = []
    odds_rows = []

    prev_drv_strength = {}
    prev_tm_strength  = {}

    # season rows
    for season_id, grp in calendar_df.groupby("season_id"):
        seasons.append([season_id, int(grp["round"].max()), "Offline calendar; synthetic outcomes"])
    seasons_df = pd.DataFrame(seasons, columns=["season_id","n_races_planned","notes"])

    # iterate races
    calendar_df = calendar_df.sort_values(["season_id","round"]).reset_index(drop=True)
    for _, r in calendar_df.iterrows():
        season_id = str(r["season_id"]); rnd = int(r["round"])
        track_id = r["track_id"]; date = r["date"]
        t = tracks_df.set_index("track_id").loc[track_id]
        is_street = bool(t["is_street"])
        laps_scheduled = int(t["typical_laps"])
        wet_prob = float(t["wet_prob"]); sc_rate = float(t["sc_rate"])

        # roster
        lineup = roster_df[(roster_df["season_id"]==season_id) & (roster_df["round"]==rnd)].copy()
        if len(lineup) == 0: continue

        # latents with continuity
        drv_lat, tm_lat = assign_latents(lineup["driver_id"].unique().tolist(), lineup["constructor_id"].unique().tolist(), prev_drv_strength, prev_tm_strength)
        prev_drv_strength.update(drv_lat); prev_tm_strength.update(tm_lat)

        # Weather + SC
        weather_roll = np.random.rand()
        if weather_roll < wet_prob * 0.6:
            weather, rain_share = "wet", float(np.clip(np.random.beta(3,2), 0.4, 1.0))
        elif weather_roll < wet_prob * 1.3:
            weather, rain_share = "mixed", float(np.clip(np.random.beta(2,3), 0.2, 0.8))
        else:
            weather, rain_share = "dry", 0.0
        sc_lambda = sc_rate * (1.15 if weather!="dry" else 1.0) * (1.10 if is_street else 1.0)
        sc_total = int(np.random.poisson(sc_lambda))
        vsc_total = int(np.random.binomial(n=max(0, sc_total+1), p=0.35))
        temp_c = float(np.round(np.random.normal(24 if weather!="wet" else 18, 4),1))
        wind_mps = float(np.round(np.abs(np.random.normal(4, 2)),1))
        rain_mm = float(np.round(3.0*rain_share + np.random.normal(0,0.5), 1)) if weather!="dry" else 0.0

        race_id = f"{season_id}_{track_id}"
        races.append([race_id, season_id, rnd, track_id, date, weather, laps_scheduled, sc_total, vsc_total, temp_c, wind_mps, rain_mm])

        # Quali-like grid
        grid_scores = []
        for _, e in lineup.iterrows():
            d, c = e["driver_id"], e["constructor_id"]
            qmu = 0.6*drv_lat[d] + 0.9*tm_lat[c] + (0.12 if is_street else 0) + (0.15 if weather=="dry" else 0) + np.random.normal(0, 0.15)
            grid_scores.append((d, c, qmu))
        grid_sorted = sorted(grid_scores, key=lambda x: x[2], reverse=True)
        quali_rank = {grid_sorted[i][0]: i+1 for i in range(len(grid_sorted))}
        penalties = {did: int(np.random.choice([0,3,5,10], p=[0.88,0.07,0.04,0.01])) for did,_,_ in grid_sorted}
        grid_tiebreak = sorted([(d,c,quali_rank[d]+penalties[d], np.random.rand()) for d,c,_ in grid_sorted], key=lambda z: (z[2], z[3]))
        grid_map = {grid_tiebreak[i][0]: i+1 for i in range(len(grid_tiebreak))}

        # Odds proxies
        pre_raw, post_raw = [], []
        for _, e in lineup.iterrows():
            d, c = e["driver_id"], e["constructor_id"]
            pre_score = 0.7*drv_lat[d] + 1.0*tm_lat[c] + 0.1*np.random.normal(0,1)
            post_score = pre_score + 0.25*(np.mean(list(grid_map.values())) - grid_map[d])*(1.0/len(grid_map))
            pre_raw.append((d, pre_score)); post_raw.append((d, post_score))
        over_pre = sample_overround(*OVERROUND_PRE_RANGE)
        over_post = sample_overround(*OVERROUND_POST_RANGE)
        p_pre = softmax(np.array([s for _, s in pre_raw]))
        p_post = softmax(np.array([s for _, s in post_raw]))
        p_pre_map = {did: float(p_pre[i]) for i, (did,_) in enumerate(pre_raw)}
        p_post_map = {did: float(p_post[i]) for i, (did,_) in enumerate(post_raw)}
        lam_pre_map = {did: hazard_from_prob(p_pre_map[did]) for did,_ in pre_raw}
        lam_post_map= {did: hazard_from_prob(p_post_map[did]) for did,_ in post_raw}

        # Pits, DNFs, score per entry
        entries_meta, dnf_flags, pits_records = [], {}, []
        for _, e in lineup.iterrows():
            did, cid = e["driver_id"], e["constructor_id"]
            # pits
            pit_count = int(np.random.choice([1,2,3], p=[0.15,0.7,0.15])) if weather=="dry" else int(np.random.choice([2,3,4], p=[0.25,0.55,0.20]))
            crew = np.random.normal(0,0.15)
            pit_mu, pit_sigma = 2.9 - 0.12*crew, 0.18
            stop_times = np.random.lognormal(mean=pit_mu, sigma=pit_sigma, size=pit_count) + np.random.exponential(0.2, size=pit_count)
            error_flags = np.random.rand(pit_count) < (0.08 - 0.05*np.clip(crew, -0.5, 0.5))
            error_add = error_flags * np.random.uniform(2.0,8.0, size=pit_count)
            stop_times = stop_times + error_add
            pit_total = float(np.round(stop_times.sum(),3))
            # penalties
            pen = float(np.random.choice([0.0, 5.0, 10.0], p=[0.92,0.06,0.02]))
            # dnf
            risk = 0.05 + 0.02*int(is_street) + 0.02*(weather!="dry") + 0.01*min(sc_total,3)
            dnf = bool(np.random.rand() < risk)
            dnf_reason = np.random.choice(["engine","crash","gearbox","electronics"], p=[0.35,0.45,0.1,0.1]) if dnf else ""
            dnf_flags[did] = dnf
            # pit detail
            if pit_count>0:
                base_laps = np.linspace(0.2, 0.85, pit_count) * laps_scheduled
                base_laps = np.clip(base_laps.astype(int), 1, laps_scheduled-1)
                for i in range(pit_count):
                    stint_from = np.random.choice(["S","M","H"]); stint_to = np.random.choice(["S","M","H"])
                    pits_records.append([f"{race_id}_{did}_{str(i+1).zfill(2)}", f"{race_id}_{did}", int(base_laps[i]), stint_from, stint_to,
                                         float(np.round(stop_times[i],3)), bool(error_flags[i]), float(np.round(error_add[i],3)) if error_flags[i] else 0.0])
            entries_meta.append({"did":did,"cid":cid,"quali_rank":int(quali_rank[did]),"grid_penalty":int(penalties[did]),"grid":int(grid_map[did]),
                                 "pit_stops":int(pit_count), "pit_time_total_s":pit_total, "penalties_s":pen, "dnf":dnf, "dnf_reason":dnf_reason})

        # scoring to finish
        scores = []
        for ent in entries_meta:
            did, cid = ent["did"], ent["cid"]
            base = drv_lat[did] + tm_lat[cid]
            grid_help = 0.18 * (np.mean([e['grid'] for e in entries_meta]) - ent["grid"])
            sc_noise = np.random.normal(0, 0.10 + 0.03*sc_total)
            pit_pen = 0.008 * ent["pit_time_total_s"]; time_pen = 0.005 * ent["penalties_s"]
            tire_bonus = np.random.normal(0, 0.05)
            score = base + grid_help + sc_noise - pit_pen - time_pen + tire_bonus + np.random.normal(0,0.10)
            if ent["dnf"]: score = -10.0 + np.random.normal(0,0.5)
            scores.append((did, score))
        sorted_finish = sorted(scores, key=lambda x: x[1], reverse=True)
        finish_map = {sorted_finish[i][0]: i+1 for i in range(len(sorted_finish))}

        # time gaps
        svals = np.array([s for _, s in sorted_finish]); leader = svals[0]
        deltas = leader - svals; track_scale = 85.0 + 0.5*float(t["length_km"])*10.0
        time_gaps = track_scale * (np.exp(np.clip(deltas, 0, None)) - 1.0)
        time_gap_map = {}
        for i, (did, _) in enumerate(sorted_finish):
            time_gap_map[did] = (np.nan if dnf_flags[did] else float(np.round(time_gaps[i],3)))

        classified_map = {ent["did"]: (not ent["dnf"]) for ent in entries_meta}

        # entries rows + odds snapshots
        for ent in entries_meta:
            did, cid = ent["did"], ent["cid"]
            finish_pos = int(finish_map[did]); pts = float(FIA_POINTS[finish_pos-1]) if classified_map[did] and finish_pos<=10 else 0.0
            ipre = float(np.round(p_pre_map[did],6)); ipost = float(np.round(p_post_map[did],6))
            odds_pre = float(np.round(1.0/np.clip(ipre, 1e-6, 1-1e-6), 4))
            odds_post= float(np.round(1.0/np.clip(ipost,1e-6, 1-1e-6), 4))
            race_entries.append([
                f"{race_id}_{did}", race_id, season_id, rnd, did, cid,
                int(ent["grid"]), int(ent["quali_rank"]), int(ent["grid_penalty"]),
                finish_pos, bool(classified_map[did]), bool(ent["dnf"]), ent["dnf_reason"],
                pts, float(time_gap_map[did]) if not math.isnan(time_gap_map[did]) else np.nan,
                int(ent["pit_stops"]), float(np.round(ent["pit_time_total_s"],3)), float(ent["penalties_s"]),
                int(sc_total*2 if sc_total>0 else 0),
                np.random.choice(["S-M","M-H","S-H","M-M","H-H"], p=[0.25,0.35,0.15,0.15,0.10]),
                int(np.random.poisson(0.3 + 0.4*int(is_street))), float(np.round(rain_share,3)), bool(is_street),
                odds_pre, odds_post, float(over_pre), float(over_post),
                ipre, ipost, float(np.round(lam_pre_map[did],6)), float(np.round(lam_post_map[did],6)),
                0.0, 0.0, float(drv_lat[did]), float(tm_lat[cid]), float(np.random.normal(0,0.15))
            ])
            odds_rows += [
                [f"{race_id}_{did}_PRE", race_id, did, "pre", odds_pre, float(over_pre), ipre],
                [f"{race_id}_{did}_POST", race_id, did, "post_quali", odds_post, float(over_post), ipost]
            ]

        # safety car detail
        lap_cursor = 1
        for i in range(sc_total):
            span = int(np.random.randint(2,5))
            low = max(1, lap_cursor); high = max(2, laps_scheduled - span)
            if low >= high: break
            start = int(np.random.randint(low, high)); end = int(min(laps_scheduled-1, start+span))
            if end <= start: continue
            kind = np.random.choice(["SC","VSC"], p=[0.65,0.35])
            sc_rows.append([f"{race_id}_SC{i+1}", race_id, kind, int(start), int(end), np.random.choice(["debris","crash","car stopped","weather"])])
            lap_cursor = end+1

        pitstops += pits_records

    # Build tables
    races_df = pd.DataFrame(races, columns=["race_id","season_id","round","track_id","date","weather","laps_scheduled","sc_total","vsc_total","temp_c","wind_mps","rain_mm"])
    race_entries_df = pd.DataFrame(race_entries, columns=[
        "entry_id","race_id","season_id","round","driver_id","constructor_id",
        "grid","quali_rank","grid_penalty","finish_position","classified","dnf","dnf_reason",
        "points","time_gap_s","pit_stops","pit_time_total_s","penalties_s","sc_laps_behind",
        "tyre_strategy","incidents","rain_share","street",
        "odds_pre","odds_post_quali","ovr_pre","ovr_post","implied_p_pre","implied_p_post",
        "hazard_lambda_pre","hazard_lambda_post","driver_form_z","constructor_form_z",
        "driver_ability_true","constructor_strength_true","race_noise_true"
    ])
    pitstops_df = pd.DataFrame(pitstops, columns=["pit_id","entry_id","lap","stint_from","stint_to","stationary_time_s","errors","error_time_s"])
    sc_df = pd.DataFrame(sc_rows, columns=["sc_id","race_id","kind","start_lap","end_lap","cause"])
    odds_snapshots_df = pd.DataFrame(odds_rows, columns=["odds_id","race_id","driver_id","snapshot","decimal_odds","overround","implied_prob"])

    # Forms
    race_entries_df = race_entries_df.sort_values(["season_id","round","driver_id"]).reset_index(drop=True)
    def compute_form(df, keycol):
        vals = []
        for (season, key), grp in df.groupby(["season_id", keycol]):
            grp = grp.sort_values("round")
            s = []
            for _, rr in grp.iterrows():
                s.append(0.0 if rr["dnf"] else max(0, 23 - rr["finish_position"]))
            s = pd.Series(s, index=grp.index, dtype=float)
            rolling = s.rolling(3, min_periods=1).mean()
            mu = rolling.mean(); sd = (rolling.std(ddof=0) or 1.0)
            z = (rolling - mu) / sd
            vals.append(pd.DataFrame({"idx": grp.index, "z": z.values}))
        return pd.concat(vals).set_index("idx").sort_index()["z"]
    race_entries_df.loc[:, "driver_form_z"] = compute_form(race_entries_df, "driver_id").values
    race_entries_df.loc[:, "constructor_form_z"] = compute_form(race_entries_df, "constructor_id").values

    # Laps (subset)
    laps_rows = []
    if GENERATE_LAPS:
        races_map = races_df.set_index("race_id").to_dict("index")
        for _, ent in race_entries_df.iterrows():
            race_id = ent["race_id"]; laps_sched = int(races_map[race_id]["laps_scheduled"])
            gen_laps = max(5, int(laps_sched * LAPS_FRACTION))
            d_ability = float(ent["driver_ability_true"]); c_strength = float(ent["constructor_strength_true"])
            base = 92.0 - 0.9*d_ability - 1.1*c_strength + np.random.normal(0, 0.8)
            for k in range(1, gen_laps+1):
                sc_flag = races_map[race_id]["sc_total"]>0 and (np.random.rand() < (0.08 + 0.03*races_map[race_id]["sc_total"]))
                lap_time = base + np.random.normal(0, 0.7) + (3.0 if sc_flag else 0.0)
                laps_rows.append([
                    f"{race_id}_{ent['driver_id']}_L{k}", f"{race_id}_{ent['driver_id']}", int(k),
                    float(np.round(lap_time,3)),
                    float(np.round(lap_time*0.29 + np.random.normal(0,0.08),3)),
                    float(np.round(lap_time*0.36 + np.random.normal(0,0.08),3)),
                    float(np.round(lap_time*0.35 + np.random.normal(0,0.08),3)),
                    bool(sc_flag),
                    float(np.round(np.random.beta(2,5),3)),
                    float(np.round(np.random.gamma(2.0, 0.2),3))
                ])
    laps_df = pd.DataFrame(laps_rows, columns=["lap_id","entry_id","lap","lap_time_s","sector1_s","sector2_s","sector3_s","sc_flag","rain_intensity","traffic_penalty_s"])

    # Dimensions: drivers & constructors
    drivers_dim = pd.DataFrame({ "driver_id":[d[0] for d in DRIVERS], "driver_name":[d[1] for d in DRIVERS],
                                 "rookie_season":[d[2] for d in DRIVERS] })
    drivers_dim["style"] = np.random.choice(["aggressive","balanced","smooth"], size=len(drivers_dim), p=[0.35,0.40,0.25])
    drivers_dim["ability_true"] = 1.0 + np.random.normal(0,0.35, size=len(drivers_dim))
    drivers_dim["wet_skill_true"] = np.random.normal(0,0.20, size=len(drivers_dim))
    drivers_dim["quali_skill_true"] = np.random.normal(0,0.20, size=len(drivers_dim))
    drivers_dim["tire_deg_mgmt_true"] = np.random.normal(0,0.20, size=len(drivers_dim))
    drivers_dim["risk_propensity_true"] = np.clip(np.random.normal(0.05, 0.10, size=len(drivers_dim)), 0.0, 0.5)

    constructors_dim = pd.DataFrame({ "constructor_id":[c[0] for c in CONSTRUCTORS], "constructor_name":[c[1] for c in CONSTRUCTORS] })
    constructors_dim["power_unit"] = constructors_dim["constructor_name"] + " PU"
    constructors_dim["base_strength_true"] = 1.0 + np.random.normal(0,0.30, size=len(constructors_dim))
    constructors_dim["aero_coeff_true"] = np.random.normal(0, 0.15, size=len(constructors_dim))
    constructors_dim["pit_crew_skill_true"] = np.random.normal(0, 0.15, size=len(constructors_dim))

    # driver-constructors spans
    dc = (roster_df
          .assign(start_round=lambda d: d["round"], end_round=lambda d: d["round"])
          .groupby(["season_id","driver_id","constructor_id"], as_index=False)
          .agg(start_round=("start_round","min"), end_round=("end_round","max")))
    dc["dc_id"] = dc.apply(lambda r: f"{r['season_id']}_{r['driver_id']}_{r['constructor_id']}", axis=1)
    dc = dc[["dc_id","season_id","driver_id","constructor_id","start_round","end_round"]]

    return seasons_df, constructors_dim, drivers_dim, tracks_df, races_df, dc, race_entries_df, pitstops_df, sc_df, odds_snapshots_df, laps_df

# ---------------- Build offline full dataset ----------------
if __name__ == "__main__":
    calendar_all = []; tracks_all = []; roster_all = []
    for yr in YEARS:
        cal, trk = build_calendar(yr, RACES_PER_SEASON)
        ros = build_roster(yr)
        calendar_all.append(cal); tracks_all.append(trk); roster_all.append(ros)
    calendar_df = pd.concat(calendar_all, ignore_index=True)
    tracks_df   = pd.concat(tracks_all, ignore_index=True).drop_duplicates("track_id")
    roster_df   = pd.concat(roster_all, ignore_index=True)

    seasons_df, constructors_df, drivers_df, tracks_df, races_df, dc_df, entries_df, pits_df, sc_df, odds_df, laps_df = simulate_season(calendar_df, tracks_df, roster_df)

    # Write files
    seasons_df.to_csv(os.path.join(DATA_DIR, "seasons.csv"), index=False)
    constructors_df.to_csv(os.path.join(DATA_DIR, "constructors.csv"), index=False)
    drivers_df.to_csv(os.path.join(DATA_DIR, "drivers.csv"), index=False)
    tracks_df.to_csv(os.path.join(DATA_DIR, "tracks.csv"), index=False)
    races_df.to_csv(os.path.join(DATA_DIR, "races.csv"), index=False)
    dc_df.to_csv(os.path.join(DATA_DIR, "driver_constructors.csv"), index=False)
    entries_df.to_csv(os.path.join(DATA_DIR, "race_entries.csv"), index=False)
    pits_df.to_csv(os.path.join(DATA_DIR, "pitstops.csv"), index=False)
    sc_df.to_csv(os.path.join(DATA_DIR, "safetycars.csv"), index=False)
    odds_df.to_csv(os.path.join(DATA_DIR, "odds_snapshots.csv"), index=False)
    laps_df.to_csv(os.path.join(DATA_DIR, "laps.csv"), index=False)

    readme = {
        "seed": SEED,
        "source": "Offline full calendar & rosters with real names; synthetic outcomes",
        "notes": {
            "grid_size": GRID_SIZE,
            "races_per_season": RACES_PER_SEASON,
            "odds": "softmax of latent driver+team with overround; hazard = alpha*logit(p)",
            "points": FIA_POINTS,
            "laps_fraction": LAPS_FRACTION
        },
        "files": ["seasons.csv","constructors.csv","drivers.csv","tracks.csv","races.csv",
                  "driver_constructors.csv","race_entries.csv","pitstops.csv",
                  "safetycars.csv","odds_snapshots.csv","laps.csv"]
    }
    with open(os.path.join(DATA_DIR, "README.json"), "w") as f:
        json.dump(readme, f, indent=2)

    # Quick shapes print
    print("✅ Built offline full real-ish dataset at:", DATA_DIR)
    for f, df in [
        ("seasons", seasons_df), ("constructors", constructors_df), ("drivers", drivers_df),
        ("tracks", tracks_df), ("races", races_df), ("driver_constructors", dc_df),
        ("race_entries", entries_df), ("pitstops", pits_df), ("safetycars", sc_df),
        ("odds_snapshots", odds_df), ("laps", laps_df)
    ]:
        print(f"{f:22s}", df.shape)
