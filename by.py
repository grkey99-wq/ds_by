# app.py
# 세종시 공공자전거(어울링) — 스토리텔링형 EDA 대시보드 (Streamlit)
# - 파일 업로드(CSV/XLSX) 또는 데모 데이터
# - 팔레트 통일, 지도/차트/표/정책요약 포함

import io
import re
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk

st.set_page_config(
    page_title="세종시 어울링 — 스토리텔링형 EDA 대시보드",
    layout="wide",
    page_icon="🚲",
)

# =========================
# 0) 전역 팔레트 / 스타일
# =========================
PALETTE = {
    "tertiary": "#F1F8F1",  # 배경
    "primary":  "#D9B526",  # 포인트(골드)
    "secondary":"#164B5F",  # 텍스트/강조(딥블루)
    "alert":    "#25ABD9",  # 하이라이트(청록)
    "muted":    "#5F7B87",
    "ink":      "#0A2530"
}
COLOR_SEQUENCE = [PALETTE["secondary"], PALETTE["primary"], PALETTE["alert"], PALETTE["muted"]]

px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = COLOR_SEQUENCE

# =========================
# 1) 유틸
# =========================
def norm_key(s: str) -> str:
    return re.sub(r"\s+", "", str(s or "")).lower()

def smart_num(v):
    if v is None or (isinstance(v, float) and math.isnan(v)) or str(v).strip()=="":
        return np.nan
    if isinstance(v, (int, float)) and np.isfinite(v):
        return float(v)
    s = str(v).strip()
    # 1.000,50 (유럽식) → 1000.50
    if re.fullmatch(r"-?\d{1,3}(\.\d{3})+(,\d+)?", s):
        s = s.replace(".", "").replace(",", ".")
    # "1,23" (소수점이 콤마 하나) → 1.23
    elif "," in s and "." not in s and s.count(",")==1:
        s = s.replace(",", ".")
    else:
        s = s.replace(",", "")
    m = re.search(r"-?\d+(\.\d+)?", s)
    return float(m.group(0)) if m else np.nan

def fix_lat_lon(lat, lon):
    """위경도 보정(엑셀에서 뒤바뀜 방지)"""
    def in_world(la, lo):
        return -90 <= la <= 90 and -180 <= lo <= 180
    if np.isfinite(lat) and np.isfinite(lon) and in_world(lat, lon):
        return lat, lon
    # 뒤바뀐 경우
    if np.isfinite(lat) and np.isfinite(lon) and in_world(lon, lat):
        return lon, lat
    return np.nan, np.nan

def score_header_for_months(cols):
    h = [norm_key(c) for c in cols]
    def has(p): return any(re.search(p, x) for x in h)
    score = 0
    months = [r"^4월|^apr$", r"^5월|^may$", r"^6월|^jun$", r"^7월|^jul$"]
    for m in months:
        if has(m): score += 2
    if has(r"대여소|정류장|station.*name|^name$"): score += 2
    if has(r"위도|lat|latitude|y좌표|^y$"): score += 1
    if has(r"경도|lon|lng|longitude|x좌표|^x$"): score += 1
    return score

def choose_best_sheet(xl: pd.ExcelFile):
    best = (None, -1)
    for name in xl.sheet_names:
        try:
            df = xl.parse(name, dtype=object)
        except Exception:
            continue
        s = score_header_for_months(df.columns)
        if s > best[1]:
            best = (name, s)
    return best[0]

def normalize_dataframe(df: pd.DataFrame):
    """업로드 데이터 → 표준 스키마로 정규화"""
    cols = list(df.columns)
    keys = {norm_key(c): c for c in cols}

    def find_key(patterns):
        for nk, raw in keys.items():
            for p in patterns:
                if re.fullmatch(p, nk):
                    return raw
        return None

    # 매핑 탐색
    k_station_name = find_key([r"(대여소명|대여소이름|정류장명|거점명|station[_]?name|name)"])
    k_station_id   = find_key([r"(대여소번호|정류장번호|station[_]?id|id|code)"])
    k_lat          = find_key([r"(위도|lat|latitude|y좌표|^y$|위도\(좌표\))"])
    k_lon          = find_key([r"(경도|lon|lng|longitude|x좌표|^x$|경도\(좌표\))"])

    def month_key(m, alias):
        return [rf"({m}|{m}월|{m}월대여|{m}월대여건수|{m}월이용건수|{alias})"]

    k_apr = find_key(month_key(4, "apr") + [r"4월.*건수|^apr$"])
    k_may = find_key(month_key(5, "may") + [r"5월.*건수|^may$"])
    k_jun = find_key(month_key(6, "jun") + [r"6월.*건수|^jun$"])
    k_jul = find_key(month_key(7, "jul") + [r"7월.*건수|^jul$"])
    k_total = find_key([r"(총대여건수|총대여|총계|총이용건수|누적대여건수|total|sum|누적)"])

    out_rows = []
    for _, r in df.iterrows():
        name = (str(r.get(k_station_name)) if k_station_name else None) or (str(r.get(k_station_id)) if k_station_id else None)
        if not name or name.strip() == "" or name.strip().lower() == "nan":
            continue
        lat = smart_num(r.get(k_lat)) if k_lat else np.nan
        lon = smart_num(r.get(k_lon)) if k_lon else np.nan
        lat, lon = fix_lat_lon(lat, lon)
        if not (np.isfinite(lat) and np.isfinite(lon)):
            # 지도는 제외되나, 나머지 차트는 동작할 수 있도록 lat/lon NaN 허용
            pass
        apr = smart_num(r.get(k_apr)) if k_apr else np.nan
        may = smart_num(r.get(k_may)) if k_may else np.nan
        jun = smart_num(r.get(k_jun)) if k_jun else np.nan
        jul = smart_num(r.get(k_jul)) if k_jul else np.nan
        total = smart_num(r.get(k_total)) if k_total else np.nan
        if not np.isfinite(total):
            total = sum(x for x in [apr, may, jun, jul] if np.isfinite(x)) or np.nan

        out_rows.append({
            "station_id": str(r.get(k_station_id) or ""),
            "station_name": name.strip(),
            "lat": lat, "lon": lon,
            "apr": float(apr) if np.isfinite(apr) else 0.0,
            "may": float(may) if np.isfinite(may) else 0.0,
            "jun": float(jun) if np.isfinite(jun) else 0.0,
            "jul": float(jul) if np.isfinite(jul) else 0.0,
            "total": float(total) if np.isfinite(total) else 0.0,
        })

    norm_df = pd.DataFrame(out_rows)
    return norm_df

def enrich_missing(df: pd.DataFrame):
    out = df.copy()
    # bike_count 보강
    if "bike_count" not in out.columns or not np.any(pd.to_numeric(out.get("bike_count", pd.Series(dtype=float)), errors="coerce").fillna(0) > 0):
        avg_per_bike = 110  # 총대여/보유수 가정
        out["bike_count"] = np.clip((out["total"].fillna(0) / max(avg_per_bike,1)).round().astype(int), 5, 80)

    # avg_duration_min, distance_km 보강
    if "avg_duration_min" not in out.columns or out["avg_duration_min"].isna().all():
        out["avg_duration_min"] = np.maximum(4, np.round(13 + 7*(np.random.rand(len(out))*2-1))).astype(int)
    if "distance_km" not in out.columns or out["distance_km"].isna().all():
        out["distance_km"] = np.round(1.8 + np.random.rand(len(out))*1.8, 2)

    return out

def build_hour_series(total):
    base = max(5, round(total / (24*60)))
    hours = []
    for h in range(24):
        morning = 80*np.exp(-((h-8)**2)/18)
        evening = 70*np.exp(-((h-19)**2)/20)
        hours.append(max(0, round(base + morning + evening)))
    return pd.DataFrame({"hour": list(range(24)), "avg": hours})

def build_hour_dow_heat():
    hours = list(range(6, 23))
    dows = list(range(7))  # 0=일 ~ 6=토
    mat = []
    for d in dows:
        row = []
        for h in hours:
            wk = 40 if (1 <= d <= 5 and 7 <= h <= 9) else (35 if (1 <= d <= 5 and 18 <= h <= 20) else 10 if (1 <= d <= 5) else 0)
            we = 30 if (d in (0,6) and 10 <= h <= 17) else (5 if d in (0,6) else 0)
            n = np.random.rand()*10 - 5
            row.append(max(0, round(20 + wk + we + n)))
        mat.append(row)
    return hours, dows, np.array(mat)

def simulate_reallocation(df, topN=5, bottomN=5, alpha=0.08):
    srt = df.sort_values("total", ascending=False)
    top = srt.head(topN)
    bottom = srt.tail(bottomN)
    take = round(alpha * top["total"].sum())
    before_total = df["total"].sum()
    return dict(
        beforeTotal=before_total,
        afterTotal=before_total,
        bottomBefore=bottom["total"].sum(),
        bottomAfter=bottom["total"].sum()+take,
        moved=take
    )

def monthly_totals(df):
    return dict(
        apr=float(df["apr"].sum()),
        may=float(df["may"].sum()),
        jun=float(df["jun"].sum()),
        jul=float(df["jul"].sum())
    )

def demo_data():
    return pd.DataFrame([
        dict(station_id="S001", station_name="어진동_푸르지오시티2차", lat=36.504, lon=127.262, apr=1800, may=4500, jun=4200, jul=4107, total=14607),
        dict(station_id="S002", station_name="나성동_현대자동차 앞",   lat=36.497, lon=127.259, apr=1200, may=3000, jun=2900, jul=2539, total=9639),
        dict(station_id="S003", station_name="어진동_행안부 별관",     lat=36.507, lon=127.266, apr= 900, may=2700, jun=2600, jul=2126, total=8326),
        dict(station_id="S004", station_name="보람동_세종시청 정문2", lat=36.477, lon=127.257, apr= 350, may= 820, jun= 780, jul= 650, total=2600),
        dict(station_id="S005", station_name="한솔동_첫마을1단지",     lat=36.490, lon=127.243, apr=  40, may=  45, jun=  30, jul=  35, total= 150),
    ])

# =========================
# 2) 사이드바: 업로드/옵션
# =========================
with st.sidebar:
    st.markdown(f"### 🚲 세종시 어울링 EDA")
    up = st.file_uploader("엑셀(.xlsx/.xls) 또는 CSV 업로드", type=["xlsx","xls","csv"])
    use_demo = st.button("데모 데이터 사용", type="primary")
    st.markdown("---")
    st.caption("팔레트: secondary(#164B5F), primary(#D9B526), alert(#25ABD9), muted(#5F7B87)")
    st.markdown(
        f"""
        <div style="background:{PALETTE['tertiary']};border:1px dashed #E6CD6A;padding:8px;border-radius:8px;color:#6a5300;">
          데이터에 시간대/요일별 세부 컬럼이 없으면 일부 그래프는 <b>시뮬레이션</b>으로 보강합니다.
        </div>
        """, unsafe_allow_html=True
    )

# =========================
# 3) 데이터 로딩
# =========================
DATA = None
status_msgs = []

if use_demo and up is None:
    df = demo_data()
    status_msgs.append("데모 데이터로 렌더링합니다.")
    DATA = enrich_missing(df)

elif up is not None:
    try:
        if up.type in ("text/csv",) or up.name.lower().endswith(".csv"):
            df = pd.read_csv(up)
        else:
            xl = pd.ExcelFile(up)
            sheet = choose_best_sheet(xl) or xl.sheet_names[0]
            df = xl.parse(sheet, dtype=object)
            status_msgs.append(f"선택된 시트: {sheet}")
        norm_df = normalize_dataframe(df)
        if norm_df.empty:
            # 좌표/정규화 실패 시 최소 집계 기반으로라도 작동하도록 데모 대체
            status_msgs.append("정규화 결과 0건 → 데모 데이터 대체")
            DATA = enrich_missing(demo_data())
        else:
            DATA = enrich_missing(norm_df)
            status_msgs.append(f"정규화 완료: {len(DATA)}개 대여소")
    except Exception as e:
        st.error(f"파일 처리 오류: {e}")
        DATA = enrich_missing(demo_data())
        status_msgs.append("예외 발생 → 데모 데이터 대체")

else:
    # 초기 상태: 안내
    st.info("좌측 사이드바에서 파일을 업로드하거나 ‘데모 데이터 사용’을 눌러주세요.")
    st.stop()

# =========================
# 4) 상단 제목/상태
# =========================
st.markdown(
    f"""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
      <span style="background:{PALETTE['primary']};color:#1a1a1a;padding:4px 8px;border-radius:999px;font-weight:700;font-size:12px;">Sejong Bike</span>
      <h2 style="margin:0;color:{PALETTE['secondary']}">세종시 공공자전거(어울링) — 스토리텔링형 EDA 대시보드</h2>
    </div>
    """, unsafe_allow_html=True
)
for m in status_msgs:
    st.caption(f"ℹ️ {m}")

# =========================
# 5) KPI & 월별 추이
# =========================
MONTHS = ["apr","may","jun","jul"]
LABELS = ["4월","5월","6월","7월"]
t = monthly_totals(DATA)
grand = t["apr"]+t["may"]+t["jun"]+t["jul"]
station_cnt = len(DATA)
monthly_avg = round(grand/4) if grand else 0
per_station_avg = round(grand/max(1,station_cnt))

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("총 대여 건수", f"{int(grand):,}")
kpi2.metric("월평균 대여 건수", f"{int(monthly_avg):,}")
kpi3.metric("총 대여소 수", f"{station_cnt:,}")
kpi4.metric("평균 대여소당 이용", f"{int(per_station_avg):,}")

col1, col2 = st.columns([2,1])
with col1:
    fig_bar = px.bar(
        x=LABELS, y=[t["apr"], t["may"], t["jun"], t["jul"]],
        labels={"x":"월", "y":"총 대여 건수"},
        title="월별 대여 건수 추이 (4~7월)",
    )
    fig_bar.update_traces(marker_color=PALETTE["primary"])
    fig_bar.update_layout(margin=dict(l=40,r=16,t=40,b=40))
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    # 간단 인사이트 박스
    order = sorted([("4월",t["apr"]),("5월",t["may"]),("6월",t["jun"]),("7월",t["jul"])], key=lambda x: x[1], reverse=True)
    peak = order[0][0] if order else "5월"
    st.markdown(
        f"""
        <div style="background:#FFF8DF;border:1px dashed #E6CD6A;padding:10px 12px;border-radius:12px;color:#6a5300;">
          <div style="font-weight:700;margin-bottom:6px;">요약 인사이트</div>
          <div><b>“{peak} 피크, 여름 장마철 감소”</b> — 6~7월 완만한 하락.</div>
          <div style="font-size:12px;color:{PALETTE['muted']};margin-top:6px;">정책 해석: 봄철 프로모션 강화, 장마철 유지보수·안전 캠페인 병행</div>
        </div>
        """, unsafe_allow_html=True
    )

st.markdown("---")

# =========================
# 6) 이용 패턴 (히트맵/라인/Top5 그룹)
# =========================
st.subheader("이용 패턴 분석 (Usage Patterns)", divider="blue")

# 시간대 평균 (시뮬레이션)
H = build_hour_series(grand)
fig_hour = go.Figure()
fig_hour.add_trace(go.Scatter(
    x=[f"{h}시" for h in H["hour"]],
    y=H["avg"],
    mode="lines+markers",
    line=dict(width=3, color=PALETTE["secondary"]),
    marker=dict(size=6, color=PALETTE["secondary"]),
    name="시간대별 평균 대여"
))
fig_hour.update_layout(title="시간대별 평균 대여량", margin=dict(l=40,r=16,t=40,b=40))
col_h1, col_h2 = st.columns(2)
with col_h1:
    # 히트맵
    hours, dows, mat = build_hour_dow_heat()
    fig_heat = go.Figure(data=go.Heatmap(
        z=mat,
        x=[f"{h}시" for h in hours],
        y=["일","월","화","수","목","금","토"],
        colorscale=[[0, PALETTE["tertiary"]],[1, PALETTE["alert"]]],
        colorbar=dict(title="대여량")
    ))
    fig_heat.update_layout(title="요일 × 시간대 히트맵", margin=dict(l=40,r=16,t=40,b=40))
    st.plotly_chart(fig_heat, use_container_width=True)
with col_h2:
    st.plotly_chart(fig_hour, use_container_width=True)

# Top5 대여소 — 월별 비교
top5 = DATA.sort_values("total", ascending=False).head(5)
fig_group = go.Figure()
for i, m in enumerate(MONTHS):
    fig_group.add_trace(go.Bar(
        x=top5["station_name"],
        y=top5[m],
        name=LABELS[i],
        marker_color=COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)]
    ))
fig_group.update_layout(
    barmode="group",
    title="Top5 대여소 — 월별 비교",
    margin=dict(l=40,r=16,t=40,b=40),
    xaxis=dict(tickfont=dict(size=11))
)
st.plotly_chart(fig_group, use_container_width=True)

st.markdown("---")

# =========================
# 7) 공간 분석 (지도/Top10/저이용 표)
# =========================
st.subheader("공간 분석 (Geo-Spatial Analysis)", divider="blue")

col_g1, col_g2 = st.columns([2,1])

with col_g1:
    # 지도 (pydeck)
    df_geo = DATA.dropna(subset=["lat","lon"])
    if df_geo.empty:
        st.info("좌표가 없어 지도를 표시할 수 없습니다.")
    else:
        tmin = df_geo["total"].min()
        tmax = df_geo["total"].max()
        def radius_scale(v):  # 6~24 px
            return 6 + 18 * (0 if tmax==tmin else (v - tmin) / (tmax - tmin))
        df_geo = df_geo.assign(radius=df_geo["total"].apply(radius_scale))
        center_lat = df_geo["lat"].mean()
        center_lon = df_geo["lon"].mean()
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_geo,
            get_position="[lon, lat]",
            get_radius="radius",
            get_fill_color=[37,171,217,140],  # alert(#25ABD9) RGBA
            get_line_color=[37,171,217],
            line_width_min_pixels=1,
            pickable=True,
            auto_highlight=True,
        )
        tooltip = {
            "text": "{station_name}\n총 대여: {total}\n4월:{apr} · 5월:{may} · 6월:{jun} · 7월:{jul}"
        }
        view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=12, pitch=0)
        st.pydeck_chart(pdk.Deck(map_style=None, initial_view_state=view_state, layers=[layer], tooltip=tooltip))
        st.caption("버블 크기 = 총 대여 건수")

with col_g2:
    top10 = DATA.sort_values("total", ascending=False).head(10).copy()
    fig_top10 = px.bar(
        top10.sort_values("total"),
        x="total", y="station_name",
        orientation="h",
        title="총 대여 건수 Top 10 대여소",
        labels={"total":"총 대여 건수","station_name":"대여소"},
        color_discrete_sequence=[PALETTE["alert"]],
        height=max(320, len(top10)*28 + 80)
    )
    fig_top10.update_layout(
        margin=dict(l=180, r=16, t=40, b=40),
        yaxis=dict(automargin=True, tickfont=dict(size=12))
    )
    st.plotly_chart(fig_top10, use_container_width=True)

# 저이용 표 (월평균 10건 이하) — 숫자 우측 정렬
low_df = DATA.assign(monthly_avg=(DATA["total"]/4)).loc[lambda d: d["monthly_avg"]<=10] \
             .sort_values("monthly_avg")
if low_df.empty:
    st.info("월평균 ≤ 10건 조건에 해당하는 대여소가 없습니다.")
else:
    show = low_df[["station_id","station_name","monthly_avg","total"]].copy()
    show["monthly_avg"] = show["monthly_avg"].round().astype(int)
    st.markdown("**저이용 대여소 (월평균 10건 이하)**")
    st.dataframe(
        show.rename(columns={"station_id":"대여소번호","station_name":"대여소 이름","monthly_avg":"월평균","total":"총 대여 건수"}),
        hide_index=True,
        use_container_width=True
    )

st.markdown("---")

# =========================
# 8) 운영 효율성 (산포/박스/히스토)
# =========================
st.subheader("운영 효율성 (Operational Efficiency)", divider="blue")

c1, c2 = st.columns(2)

with c1:
    # 회전율 산포도: x=보유수, y=총 대여, size=회전율
    x = DATA["bike_count"].fillna(0).astype(float)
    yv = DATA["total"].fillna(0).astype(float)
    rot = np.where(x>0, yv/x, 0.0)
    # size 스케일
    max_rot = max(rot.max(), 1.0)
    size = 8 + 20*(rot/max_rot)
    fig_scatter = px.scatter(
        DATA, x=x, y=yv, size=size, hover_name="station_name",
        title="보유 자전거 수 vs 총 대여 (버블=회전율)",
        labels={"x":"보유 자전거 수","y":"총 대여 건수"},
        color_discrete_sequence=[PALETTE["secondary"]],
    )
    fig_scatter.update_traces(marker=dict(opacity=0.8, line=dict(width=0)))
    st.plotly_chart(fig_scatter, use_container_width=True)

with c2:
    # 박스플롯: 대여소별 총 대여 분포
    fig_box = go.Figure(data=[go.Box(
        y=DATA["total"], boxpoints="outliers", marker_color=PALETTE["primary"], name="총 대여"
    )])
    fig_box.update_layout(title="대여소별 총 대여 분포", margin=dict(l=40,r=16,t=40,b=40))
    st.plotly_chart(fig_box, use_container_width=True)

# 히스토그램: 평균 이용 시간/거리
dur = pd.to_numeric(DATA["avg_duration_min"], errors="coerce").dropna()
dst = pd.to_numeric(DATA["distance_km"], errors="coerce").dropna()

fig_hist = go.Figure()
if len(dur):
    fig_hist.add_trace(go.Histogram(x=dur, name="평균 이용 시간(분)", opacity=0.75, marker_color=PALETTE["secondary"]))
if len(dst):
    fig_hist.add_trace(go.Histogram(x=dst, name="평균 이동 거리(km)", opacity=0.60, marker_color=PALETTE["alert"]))
if not len(fig_hist.data):
    fig_hist.add_trace(go.Histogram(x=[0], name="데이터 없음", marker_color=PALETTE["muted"]))

fig_hist.update_layout(
    title="평균 이용 시간/거리 분포", barmode="overlay", margin=dict(l=40,r=16,t=40,b=40)
)
st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")

# =========================
# 9) 정책 시뮬레이션 & 이벤트
# =========================
st.subheader("정책 제안 & 시뮬레이션 (Policy Insights)", divider="blue")

c3, c4 = st.columns(2)

with c3:
    sim = simulate_reallocation(DATA, topN=5, bottomN=5, alpha=0.08)
    fig_ba = go.Figure()
    fig_ba.add_trace(go.Bar(x=["Before","After"], y=[sim["beforeTotal"], sim["afterTotal"]],
                            name="총 대여(보존 가정)", marker_color=PALETTE["alert"]))
    fig_ba.add_trace(go.Bar(x=["Before","After"], y=[sim["bottomBefore"], sim["bottomAfter"]],
                            name="하위권 총 대여", marker_color=PALETTE["primary"]))
    fig_ba.update_layout(barmode="group", title="대여소 재배치 전후 (단순 시뮬레이션)",
                         margin=dict(l=40,r=16,t=40,b=40))
    st.plotly_chart(fig_ba, use_container_width=True)

with c4:
    series = [t["apr"], t["may"], t["jun"], t["jul"]]
    fig_evt = go.Figure()
    fig_evt.add_trace(go.Scatter(
        x=LABELS, y=series, mode="lines+markers",
        line=dict(width=3, color=PALETTE["secondary"]),
        marker=dict(color=PALETTE["secondary"]), name="월별 총합"
    ))
    fig_evt.add_shape(type="line", x0="5월", x1="6월", y0=0, y1=max(series) if series else 1,
                      line=dict(dash="dot", color=PALETTE["primary"]))
    fig_evt.update_layout(title="프로모션/이벤트 전후 추세 (예시)", margin=dict(l=40,r=16,t=40,b=40))
    st.plotly_chart(fig_evt, use_container_width=True)

# =========================
# 10) 정책 제안 요약 (하단)
# =========================
st.subheader("정책 제안 요약", divider="blue")
st.markdown(
    """
- **수요 재배치**: 상위 거점 과밀 완화, 저활성 지역 보강(실시간 리밸런싱/야간 이동).
- **수요 창출**: 봄철 집중 프로모션·정기권 할인, 출퇴근 연계 혜택(환승/시간대별 할인).
- **인프라 최적화**: 인기 거점 거치대 확충, 외곽 생활권 신규 거점, 전기자전거 단계 도입.
- **안전/품질**: 장마철 점검 주기 단축, 소모품 교체 주기 관리, 신고 처리 SLA 설정.
- **데이터 기반 운영**: 혼잡 예측 기반 차량 배치, 저이용 거점 대상 A/B 테스트(홍보/가격/경로).
"""
)

st.caption("© Sejong Bike · 본 대시보드는 4~7월 월별 수요/공간 분포/회전율 분석 결과를 기반으로 구성되었습니다.")
