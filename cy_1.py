# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

def main():
    st.set_page_config(page_title="어울링 공공자전거 심층 대시보드", layout="wide")

    # -----------------------------
    # 색상 팔레트
    # -----------------------------
    COLORS = {
        "tertiary": "#F1F8F1",
        "primary":  "#D9B526",
        "secondary":"#164B5F",
        "alert":    "#25ABD9",
        "gray":     "#CBD5E1"
    }

    st.title("🚲 어울링 공공자전거 심층 대시보드")
    st.caption("권역·시간대·저이용 대여소 분석 — 수치 레이블/툴팁 중심")

    # -----------------------------
    # 데모 데이터 (CSV 미업로드 시 사용)
    # -----------------------------
    DEMO_MONTHLY = pd.DataFrame({
        "대여소번호":["SJ_00420","SJ_00550","SJ_00455","SJ_00117","SJ_00436"],
        "대여소 이름":["가람동_세종이마트앞","가람동_세종중부발전(입구)","가람동_지역난방공사 앞","고운동_ 침례교회앞","고운동_가락 111동"],
        "4월 대여 건수":[740,47,123,9,207],
        "5월 대여 건수":[890,41,146,17,218],
        "6월 대여 건수":[772,37,163,16,295],
        "7월 대여 건수":[731,21,151,11,212],
        "총 대여 건수":[3133,146,583,53,932],
        "위도":[36.470509,36.466008,36.465209,36.517593,36.504293],
        "경도":[127.251684,127.247185,127.245724,127.231787,127.234777],
        "POI_500m":[24,6,12,1,15],
    })
    DEMO_STATIONS = pd.DataFrame({
        "대여소 아이디":["SJ_00420","SJ_00550","SJ_00455","SJ_00117","SJ_00436"],
        "대여소 이름":["가람동_세종이마트앞","가람동_세종중부발전(입구)","가람동_지역난방공사 앞","고운동_ 침례교회앞","고운동_가락 111동"],
        "권역":["가람동","가람동","가람동","고운동","고운동"],
        "위도":[36.470509,36.466008,36.465209,36.517593,36.504293],
        "경도":[127.251684,127.247185,127.245724,127.231787,127.234777],
    })
    DEMO_USAGE = pd.DataFrame({
        "시작대여일자":pd.to_datetime([
            "2022-06-01 07:12:00","2022-06-01 08:05:00","2022-06-02 18:21:00",
            "2022-06-04 11:30:00","2022-06-05 21:10:00"
        ]),
        "주행거리":[1450,980,2100,3500,5200],
        "주행시간":[9,7,13,26,35],
        "시작 대여소":["SJ_00420","SJ_00420","SJ_00455","SJ_00436","SJ_00550"]
    })

    # -----------------------------
    # 사이드바: 파일 업로드
    # -----------------------------
    st.sidebar.header("데이터 업로드 (CSV)")
    st.sidebar.markdown(
        "- 엑셀 원본은 시트별 CSV(이용현황/대여소현황/이용정보)로 내보낸 뒤 업로드하세요.\n"
        "- 컬럼명이 다르면 아래 매핑에서 실제 이름을 선택하세요."
    )
    usage_file    = st.sidebar.file_uploader("이용현황 CSV (행 단위 기록)", type=["csv"])
    stations_file = st.sidebar.file_uploader("대여소현황 CSV", type=["csv"])
    monthly_file  = st.sidebar.file_uploader("이용정보(월별) CSV", type=["csv"])

    if monthly_file is not None:
        monthly = pd.read_csv(monthly_file)
    else:
        monthly = DEMO_MONTHLY.copy()

    if stations_file is not None:
        stations = pd.read_csv(stations_file)
    else:
        stations = DEMO_STATIONS.copy()

    if usage_file is not None:
        usage = pd.read_csv(usage_file, parse_dates=True, infer_datetime_format=True)
    else:
        usage = DEMO_USAGE.copy()

    # -----------------------------
    # 컬럼 매핑(드롭다운) — 기본값을 실제 컬럼명으로 설정
    # -----------------------------
    st.sidebar.header("컬럼 매핑")
    station_cols = stations.columns.tolist()
    monthly_cols = monthly.columns.tolist()
    usage_cols   = usage.columns.tolist()

    # 권역: stations['권역']
    region_col = st.sidebar.selectbox(
        "권역 컬럼(대여소현황)",
        options=station_cols,
        index=station_cols.index("권역") if "권역" in station_cols else 0
    )
    # 조인 키: monthly['대여소번호'] ↔ stations['대여소 아이디']
    join_monthly_col = st.sidebar.selectbox(
        "조인 키(월별→대여소현황)",
        options=monthly_cols,
        index=monthly_cols.index("대여소번호") if "대여소번호" in monthly_cols else 0
    )
    join_station_col = st.sidebar.selectbox(
        "조인 키(대여소현황)",
        options=station_cols,
        index=station_cols.index("대여소 아이디") if "대여소 아이디" in station_cols else 0
    )
    # 시간: usage['시작대여일자']
    time_col = st.sidebar.selectbox(
        "시간 컬럼(이용현황)",
        options=usage_cols,
        index=usage_cols.index("시작대여일자") if "시작대여일자" in usage_cols else 0
    )

    # 타입 보정
    if time_col in usage.columns:
        usage[time_col] = pd.to_datetime(usage[time_col], errors="coerce")

    # -----------------------------
    # KPI
    # -----------------------------
    total_rides = monthly.get("총 대여 건수", pd.Series(dtype=float)).sum()
    num_stations = len(stations) if len(stations)>0 else len(monthly)
    avg_dist = usage.get("주행거리", pd.Series(dtype=float)).mean()
    avg_time = usage.get("주행시간", pd.Series(dtype=float)).mean()

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("총 대여 건수", f"{int(total_rides):,}" if pd.notnull(total_rides) else "-")
    with k2:
        st.metric("대여소 수", f"{int(num_stations):,}")
    with k3:
        st.metric("평균 주행거리(m)", f"{int(avg_dist):,}" if pd.notnull(avg_dist) else "-")
    with k4:
        st.metric("평균 주행시간(분)", f"{int(avg_time):,}" if pd.notnull(avg_time) else "-")

    st.markdown("---")

    # -----------------------------
    # 권역별 비교 (총 대여 건수 vs 대여소 수) — 색상 오류 수정 & 수치 레이블
    # -----------------------------
    # 조인용 딕셔너리: (stations) 대여소 아이디 -> 권역
    id_to_region = dict(zip(stations[join_station_col].astype(str), stations[region_col]))
    monthly["_region_"] = monthly[join_monthly_col].astype(str).map(id_to_region).fillna("기타")

    region_agg = monthly.groupby("_region_", as_index=False).agg(
        총대여=("총 대여 건수","sum"),
        대여소수=(join_monthly_col, "nunique")
    )

    colA, colB = st.columns([1.2,1])

    # 명시적 색상 지정 + 서로 다른 offsetgroup으로 색상/레이어 충돌 방지
    fig_region = go.Figure()
    fig_region.add_trace(go.Bar(
        name="총 대여 건수",
        x=region_agg["_region_"],
        y=region_agg["총대여"],
        marker=dict(color=COLORS["secondary"]),
        text=region_agg["총대여"],
        textposition="outside",
        offsetgroup="g_total",
        hovertemplate="권역=%{x}<br>총 대여 건수=%{y:,}<extra></extra>"
    ))
    fig_region.add_trace(go.Bar(
        name="대여소 수",
        x=region_agg["_region_"],
        y=region_agg["대여소수"],
        marker=dict(color=COLORS["primary"]),
        text=region_agg["대여소수"],
        textposition="outside",
        offsetgroup="g_cnt",
        yaxis="y2",
        hovertemplate="권역=%{x}<br>대여소 수=%{y:,}<extra></extra>"
    ))
    fig_region.update_layout(
        barmode="group",
        title="권역별 총대여 수 vs 대여소 수",
        xaxis_title="권역",
        yaxis=dict(title="총 대여 건수"),
        yaxis2=dict(title="대여소 수", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=1.12),
        template="plotly_white"
    )
    with colA:
        st.plotly_chart(fig_region, use_container_width=True)

    # -----------------------------
    # 월별 총 대여 추세 (4~7월) — 수치 레이블
    # -----------------------------
    monthly_cols_ = [c for c in ["4월 대여 건수","5월 대여 건수","6월 대여 건수","7월 대여 건수"] if c in monthly.columns]
    if monthly_cols_:
        sums = monthly[monthly_cols_].sum()
        trend_df = pd.DataFrame({"월": [c.replace(" 대여 건수","") for c in monthly_cols_], "대여건수": sums.values})
        fig_month = px.bar(trend_df, x="월", y="대여건수",
                           text="대여건수",
                           color_discrete_sequence=[COLORS["primary"]])
        fig_month.update_traces(textposition="outside")
        fig_month.update_layout(title="월별 총 대여 추세", yaxis_title="대여 건수", xaxis_title="월", template="plotly_white")
        with colB:
            st.plotly_chart(fig_month, use_container_width=True)
    else:
        with colB:
            st.info("월별 컬럼(4~7월)이 없어 추세 그래프를 생략합니다.")

    st.markdown("---")

    # -----------------------------
    # 지도 버블 맵 (총 대여 건수)
    # -----------------------------
    st.subheader("지도 버블 맵 — 버블 크기=총 대여 건수, 색상=권역")

    # monthly에 좌표가 없으면 stations로부터 조인
    has_latlon_monthly = "위도" in monthly.columns and "경도" in monthly.columns
    if not has_latlon_monthly:
        latlon_map = stations.set_index(join_station_col)[["위도","경도"]].to_dict("index")
        monthly["위도"] = monthly[join_monthly_col].astype(str).map(lambda k: latlon_map.get(str(k),{}).get("위도"))
        monthly["경도"] = monthly[join_monthly_col].astype(str).map(lambda k: latlon_map.get(str(k),{}).get("경도"))

    if monthly["위도"].notna().any() and monthly["경도"].notna().any():
        center_lat = monthly["위도"].dropna().mean()
        center_lon = monthly["경도"].dropna().mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="OpenStreetMap")

        for _, r in monthly.iterrows():
            lat = r.get("위도", None); lon = r.get("경도", None)
            if pd.notnull(lat) and pd.notnull(lon):
                total = r.get("총 대여 건수", 0)
                st_id = str(r.get(join_monthly_col,""))
                region = id_to_region.get(st_id, "기타")
                name = r.get("대여소 이름", st_id)
                radius = max(6, (float(total)**0.5)*0.9 if pd.notnull(total) else 6)
                color = COLORS["secondary"] if region != "기타" else COLORS["primary"]
                folium.CircleMarker(
                    location=[lat,lon],
                    radius=radius,
                    color=color, fill=True, fill_color=color, fill_opacity=0.35,
                    popup=folium.Popup(f"<b>{name}</b><br>총 대여 건수: {int(total) if pd.notnull(total) else 0:,}<br>권역: {region}", max_width=260)
                ).add_to(m)
        st_folium(m, width=None, height=520)
    else:
        st.info("📍 월별/대여소현황 어디에도 위도/경도 컬럼이 없어 지도를 표시할 수 없습니다.")

    st.markdown("---")

    # -----------------------------
    # 시간대별 이용량 (평일 vs 주말) — 수치 레이블
    # -----------------------------
    st.subheader("시간대별 이용량 (평일 vs 주말)")
    mode = st.radio("보기", options=["평일","주말"], horizontal=True, index=0)
    if time_col in usage.columns:
        usage["_hour_"] = pd.to_datetime(usage[time_col], errors="coerce").dt.hour
        usage["_is_weekend_"] = pd.to_datetime(usage[time_col], errors="coerce").dt.dayofweek.isin([5,6])
        tmp = usage[usage["_is_weekend_"]] if mode=="주말" else usage[~usage["_is_weekend_"]]
        hourly = tmp.groupby("_hour_").size().reindex(range(24), fill_value=0).reset_index()
        hourly.columns = ["hour","count"]

        fig_hour = go.Figure()
        fig_hour.add_trace(go.Scatter(
            x=hourly["hour"], y=hourly["count"],
            mode="lines+markers+text",
            text=[str(int(v)) for v in hourly["count"]],
            textposition="top center",
            name=mode,
            line=dict(color=COLORS["secondary"] if mode=="평일" else COLORS["alert"])
        ))
        fig_hour.update_layout(xaxis=dict(dtick=1,title="시각(0~23)"), yaxis_title="대여 건수", height=420, template="plotly_white")
        st.plotly_chart(fig_hour, use_container_width=True)
    else:
        st.info("시간 컬럼을 찾을 수 없습니다. 사이드바에서 시간 컬럼을 지정해주세요. (예: 시작대여일자)")

    st.markdown("---")

    # -----------------------------
    # 구간 비중 (출근/점심/퇴근/야간) — 누적막대 수치 레이블
    # -----------------------------
    st.subheader("구간 비중 (출근/점심/퇴근/야간) — 평일 vs 주말")
    if time_col in usage.columns:
        u2 = usage.copy()
        u2["_dt_"] = pd.to_datetime(u2[time_col], errors="coerce")
        u2["hour"] = u2["_dt_"].dt.hour

        def seg(h):
            if pd.isna(h): return "야간"
            h = int(h)
            if 6 <= h < 9: return "출근"
            if 11 <= h <14: return "점심"
            if 17 <= h <20: return "퇴근"
            return "야간"

        u2["seg"] = u2["hour"].apply(seg)
        u2["is_weekend"] = u2["_dt_"].dt.dayofweek.isin([5,6]).map({True:"주말", False:"평일"})
        blocks = u2.groupby(["is_weekend","seg"]).size().reset_index(name="count")
        blocks = blocks.pivot(index="seg", columns="is_weekend", values="count").fillna(0).reindex(["출근","점심","퇴근","야간"])

        fig_blk = go.Figure()
        fig_blk.add_bar(
            name="평일", x=blocks.index,
            y=blocks["평일"] if "평일" in blocks else [0]*len(blocks),
            marker=dict(color=COLORS["secondary"]),
            text=(blocks["평일"] if "평일" in blocks else [0]*len(blocks)),
            textposition="inside",
            offsetgroup="blk_wd"
        )
        fig_blk.add_bar(
            name="주말", x=blocks.index,
            y=blocks["주말"] if "주말" in blocks else [0]*len(blocks),
            marker=dict(color=COLORS["alert"]),
            text=(blocks["주말"] if "주말" in blocks else [0]*len(blocks)),
            textposition="inside",
            offsetgroup="blk_we"
        )
        fig_blk.update_layout(barmode="stack", yaxis_title="건수", template="plotly_white")
        st.plotly_chart(fig_blk, use_container_width=True)
    else:
        st.info("시간 컬럼을 찾을 수 없습니다. 사이드바에서 시간 컬럼을 지정해주세요. (예: 시작대여일자)")

    st.markdown("---")

    # -----------------------------
    # 대여소 순위 (TOP 10 / BOTTOM 10) — 수치 레이블
    # -----------------------------
    st.subheader("대여소 순위 — TOP 10 / BOTTOM 10")
    if "총 대여 건수" in monthly.columns:
        sorted_m = monthly.sort_values("총 대여 건수", ascending=False)
        top10 = sorted_m.head(10)
        bot10 = sorted_m.tail(10)

        labels = list(top10["대여소 이름"].fillna(top10[join_monthly_col])) + \
                 list(bot10["대여소 이름"].fillna(bot10[join_monthly_col]))
        values = list(top10["총 대여 건수"]) + list(bot10["총 대여 건수"])
        colors_bars = [COLORS["secondary"]]*len(top10) + [COLORS["primary"]]*len(bot10)

        fig_rank = go.Figure(go.Bar(
            x=values, y=labels, orientation="h",
            marker=dict(color=colors_bars),
            text=[int(v) if pd.notnull(v) else 0 for v in values],
            textposition="outside",
            offsetgroup="rank"
        ))
        fig_rank.update_layout(height=600, xaxis_title="총 대여 건수", template="plotly_white")
        st.plotly_chart(fig_rank, use_container_width=True)
    else:
        st.info("월별 데이터에 '총 대여 건수' 컬럼이 필요합니다.")

    st.markdown("---")

    # -----------------------------
    # 저이용 원인 탐색 — POI_500m vs 총 대여 건수 (툴팁 수치)
    # -----------------------------
    st.subheader("저이용 원인 탐색 — POI_500m vs 총 대여 건수")
    # 다양한 이름 대응
    poi_key = None
    for k in ["POI_500m","poi_count_500m","POI","POI500m","poi_500m"]:
        if k in monthly.columns:
            poi_key = k
            break

    if poi_key and "총 대여 건수" in monthly.columns:
        # 하나의 색상으로 툴팁 강조(수치 레이블은 툴팁으로 제공)
        fig_sc = px.scatter(
            monthly,
            x=poi_key, y="총 대여 건수",
            hover_data={"대여소 이름":True, "대여소번호":True, poi_key:":,", "총 대여 건수":":,"},
            color_discrete_sequence=[COLORS["secondary"]]
        )
        fig_sc.update_traces(
            hovertemplate=(
                "POI_500m=%{x:,}<br>" +
                "총 대여 건수=%{y:,}<br>" +
                "대여소 이름=%{customdata[0]}<br>" +
                "대여소번호=%{customdata[1]}<extra></extra>"
            )
        )
        fig_sc.update_layout(
            xaxis_title="POI_500m",
            yaxis_title="총 대여 건수",
            height=520,
            template="plotly_white"
        )
        st.plotly_chart(fig_sc, use_container_width=True)
    else:
        st.info("산점도 생성을 위해 'POI_500m'(또는 poi_count_500m 등)과 '총 대여 건수' 컬럼이 필요합니다.")

    st.caption("© 2025 어울링 분석 — palette: secondary / primary / tertiary / alert")

if __name__ == "__main__":
    main()
