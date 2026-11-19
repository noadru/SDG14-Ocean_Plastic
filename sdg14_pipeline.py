
import requests
import pandas as pd
import pycountry

START_YEAR = 2015
END_YEAR = 2024

UN_SERIES = {
    "EN_MAR_BEALITSQ": "beach_litter_per_km2",
    "EN_MAR_BEALIT_BV": "beach_litter_tonnes",
    "EN_MAR_PLASDD": "floating_plastic_per_km2",
    "EN_MAR_CHLANM": "chlorophyll_anomaly_pct",
}

WB_SERIES = {
    "NY.GDP.PCAP.CD": "gdp_per_capita",
    "SP.URB.TOTL.IN.ZS": "urban_pop_pct",
}

def get_un_series(code, friendly_name):
    url = "https://unstats.un.org/sdgs/UNSDGAPIV5/v1/sdg/Series/Data"
    rows = []
    page = 1
    while True:
        params = {
            "seriesCode": code,
            "timePeriodStart": START_YEAR,
            "timePeriodEnd": END_YEAR,
            "pageSize": 1000,
            "page": page,
        }
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        rows.extend(payload.get("data", []))
        total = payload.get("totalElements", 0)
        if len(rows) >= total or not payload.get("data"):
            break
        page += 1

    if not rows:
        print(f"WARNING: no data returned for {code}")
        return pd.DataFrame(columns=["iso3", "year", friendly_name])

    df = pd.DataFrame(rows)
    df = df[["geoAreaCode", "geoAreaName", "timePeriodStart", "value"]].copy()
    df.columns = ["m49", "country", "year", friendly_name]
    df[friendly_name] = pd.to_numeric(df[friendly_name], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["iso3"] = df["m49"].apply(m49_to_iso3)
    df = df.dropna(subset=["iso3", friendly_name])
    df = df.groupby(["iso3", "country", "year"], as_index=False)[friendly_name].mean()
    return df


def m49_to_iso3(code):
    try:
        return pycountry.countries.get(numeric=str(int(code)).zfill(3)).alpha_3
    except Exception:
        return None

def get_wb_series(code, friendly_name):
    url = f"https://api.worldbank.org/v2/country/all/indicator/{code}"
    params = {"format": "json", "date": f"{START_YEAR}:{END_YEAR}", "per_page": 20000}
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()[1]
    df = pd.DataFrame(
        [
            {"iso3": r["countryiso3code"], "year": int(r["date"]), friendly_name: r["value"]}
            for r in data
        ]
    )
    df = df.dropna(subset=[friendly_name])
    df = df[df["iso3"] != ""]
    df["year"] = df["year"].astype("Int64")
    return df



if __name__ == "__main__":
    un_tables = []
    for code, name in UN_SERIES.items():
        print(f"Downloading UN series {code} ...")
        t = get_un_series(code, name)
        print(f"   -> {len(t)} country-year rows, {t['iso3'].nunique()} countries")
        un_tables.append(t)

    merged = un_tables[0]
    for t in un_tables[1:]:
        merged = merged.merge(t.drop(columns=["country"]), on=["iso3", "year"], how="outer")

    for code, name in WB_SERIES.items():
        print(f"Downloading World Bank indicator {code} ...")
        t = get_wb_series(code, name)
        merged = merged.merge(t, on=["iso3", "year"], how="left")

    merged = merged.sort_values(["iso3", "year"]).reset_index(drop=True)
    merged.to_csv("ocean_plastic_clean.csv", index=False)

    print("\nNon-empty values per column:")
    print(merged.count())
    print("\nPreview:")
    print(merged.head(10))
    print("\nSaved ocean_plastic_clean.csv")