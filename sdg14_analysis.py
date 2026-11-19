import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import r2_score

#Import
df = pd.read_csv("ocean_plastic_clean.csv")


#Add data
r = requests.get("https://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL",
    params={"format": "json", "date": "2015:2024", "per_page": 20000},
    timeout=60,
)

r.raise_for_status()

pop = pd.DataFrame(
    [
        {"iso3": r["countryiso3code"], "year": int(r["date"]), "population": r["value"]}
        for r in r.json()[1]
        if r['value'] is not None and r['countryiso3code']
    ]
)

df = df.merge(pop, on=["iso3", "year"], how="left")

#Build analysis variables 

d = df.dropna(subset=["beach_litter_tonnes", "gdp_per_capita", "urban_pop_pct", "population"]).copy()

d = d[d["beach_litter_tonnes"] > 0]

d["litter_per_1000"] = d["beach_litter_tonnes"] / d["population"] * 1000

d['ln_litter_pc'] = np.log(d['litter_per_1000'])
d['ln_gdp_pc'] = np.log(d['gdp_per_capita'])

print(f"Analysis sample: {len(d)} rows, {d['iso3'].nunique()} countries, years {d['year'].min()}-{d['year'].max()}")

#Regression analysis
ols = smf.ols("ln_litter_pc ~ ln_gdp_pc + urban_pop_pct + year", data=d).fit(
    cov_type="cluster", cov_kwds={"groups": d["iso3"]}
)

print("\nOLS regression results:")
print(ols.summary().tables[1])

b = ols.params['ln_gdp_pc']
print(f"\nInterpretation: a 10% higher GDP per capita goes with about {b * 10:.1f}% "
      f"{'more' if b > 0 else 'less'} litter per person (correlation, not causation).")

#Random Forest analysis

d["chl"] = d["chlorophyll_anomaly_pct"].fillna(d["chlorophyll_anomaly_pct"].median())
features = ["ln_gdp_pc", "urban_pop_pct", "year", "chl"]
rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
n_groups = min(5, d["iso3"].nunique())
pred = cross_val_predict(rf, d[features], d["ln_litter_pc"], groups=d["iso3"], cv=GroupKFold(n_groups))
print(f"\nRANDOM FOREST out-of-sample R2 (unseen countries): {r2_score(d['ln_litter_pc'], pred):.2f}")
rf.fit(d[features], d["ln_litter_pc"])
print(pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False).round(3))

#Chart 1
trend = d.groupby("year")["litter_per_1000"].median()
plt.figure(figsize=(7, 4))
plt.plot(trend.index, trend.values, marker="o")
plt.title("Median land-based beach litter per 1,000 people (2015-2024)")
plt.xlabel("Year"); plt.ylabel("Tonnes per 1,000 people")
plt.tight_layout(); plt.savefig("chart_trend.png", dpi=150); plt.close()

#Chart 2
plt.figure(figsize=(7, 4))
plt.scatter(d["ln_gdp_pc"], d["ln_litter_pc"], s=8,alpha=0.4)
xs = np.linspace(d["ln_gdp_pc"].min(), d["ln_gdp_pc"].max(), 50)
plt.plot(xs, ols.params["Intercept"] + b * xs + ols.params["urban_pop_pct"] * d["urban_pop_pct"].mean()
         + ols.params["year"] * d["year"].mean(), color="red")
plt.title("GDP per capita vs. beach litter per person (logs)")
plt.xlabel("ln(GDP per capita)"); plt.ylabel("ln(tonnes per 1,000 people)")
plt.tight_layout(); plt.savefig("chart_scatter.png", dpi=150); plt.close()

#Export 
d["income_group"] = pd.cut(d["gdp_per_capita"], [0, 1500, 4500, 14000, 1e9],
                           labels=["Low", "Lower-middle", "Upper-middle", "High"])
d.drop(columns=["chl"]).to_csv("dashboard_data.csv", index=False)
print("\nSaved dashboard_data.csv, chart_trend.png, chart_scatter.png")
