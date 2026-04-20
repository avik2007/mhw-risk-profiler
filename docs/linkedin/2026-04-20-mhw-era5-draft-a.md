# LinkedIn Draft A — Short (~300 words)
# Placeholders: [INSERT: ...] — fill in after B3 training completes
# Visuals needed: SVaR map, IG attribution bar chart (from data/results/plots/)
# Review checklist before posting:
#   - [ ] Replace all [INSERT] placeholders with real values
#   - [ ] Confirm spread > 0 in B3 results
#   - [ ] Confirm SVaR_95 > SVaR_50 > SVaR_05
#   - [ ] Physical interpretation of IG attribution fact-checked
#   - [ ] Repo link added

---

Can atmospheric data predict the financial impact of ocean heatwaves?

I trained a 1D-CNN on ERA5 reanalysis data — accessed via Google Earth Engine — to estimate Stress Degree Days across the Gulf of Maine: cumulative thermal load above the Hobday (2016) marine heatwave threshold. The output is a spatial Value-at-Risk map for aquaculture assets.

[INSERT: SVaR map — Gulf of Maine grid, colored by SVaR_95 in °C·day]

[INSERT: 1-sentence description of spatial pattern — e.g. "Risk concentrates in the western GoM, consistent with documented warm-core ring intrusions from the Gulf Stream."]

The more interesting output is the Integrated Gradients attribution — which atmospheric variables actually drive the risk signal?

[INSERT: IG attribution bar chart — variables ranked by mean |IG|, annual mean]

[INSERT: 2-sentence interpretation — e.g. "SST dominates at ~X%, followed by 2m air temperature at ~Y%. Wind components contribute primarily in [season], consistent with wind-driven mixing."]

This is an ERA5 baseline: real atmospheric physics, synthetic ensemble spread. The attributions are valid. The probabilistic uncertainty isn't — yet.

Next: Google WeatherNext 2. Sixty-four genuine ensemble members initialized from real atmospheric uncertainty. ERA5 tells us *which variables matter*. WN2 will tell us *how uncertain that is*.

Code: [INSERT: repo link]
Data: NOAA OISST v2.1 (1982–2011), HYCOM GLBv0.08, ERA5 via GEE

#MarineHeatwaves #XAI #ClimateRisk #MachineLearning #EarthEngine
