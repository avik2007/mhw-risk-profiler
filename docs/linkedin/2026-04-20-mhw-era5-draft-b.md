# LinkedIn Draft B — Medium (~600 words)
# Placeholders: [INSERT: ...] — fill in after B3 training completes
# Visuals needed: SVaR map, IG attribution bar chart + seasonal breakdown (from data/results/plots/)
# Review checklist before posting:
#   - [ ] Replace all [INSERT] placeholders with real values
#   - [ ] Confirm spread > 0 in B3 results
#   - [ ] Confirm SVaR_95 > SVaR_50 > SVaR_05
#   - [ ] Physical interpretation of IG attribution fact-checked (seasonal breakdown)
#   - [ ] Repo link added

---

Can atmospheric data predict the financial impact of ocean heatwaves?

Marine heatwaves — periods when SST exceeds the 90th percentile of a 30-year climatological baseline — are increasingly frequent and economically damaging. For aquaculture operators in the Gulf of Maine, a sustained MHW can mean mass mortality events, suppressed growth, and uninsured losses.

The question: can atmospheric ensemble forecasts be translated into spatially explicit financial risk estimates?

**The model**

I trained a 1D-CNN on ERA5 reanalysis (via Google Earth Engine) to predict Stress Degree Days — cumulative thermal load above the Hobday (2016) MHW threshold — at each grid cell across the Gulf of Maine. Inputs: 2m air temperature, 10m wind vectors, mean sea level pressure, and sea surface temperature. Labels derived from a 30-year NOAA OISST v2.1 climatology (1982–2011).

Output: a Synthetic Value-at-Risk (SVaR) map — at the 95th percentile across ensemble members, what is the expected thermal stress loading?

[INSERT: SVaR map — Gulf of Maine, SVaR_95 in °C·day, colorbar labeled]

[INSERT: 2-sentence spatial interpretation — e.g. "Elevated risk concentrates in the western GoM, consistent with warm-core ring intrusions from the Gulf Stream. Eastern GoM risk is buffered by deeper mixed layers."]

**What drives the risk signal?**

Using Integrated Gradients (Sundararajan et al., 2017), I attributed each model prediction back to its input variables. IG answers: given this prediction, which inputs were responsible?

[INSERT: IG attribution bar chart — mean |IG| by variable, all grid cells, annual + seasonal breakdown]

[INSERT: 3-4 sentence attribution interpretation — variable rankings, percentages, seasonal contrast. E.g.: "SST dominates at ~X%. 2m air temperature ranks second — atmospheric forcing of the ocean surface. Wind contributions peak in [season], consistent with wind-driven mixing controlling heat accumulation."]

The seasonal breakdown is where it gets physically interesting: [INSERT: key seasonal finding — e.g. "summer is SST-dominated; winter shows stronger wind influence, consistent with deeper winter mixed layers"].

**Limitations and what's next**

This run uses ERA5 — deterministic reanalysis — with synthetic Gaussian ensemble perturbations to simulate spread. The IG attributions reflect real atmospheric physics. The probabilistic spread does not: all synthetic members cross the MHW threshold on essentially the same days, so confidence intervals on the SVaR are artificial.

The fix is already in motion: Google WeatherNext 2 provides 64 genuine ensemble members initialized from real atmospheric uncertainty. ERA5 tells us *which variables matter*. WN2 will tell us *how uncertain that estimate is* — the difference between a risk estimate and a risk distribution.

That post is coming soon.

Code: [INSERT: repo link]
Data: NOAA OISST v2.1, HYCOM GLBv0.08, ERA5 + WeatherNext 2 via Google Earth Engine

#MarineHeatwaves #XAI #IntegratedGradients #ClimateRisk #MachineLearning #EarthEngine #Aquaculture
