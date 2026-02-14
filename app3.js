// app.js – Kishan HAL full prototype
document.addEventListener("DOMContentLoaded", () => {
  /* --------------------------------------------------
   *  GLOBAL DATA
   * -------------------------------------------------- */
  const crops = [
    {
      id: "wheat",
      name: "Wheat",
      season: "Rabi",
      water: "Medium",
      waterIndex: 2,
      profitIndex: 7,
      durationDays: 120,
      baseYieldQtlPerAcre: 18,
      basePricePerQtl: 2500,
      npk: { N: 120, P: 60, K: 40 },
      tempRange: [15, 25],
    },
    {
      id: "rice",
      name: "Rice (Paddy)",
      season: "Kharif",
      water: "High",
      waterIndex: 3,
      profitIndex: 8,
      durationDays: 135,
      baseYieldQtlPerAcre: 20,
      basePricePerQtl: 2800,
      npk: { N: 150, P: 75, K: 60 },
      tempRange: [22, 32],
    },
    {
      id: "maize",
      name: "Maize",
      season: "Both",
      water: "Medium",
      waterIndex: 2,
      profitIndex: 6,
      durationDays: 110,
      baseYieldQtlPerAcre: 16,
      basePricePerQtl: 2200,
      npk: { N: 100, P: 60, K: 40 },
      tempRange: [18, 30],
    },
    {
      id: "mustard",
      name: "Mustard",
      season: "Rabi",
      water: "Low",
      waterIndex: 1,
      profitIndex: 7,
      durationDays: 115,
      baseYieldQtlPerAcre: 10,
      basePricePerQtl: 5500,
      npk: { N: 80, P: 40, K: 40 },
      tempRange: [10, 22],
    },
    {
      id: "cotton",
      name: "Cotton",
      season: "Kharif",
      water: "High",
      waterIndex: 3,
      profitIndex: 9,
      durationDays: 165,
      baseYieldQtlPerAcre: 8,
      basePricePerQtl: 6500,
      npk: { N: 120, P: 60, K: 60 },
      tempRange: [20, 32],
    },
  ];

  const cropMap = Object.fromEntries(crops.map((c) => [c.id, c]));

  const pestInfoByCrop = {
    wheat: [
      "Rust (Brown/Black): Use resistant varieties, timely fungicide spray.",
      "Aphids: Monitor leaf underside, use neem-based spray first.",
    ],
    rice: [
      "Bacterial Leaf Blight: Avoid late nitrogen, use clean seed.",
      "Stem Borer: Use pheromone traps, maintain field water level properly.",
    ],
    maize: ["Fall Armyworm: Install pheromone traps, early scouting is crucial."],
    mustard: ["Aphids: Yellow sticky traps, need-based insecticide."],
    cotton: [
      "Pink Bollworm: Use pheromone traps, follow recommended sowing dates.",
    ],
  };

  const cultivationTipsByCrop = {
    wheat: [
      "Use certified seeds and proper seed treatment.",
      "Sow at proper depth (4–5 cm) and maintain spacing.",
      "First irrigation at Crown Root Initiation stage (20–25 days).",
    ],
    rice: [
      "Maintain 2–5 cm standing water in field.",
      "Use line transplanting for better aeration and nutrient uptake.",
    ],
    maize: [
      "Avoid waterlogging; ensure good drainage.",
      "Apply nitrogen in split doses for better uptake.",
    ],
    mustard: [
      "Sow on time for your region to escape frost and heat stress.",
      "Keep field weed-free during first 30–40 days.",
    ],
    cotton: [
      "Adopt proper crop rotation to avoid soil fatigue.",
      "Balanced fertilization and timely pest monitoring is crucial.",
    ],
  };

  let lastWeather = null; // store last weather + forecast for alerts & smart selector
  let forecastChart = null;
  let compareChart = null;
  let leafModel = null;
  let mediaStream = null;

  /* --------------------------------------------------
   *  PANEL SWITCHING
   * -------------------------------------------------- */
  const panels = document.querySelectorAll(".panel");
  const navItems = document.querySelectorAll(".nav-item");

  navItems.forEach((btn) => {
    btn.addEventListener("click", () => {
      navItems.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      panels.forEach((p) => p.classList.remove("visible"));
      const tabId = btn.dataset.tab;
      const panel = document.getElementById(tabId);
      if (panel) panel.classList.add("visible");
    });
  });

  /* --------------------------------------------------
   *  POPULATE ALL CROP SELECTS
   * -------------------------------------------------- */
  function populateSelect(id) {
    const el = document.getElementById(id);
    if (!el) return;
    el.innerHTML = crops
      .map((c) => `<option value="${c.id}">${c.name}</option>`)
      .join("");
  }

  [
    "crop",
    "irrigCrop",
    "mandiCrop",
    "crop1",
    "crop2",
    "fertCrop",
    "diseaseCrop",
    "tipCrop",
    "profitCrop",
  ].forEach(populateSelect);

  /* --------------------------------------------------
   *  WEATHER API (Open-Meteo)
   * -------------------------------------------------- */
  async function geocodeCity(city) {
    const url =
      "https://geocoding-api.open-meteo.com/v1/search?name=" +
      encodeURIComponent(city) +
      "&count=1&language=en&format=json";
    const res = await fetch(url);
    const data = await res.json();
    if (!data.results || !data.results.length) {
      throw new Error("City not found");
    }
    const { latitude, longitude, name, country } = data.results[0];
    return { lat: latitude, lon: longitude, label: `${name}, ${country}` };
  }

  async function fetchWeatherForecast(lat, lon) {
    const url =
      "https://api.open-meteo.com/v1/forecast?latitude=" +
      lat +
      "&longitude=" +
      lon +
      "&current_weather=true&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max&timezone=auto";
    const res = await fetch(url);
    return await res.json();
  }

  function updateWeatherUI(label, weatherData) {
    const weatherEl = document.getElementById("weatherResult");
    const dashWeather = document.getElementById("dashWeather");

    if (!weatherData.current_weather) {
      weatherEl.textContent = "Weather data not available.";
      return;
    }

    const cw = weatherData.current_weather;
    const daily = weatherData.daily;
    lastWeather = { label, ...weatherData };

    const html = `
      <strong>${label}</strong><br>
      Temp: ${cw.temperature} °C, Wind: ${cw.windspeed} km/h<br>
      Condition (code): ${cw.weathercode}
    `;
    weatherEl.innerHTML = html;
    dashWeather.textContent = `${cw.temperature} °C`;

    // build chart
    if (daily && daily.time) {
      const ctx = document.getElementById("weatherChart").getContext("2d");
      const labels = daily.time;
      const maxT = daily.temperature_2m_max;
      const minT = daily.temperature_2m_min;

      if (forecastChart) forecastChart.destroy();
      forecastChart = new Chart(ctx, {
        type: "line",
        data: {
          labels,
          datasets: [
            {
              label: "Max Temp (°C)",
              data: maxT,
            },
            {
              label: "Min Temp (°C)",
              data: minT,
            },
          ],
        },
        options: {
          responsive: true,
          scales: {
            y: { beginAtZero: false },
          },
        },
      });
    }
  }

  const weatherForm = document.getElementById("weatherForm");
  weatherForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const city = document.getElementById("city").value.trim();
    if (!city) return;
    const weatherEl = document.getElementById("weatherResult");
    weatherEl.textContent = "Loading weather...";
    try {
      const { lat, lon, label } = await geocodeCity(city);
      const data = await fetchWeatherForecast(lat, lon);
      updateWeatherUI(label, data);
    } catch (err) {
      weatherEl.textContent = "Error getting weather: " + err.message;
    }
  });

  document.getElementById("autoWeather").addEventListener("click", () => {
    const weatherEl = document.getElementById("weatherResult");
    if (!navigator.geolocation) {
      weatherEl.textContent = "Geolocation not supported in this browser.";
      return;
    }
    weatherEl.textContent = "Getting your location...";
    navigator.geolocation.getCurrentPosition(
      async (pos) => {
        try {
          const { latitude, longitude } = pos.coords;
          const data = await fetchWeatherForecast(latitude, longitude);
          updateWeatherUI("Your location", data);
        } catch (err) {
          weatherEl.textContent = "Error: " + err.message;
        }
      },
      (err) => {
        weatherEl.textContent = "Location error: " + err.message;
      }
    );
  });

  /* --------------------------------------------------
   *  SOIL TEST & FERTILIZER RECOMMENDATION
   * -------------------------------------------------- */
  function classifyPH(ph) {
    if (ph < 5.5) return "Strongly acidic";
    if (ph < 6.5) return "Slightly acidic (good for many crops)";
    if (ph <= 7.5) return "Neutral / Slightly alkaline (good)";
    if (ph <= 8.5) return "Alkaline – gypsum / organic matter recommended";
    return "Strongly alkaline – needs reclamation";
  }

  function nutrientStatus(actual, ideal) {
    const ratio = actual / ideal;
    if (ratio < 0.7) return "Low";
    if (ratio <= 1.2) return "Optimal";
    return "High";
  }

  const soilForm = document.getElementById("soilForm");
  soilForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const cropId = document.getElementById("crop").value;
    const ph = parseFloat(document.getElementById("ph").value);
    const N = parseFloat(document.getElementById("n").value);
    const P = parseFloat(document.getElementById("p").value);
    const K = parseFloat(document.getElementById("k").value);
    const moisture = parseFloat(document.getElementById("moisture").value);

    const crop = cropMap[cropId];
    const ideal = crop.npk;

    const soilStatus = classifyPH(ph);
    const nStat = nutrientStatus(N, ideal.N);
    const pStat = nutrientStatus(P, ideal.P);
    const kStat = nutrientStatus(K, ideal.K);

    const deficitN = Math.max(0, ideal.N - N);
    const deficitP = Math.max(0, ideal.P - P);
    const deficitK = Math.max(0, ideal.K - K);

    const resultEl = document.getElementById("soilResult");
    resultEl.innerHTML = `
      <h3>Soil Analysis for ${crop.name}</h3>
      <ul>
        <li><strong>pH:</strong> ${ph} (${soilStatus})</li>
        <li><strong>N:</strong> ${N} mg/kg (${nStat})</li>
        <li><strong>P:</strong> ${P} mg/kg (${pStat})</li>
        <li><strong>K:</strong> ${K} mg/kg (${kStat})</li>
        <li><strong>Moisture:</strong> ${moisture}%</li>
      </ul>
      <h4>Fertilizer Recommendation (per acre, rough prototype)</h4>
      <p>
        Additional N: <strong>${deficitN.toFixed(0)} kg</strong><br>
        Additional P₂O₅: <strong>${deficitP.toFixed(0)} kg</strong><br>
        Additional K₂O: <strong>${deficitK.toFixed(0)} kg</strong>
      </p>
      <p><em>Split nitrogen in 2–3 doses; apply P & K as basal.</em></p>
    `;

    // Dashboard soil summary
    document.getElementById("dashSoil").textContent = `${soilStatus}, N:${nStat}, P:${pStat}, K:${kStat}`;
  });

  /* --------------------------------------------------
   *  IRRIGATION ADVICE (simple rules)
   * -------------------------------------------------- */
  const irrigationForm = document.getElementById("irrigationForm");
  irrigationForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const cropId = document.getElementById("irrigCrop").value;
    const soilType = document.getElementById("soilType").value;
    const stage = document.getElementById("stage").value;
    const crop = cropMap[cropId];

    // base interval (days) by stage
    const stageBase = {
      initial: 6,
      vegetative: 5,
      flowering: 4,
      maturity: 7,
    }[stage];

    // soil adjustment
    const soilFactor = soilType === "sandy" ? 0.7 : soilType === "clay" ? 1.3 : 1.0;
    const waterNeedFactor =
      crop.waterIndex === 3 ? 0.9 : crop.waterIndex === 1 ? 1.2 : 1.0;

    let interval = stageBase * soilFactor * waterNeedFactor;
    interval = Math.max(2, Math.round(interval));

    const waterDepthMm =
      crop.waterIndex === 3 ? 60 : crop.waterIndex === 2 ? 50 : 40;

    const result = document.getElementById("irrigationResult");
    result.innerHTML = `
      <h3>Irrigation Plan for ${crop.name}</h3>
      <p>
        ➤ <strong>Stage:</strong> ${stage.toUpperCase()}<br>
        ➤ <strong>Soil:</strong> ${soilType.toUpperCase()}
      </p>
      <p>
        Irrigate every <strong>${interval} days</strong> with
        around <strong>${waterDepthMm} mm</strong> water depth
        (≈ ${waterDepthMm * 27} litres/acre).
      </p>
      <p><em>Adjust interval based on actual rainfall and soil moisture.</em></p>
    `;
  });

  /* --------------------------------------------------
   *  DISEASE CHECK (camera + heuristic AI)
   * -------------------------------------------------- */
  const camera = document.getElementById("camera");
  const snapshotCanvas = document.getElementById("snapshot");
  const openCamBtn = document.getElementById("openCam");
  const captureBtn = document.getElementById("capture");
  const closeCamBtn = document.getElementById("closeCam");
  const leafInput = document.getElementById("leafInput");
  const diseaseResult = document.getElementById("diseaseResult");
  const clearDiseaseBtn = document.getElementById("clearDisease");

  // optional model loader (replace URL with your own hosted TF model)
  async function loadLeafModel() {
    try {
      // Example placeholder: host your model at /leaf-model/model.json
      leafModel = await tf.loadLayersModel("/leaf-model/model.json");
      console.log("Leaf model loaded.");
    } catch (err) {
      console.warn("Leaf model not loaded, using heuristic only.", err);
    }
  }
  loadLeafModel(); // fire and forget

  openCamBtn.addEventListener("click", async () => {
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
      camera.srcObject = mediaStream;
      camera.classList.remove("hidden");
      captureBtn.classList.remove("hidden");
      closeCamBtn.classList.remove("hidden");
    } catch (err) {
      alert("Unable to access camera: " + err.message);
    }
  });

  captureBtn.addEventListener("click", () => {
    const ctx = snapshotCanvas.getContext("2d");
    snapshotCanvas.width = camera.videoWidth;
    snapshotCanvas.height = camera.videoHeight;
    ctx.drawImage(camera, 0, 0);
    snapshotCanvas.classList.remove("hidden");
  });

  closeCamBtn.addEventListener("click", () => {
    if (mediaStream) {
      mediaStream.getTracks().forEach((t) => t.stop());
    }
    camera.classList.add("hidden");
    captureBtn.classList.add("hidden");
    closeCamBtn.classList.add("hidden");
  });

  clearDiseaseBtn.addEventListener("click", () => {
    diseaseResult.textContent = "";
    snapshotCanvas.classList.add("hidden");
    leafInput.value = "";
  });

  async function heuristicLeafAnalysis(img) {
    // Simple heuristic: analyze average green value
    const tmpCanvas = document.createElement("canvas");
    const ctx = tmpCanvas.getContext("2d");
    tmpCanvas.width = 128;
    tmpCanvas.height = 128;
    ctx.drawImage(img, 0, 0, 128, 128);

    const imgData = ctx.getImageData(0, 0, 128, 128).data;
    let sumG = 0;
    let sumR = 0;
    let sumB = 0;
    for (let i = 0; i < imgData.length; i += 4) {
      sumR += imgData[i];
      sumG += imgData[i + 1];
      sumB += imgData[i + 2];
    }
    const pixels = imgData.length / 4;
    const avgR = sumR / pixels;
    const avgG = sumG / pixels;
    const avgB = sumB / pixels;

    if (avgG < avgR && avgG < avgB) {
      return {
        label: "Likely nutrient deficiency / yellowing",
        advice:
          "Leaves look less green. Check nitrogen level and apply balanced fertilizer.",
      };
    } else if (avgR > 130) {
      return {
        label: "Possible rust / spot disease",
        advice: "Check for rust spots; consider preventive fungicide.",
      };
    } else {
      return {
        label: "Leaf appears mostly healthy",
        advice:
          "Continue regular monitoring. Ensure proper irrigation and nutrition.",
      };
    }
  }

  async function runLeafAnalysisFromElement(img) {
    diseaseResult.textContent = "Analyzing leaf...";
    let finalResult;
    try {
      if (leafModel) {
        // Very simple demo: use the heuristic anyway, or write your own preprocessing
        finalResult = await heuristicLeafAnalysis(img);
        finalResult.label += " (model available - hook your pipeline)";
      } else {
        finalResult = await heuristicLeafAnalysis(img);
      }
    } catch (err) {
      finalResult = {
        label: "Error while analyzing",
        advice: err.message,
      };
    }
    diseaseResult.innerHTML = `<strong>${finalResult.label}</strong><br>${finalResult.advice}`;
  }

  document.getElementById("analyzeLeaf").addEventListener("click", () => {
    // if file chosen
    if (leafInput.files && leafInput.files[0]) {
      const file = leafInput.files[0];
      const img = new Image();
      img.onload = () => runLeafAnalysisFromElement(img);
      img.src = URL.createObjectURL(file);
    } else if (!snapshotCanvas.classList.contains("hidden")) {
      const img = new Image();
      img.onload = () => runLeafAnalysisFromElement(img);
      img.src = snapshotCanvas.toDataURL();
    } else {
      alert("Upload a leaf image or capture from camera first.");
    }
  });

  /* --------------------------------------------------
   *  MANDI PRICES (mocked prototype)
   * -------------------------------------------------- */
  async function fetchMandiPrices(cropId) {
    // Simulate async API call
    await new Promise((r) => setTimeout(r, 400));
    const mock = {
      wheat: [
        { market: "Delhi", min: 2300, max: 2500, modal: 2400 },
        { market: "Kanpur", min: 2250, max: 2450, modal: 2350 },
      ],
      rice: [
        { market: "Patna", min: 2600, max: 3000, modal: 2800 },
        { market: "Kolkata", min: 2650, max: 3050, modal: 2900 },
      ],
      maize: [
        { market: "Indore", min: 2000, max: 2300, modal: 2150 },
        { market: "Nagpur", min: 2050, max: 2350, modal: 2200 },
      ],
      mustard: [
        { market: "Jaipur", min: 5000, max: 6000, modal: 5500 },
        { market: "Alwar", min: 5100, max: 6100, modal: 5600 },
      ],
      cotton: [
        { market: "Surat", min: 6200, max: 7000, modal: 6600 },
        { market: "Aurangabad", min: 6100, max: 6900, modal: 6500 },
      ],
    };
    return mock[cropId] || [];
  }

  const mandiForm = document.getElementById("mandiForm");
  mandiForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const cropId = document.getElementById("mandiCrop").value;
    const crop = cropMap[cropId];
    const resultEl = document.getElementById("mandiResult");
    resultEl.textContent = "Loading sample mandi data...";
    const rows = await fetchMandiPrices(cropId);
    if (!rows.length) {
      resultEl.textContent = "No data available (mock).";
      return;
    }
    const table = `
      <h3>${crop.name} – Sample Mandi Prices</h3>
      <table>
        <thead>
          <tr>
            <th>Market</th><th>Min</th><th>Max</th><th>Modal</th>
          </tr>
        </thead>
        <tbody>
          ${rows
            .map(
              (r) => `
            <tr>
              <td>${r.market}</td>
              <td>₹${r.min}</td>
              <td>₹${r.max}</td>
              <td>₹${r.modal}</td>
            </tr>`
            )
            .join("")}
        </tbody>
      </table>
    `;
    resultEl.innerHTML = table;
  });

  /* --------------------------------------------------
   *  AI RECOMMENDATIONS (combine soil + weather + mandi)
   * -------------------------------------------------- */
  function generateAIRecommendations() {
    const lines = [];
    if (lastWeather && lastWeather.current_weather) {
      const t = lastWeather.current_weather.temperature;
      if (t > 32) {
        lines.push(
          "Temperature is high – prefer heat-tolerant crops (e.g., cotton, rice) and irrigate in evening."
        );
      } else if (t < 15) {
        lines.push(
          "Temperature is low – Rabi crops like wheat and mustard are suitable."
        );
      } else {
        lines.push(
          "Temperature is moderate – most crops are fine; focus on balanced nutrition."
        );
      }
    } else {
      lines.push("Get live weather data to improve recommendations.");
    }

    // quick price-based pick
    const bestProfitCrop = [...crops].sort(
      (a, b) =>
        a.baseYieldQtlPerAcre * a.basePricePerQtl <
        b.baseYieldQtlPerAcre * b.basePricePerQtl
          ? 1
          : -1
    )[0];
    lines.push(
      `Based on average price × yield, <strong>${bestProfitCrop.name}</strong> looks highly profitable in many regions.`
    );

    lines.push(
      "Use crop rotation: avoid repeating same crop to reduce disease pressure."
    );
    lines.push(
      "Apply organic matter (FYM/compost) to improve soil structure and water holding."
    );

    return lines;
  }

  document.getElementById("generateAI").addEventListener("click", () => {
    const aiEl = document.getElementById("aiResult");
    const lines = generateAIRecommendations();
    aiEl.innerHTML = "<ul>" + lines.map((l) => `<li>${l}</li>`).join("") + "</ul>";
    document.getElementById("dashAI").innerHTML = lines[0];
  });

  /* --------------------------------------------------
   *  CROP COMPARISON + CHART
   * -------------------------------------------------- */
  const compareForm = document.getElementById("compareForm");
  compareForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const c1Id = document.getElementById("crop1").value;
    const c2Id = document.getElementById("crop2").value;
    const c1 = cropMap[c1Id];
    const c2 = cropMap[c2Id];

    const result = document.getElementById("compareResult");
    result.innerHTML = `
      <h3>${c1.name} vs ${c2.name}</h3>
      <p><strong>Season:</strong> ${c1.season} vs ${c2.season}</p>
      <p><strong>Water Need:</strong> ${c1.water} vs ${c2.water}</p>
      <p><strong>Duration:</strong> ${c1.durationDays} days vs ${c2.durationDays} days</p>
      <p><strong>Profit Index (prototype):</strong> ${c1.profitIndex}/10 vs ${c2.profitIndex}/10</p>
    `;

    // Chart: yield & profit per acre
    const labels = [c1.name, c2.name];
    const yieldArr = [c1.baseYieldQtlPerAcre, c2.baseYieldQtlPerAcre];
    const profitArr = [
      c1.baseYieldQtlPerAcre * c1.basePricePerQtl,
      c2.baseYieldQtlPerAcre * c2.basePricePerQtl,
    ];

    const ctx = document.getElementById("compareChart").getContext("2d");
    if (compareChart) compareChart.destroy();
    compareChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels,
        datasets: [
          { label: "Yield (qtl/acre)", data: yieldArr },
          { label: "Gross income (₹/acre)", data: profitArr },
        ],
      },
      options: {
        responsive: true,
        scales: {
          y: { beginAtZero: true },
        },
      },
    });
  });

  /* --------------------------------------------------
   *  SMART CROP SELECTOR (based on forecast)
   * -------------------------------------------------- */
  const smartForm = document.getElementById("smartForm");
  smartForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const city = document.getElementById("smartCity").value.trim();
    if (!city) return;
    const smartEl = document.getElementById("smartResult");
    smartEl.textContent = "Analyzing area...";
    try {
      const { lat, lon, label } = await geocodeCity(city);
      const data = await fetchWeatherForecast(lat, lon);
      const daily = data.daily;
      if (!daily || !daily.temperature_2m_max) {
        throw new Error("No forecast data");
      }
      // average next 7 days temp
      const temps = daily.temperature_2m_max;
      const avgT =
        temps.reduce((sum, t) => sum + t, 0) / (temps.length || 1);

      const suggestions = crops.filter((c) => {
        const [tmin, tmax] = c.tempRange;
        return avgT >= tmin && avgT <= tmax;
      });

      smartEl.innerHTML = `
        <p>Location: <strong>${label}</strong><br>
        Avg upcoming temperature: <strong>${avgT.toFixed(1)} °C</strong></p>
        <h4>Suggested crops (prototype):</h4>
        <ul>
          ${suggestions
            .map(
              (c) =>
                `<li>${c.name} – Season: ${c.season}, Water: ${c.water}, Duration: ${c.durationDays} days</li>`
            )
            .join("") || "<li>No strong matches, choose based on soil & market.</li>"}
        </ul>
      `;
    } catch (err) {
      smartEl.textContent = "Error: " + err.message;
    }
  });

  /* --------------------------------------------------
   *  FERTILIZER CALCULATOR (per area)
   * -------------------------------------------------- */
  const fertForm = document.getElementById("fertForm");
  fertForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const cropId = document.getElementById("fertCrop").value;
    const area = parseFloat(document.getElementById("area").value);
    const crop = cropMap[cropId];
    const req = crop.npk;

    const totalN = req.N * area;
    const totalP = req.P * area;
    const totalK = req.K * area;

    const fertEl = document.getElementById("fertResult");
    fertEl.innerHTML = `
      <h3>Fertilizer Requirement for ${crop.name}</h3>
      <p>For <strong>${area}</strong> acre(s):</p>
      <ul>
        <li>N: <strong>${totalN.toFixed(0)} kg</strong></li>
        <li>P₂O₅: <strong>${totalP.toFixed(0)} kg</strong></li>
        <li>K₂O: <strong>${totalK.toFixed(0)} kg</strong></li>
      </ul>
      <p><em>Split N; apply full P & K as basal, as per local recommendation.</em></p>
    `;
  });

  /* --------------------------------------------------
   *  PEST INFO & CULTIVATION TIPS
   * -------------------------------------------------- */
  document
    .getElementById("showDisease")
    .addEventListener("click", () => {
      const cropId = document.getElementById("diseaseCrop").value;
      const crop = cropMap[cropId];
      const info = pestInfoByCrop[cropId] || ["No data (prototype)."];
      document.getElementById("diseaseInfo").innerHTML = `
        <h3>Pests & Diseases – ${crop.name}</h3>
        <ul>${info.map((i) => `<li>${i}</li>`).join("")}</ul>
      `;
    });

  document.getElementById("showTip").addEventListener("click", () => {
    const cropId = document.getElementById("tipCrop").value;
    const crop = cropMap[cropId];
    const tips = cultivationTipsByCrop[cropId] || ["No tips (prototype)."];
    document.getElementById("tipResult").innerHTML = `
      <h3>Cultivation Tips – ${crop.name}</h3>
      <ul>${tips.map((t) => `<li>${t}</li>`).join("")}</ul>
    `;
  });

  /* --------------------------------------------------
   *  ALERTS USING LAST WEATHER
   * -------------------------------------------------- */
  document.getElementById("checkAlert").addEventListener("click", () => {
    const alertEl = document.getElementById("alertResult");
    if (!lastWeather || !lastWeather.daily) {
      alertEl.textContent = "Fetch weather first to generate alerts.";
      return;
    }
    const rainProb = lastWeather.daily.precipitation_probability_max[0];
    let msg = "";
    if (rainProb >= 70) {
      msg =
        "High rain probability – risk of fungal diseases in rice & vegetables. Ensure drainage and consider preventive fungicide.";
    } else if (rainProb >= 40) {
      msg =
        "Moderate rain chance – monitor for leaf spots and rust. Avoid spraying just before rain.";
    } else {
      msg =
        "Low rain chance – more risk of sucking pests like aphids/mites, especially under high temperature.";
    }
    alertEl.innerHTML = `<strong>Alert (today):</strong> ${msg}`;
    document.getElementById("dashPest").textContent = msg;
  });

  /* --------------------------------------------------
   *  PROFIT ESTIMATOR
   * -------------------------------------------------- */
  const profitForm = document.getElementById("profitForm");
  profitForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const cropId = document.getElementById("profitCrop").value;
    const area = parseFloat(document.getElementById("profitArea").value);
    const cost = parseFloat(document.getElementById("cost").value);
    const crop = cropMap[cropId];

    const expectedYield = crop.baseYieldQtlPerAcre * area;
    const gross = expectedYield * crop.basePricePerQtl;
    const totalCost = cost * area;
    const net = gross - totalCost;

    const result = document.getElementById("profitResult");
    result.innerHTML = `
      <h3>Profit Estimate – ${crop.name}</h3>
      <p>
        Area: <strong>${area} acre(s)</strong><br>
        Expected yield: <strong>${expectedYield.toFixed(1)} qtl</strong><br>
        Gross income: <strong>₹${gross.toFixed(0)}</strong><br>
        Total cost: <strong>₹${totalCost.toFixed(0)}</strong><br>
        <span style="color:${net >= 0 ? "green" : "red"}">
          Net profit: <strong>₹${net.toFixed(0)}</strong>
        </span>
      </p>
    `;
    document.getElementById("dashProfit").textContent = `₹${net.toFixed(0)}`;
  });

  /* --------------------------------------------------
   *  VOICE ASSISTANT (Web Speech API – prototype)
   * -------------------------------------------------- */
  const startVoice = document.getElementById("startVoice");
  const stopVoice = document.getElementById("stopVoice");
  const voiceResult = document.getElementById("voiceResult");
  const asrLang = document.getElementById("asrLang");

  let recognition = null;
  if ("webkitSpeechRecognition" in window) {
    recognition = new webkitSpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;

    recognition.onresult = (event) => {
      const text = event.results[0][0].transcript;
      voiceResult.innerHTML = `You said: <strong>${text}</strong>`;
      // simple intent detection
      const lower = text.toLowerCase();
      if (lower.includes("weather")) {
        navTo("weather");
      } else if (lower.includes("soil")) {
        navTo("soil");
      } else if (lower.includes("price") || lower.includes("mandi")) {
        navTo("mandi");
      } else if (lower.includes("profit")) {
        navTo("profit");
      }
    };

    recognition.onerror = (e) => {
      voiceResult.textContent = "Error: " + e.error;
    };
  } else {
    voiceResult.textContent =
      "Voice recognition not supported in this browser (use Chrome).";
  }

  function navTo(tabId) {
    panels.forEach((p) => p.classList.remove("visible"));
    navItems.forEach((b) =>
      b.classList.toggle("active", b.dataset.tab === tabId)
    );
    const panel = document.getElementById(tabId);
    if (panel) panel.classList.add("visible");
  }

  startVoice.addEventListener("click", () => {
    if (!recognition) return;
    recognition.lang = asrLang.value;
    recognition.start();
    voiceResult.textContent = "Listening...";
  });

  stopVoice.addEventListener("click", () => {
    if (!recognition) return;
    recognition.stop();
    voiceResult.textContent = "Stopped.";
  });

  /* --------------------------------------------------
   *  SIMPLE RULE-BASED CHATBOT
   * -------------------------------------------------- */
  const chatBox = document.getElementById("chatBox");
  const chatInput = document.getElementById("chatInput");
  const sendChat = document.getElementById("sendChat");
  const micBtn = document.getElementById("micBtn");

  function addChatMessage(sender, text) {
    const div = document.createElement("div");
    div.className = "chat-msg " + sender;
    div.innerHTML = `<strong>${sender === "user" ? "You" : "Bot"}:</strong> ${text}`;
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
  }

  function botReply(msg) {
    const m = msg.toLowerCase();
    if (m.includes("hello") || m.includes("namaste")) {
      return "Namaste! I can help with soil, irrigation, weather, crop planning, etc.";
    }
    if (m.includes("irrigation")) {
      return "For irrigation, select the Irrigation tab and choose crop, soil type and growth stage to get a schedule.";
    }
    if (m.includes("fertilizer")) {
      return "Use the Soil Test or Fertilizer Calc tabs for basic NPK recommendations.";
    }
    if (m.includes("weather")) {
      return "Go to the Weather tab, enter your city or use 'Use My Location' to get live forecast.";
    }
    if (m.includes("profit")) {
      return "Use the Profit tab, select crop, area and cost to estimate profit.";
    }
    return "I am a simple rule-based assistant. Ask about weather, soil, irrigation, fertilizer, profit, or mandi prices.";
  }

  sendChat.addEventListener("click", () => {
    const text = chatInput.value.trim();
    if (!text) return;
    addChatMessage("user", text);
    chatInput.value = "";
    const reply = botReply(text);
    addChatMessage("bot", reply);
  });

  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      sendChat.click();
    }
  });

  // chat mic shares same recognition instance
  micBtn.addEventListener("click", () => {
    if (!recognition) {
      alert("Speech recognition not supported in this browser.");
      return;
    }
    recognition.lang = asrLang.value;
    recognition.onresult = (event) => {
      const text = event.results[0][0].transcript;
      chatInput.value = text;
      sendChat.click();
    };
    recognition.start();
  });

  /* --------------------------------------------------
   *  SIMPLE LOCAL FORUM (localStorage)
   * -------------------------------------------------- */
  const postForm = document.getElementById("postForm");
  const postsList = document.getElementById("postsList");

  function loadPosts() {
    const stored = localStorage.getItem("kishan_forum_posts");
    let posts = [];
    if (stored) {
      try {
        posts = JSON.parse(stored);
      } catch (err) {
        posts = [];
      }
    }
    postsList.innerHTML =
      posts
        .map(
          (p) => `
      <div class="post">
        <strong>${p.name || "Anonymous"}</strong><br>
        <span>${p.text}</span><br>
        <small>${p.time}</small>
      </div>`
        )
        .join("") || "<em>No posts yet. Be the first to ask!</em>";
  }

  function savePost(post) {
    const stored = localStorage.getItem("kishan_forum_posts");
    let posts = [];
    if (stored) {
      try {
        posts = JSON.parse(stored);
      } catch (err) {
        posts = [];
      }
    }
    posts.unshift(post);
    localStorage.setItem("kishan_forum_posts", JSON.stringify(posts));
  }

  postForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const name = document.getElementById("postName").value.trim();
    const text = document.getElementById("postText").value.trim();
    if (!text) return;
    const post = {
      name,
      text,
      time: new Date().toLocaleString(),
    };
    savePost(post);
    document.getElementById("postText").value = "";
    loadPosts();
  });

  loadPosts();
});
