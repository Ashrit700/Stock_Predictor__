// app.js – cleaned, browser-ready, fully working prototype

// =========================================================
// HELPERS + NAV
// =========================================================
const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

// CROPS used in dropdowns
const CROPS = [
  "wheat",
  "rice",
  "cotton",
  "maize",
  "mustard",
  "sugarcane",
  "carrot",
  "brinjal",
  "banana",
  "potato",
];

function setupNavigation() {
  const navItems = $$(".nav-item");
  const panels = $$(".panel");
  navItems.forEach((btn) => {
    btn.addEventListener("click", () => {
      navItems.forEach((b) => b.classList.remove("active"));
      panels.forEach((p) => p.classList.remove("visible"));
      btn.classList.add("active");
      const target = btn.getAttribute("data-tab");
      const panel = document.getElementById(target);
      if (panel) panel.classList.add("visible");
    });
  });
}

function fillDropdowns() {
  const opts = CROPS.map(
    (c) =>
      `<option value="${c}">${c[0].toUpperCase()}${c.slice(1)}</option>`
  ).join("");
  [
    "crop",
    "irrigCrop",
    "mandiCrop",
    "fertCrop",
    "diseaseCrop",
    "tipCrop",
    "profitCrop",
    "crop1",
    "crop2",
  ].forEach((id) => {
    const el = $("#" + id);
    if (el) el.innerHTML = opts;
  });
}

window.addEventListener("DOMContentLoaded", () => {
  setupNavigation();
  fillDropdowns();
  initSoil();
  initIrrigation();
  initDisease();
  initMandi();
  initWeather();
  initAIReco();
  initCompare();
  initSmartSelector();
  initFertCalc();
  initPestInfo();
  initTips();
  initAlerts();
  initProfit();
  initVoiceAssistant();
  initChatbot();
  initForum();
  updateDashboard();
  setInterval(updateDashboard, 10000);
});

// =========================================================
// SOIL
// =========================================================
function initSoil() {
  const baseDose = {
    wheat: { n: 60, p: 30, k: 20 },
    rice: { n: 80, p: 40, k: 20 },
    cotton: { n: 60, p: 30, k: 30 },
    maize: { n: 70, p: 35, k: 25 },
    mustard: { n: 50, p: 25, k: 15 },
    sugarcane: { n: 150, p: 60, k: 60 },
    carrot: { n: 100, p: 50, k: 80 },
    brinjal: { n: 90, p: 60, k: 60 },
    banana: { n: 200, p: 100, k: 200 },
    potato: { n: 120, p: 50, k: 100 },
  };

  const form = $("#soilForm");
  if (!form) return;

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const crop = $("#crop").value;
    const ph = parseFloat($("#ph").value);
    const N = parseFloat($("#n").value);
    const P = parseFloat($("#p").value);
    const K = parseFloat($("#k").value);
    const moisture = parseFloat($("#moisture").value);

    const lines = [];
    lines.push("<h4>🌿 Fertilizer Plan</h4>");

    if (ph < 6 || ph > 7.5) {
      lines.push(`<div>⚠ pH ${ph} out of 6.0–7.5</div>`);
    } else {
      lines.push(`<div>✅ pH ok</div>`);
    }

    if (N < 250) lines.push("<div>🧪 N LOW → add Urea / organic nitrogen</div>");
    else lines.push("<div>N level ok</div>");

    if (P < 25) lines.push("<div>🧪 P LOW → DAP / SSP recommended</div>");
    else lines.push("<div>P level ok</div>");

    if (K < 200) lines.push("<div>🧪 K LOW → MOP recommended</div>");
    else lines.push("<div>K level ok</div>");

    if (moisture < 18)
      lines.push(
        "<div>💧 Moisture low → irrigate lightly before fertilizing</div>"
      );

    const d = baseDose[crop];
    if (d) {
      lines.push(
        `<div><b>Base dose / acre:</b> N ${d.n} kg, P₂O₅ ${d.p} kg, K₂O ${d.k} kg</div>`
      );
    }

    $("#soilResult").innerHTML = lines.join("");
    updateDashboard();
  });
}

// =========================================================
// IRRIGATION
// =========================================================
function initIrrigation() {
  const form = $("#irrigationForm");
  if (!form) return;

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const soil = $("#soilType").value;
    const stage = $("#stage").value;
    const crop = $("#irrigCrop").value;

    let days = { initial: 5, vegetative: 4, flowering: 3, maturity: 6 }[
      stage
    ];
    if (soil === "sandy") days -= 1;
    if (soil === "clay") days += 1;
    if (days < 2) days = 2;

    const method =
      crop === "rice"
        ? "Flooding / Alternate Wetting & Drying"
        : soil === "sandy"
        ? "Drip"
        : soil === "loam"
        ? "Sprinkler / Drip"
        : "Furrow / Drip";

    const mm = soil === "sandy" ? 35 : soil === "loam" ? 45 : 55;

    $("#irrigationResult").innerHTML = `
      <h4>💧 Irrigation Plan</h4>
      <p>🗓 Every <b>${days}</b> days</p>
      <p>🛠 Method: <b>${method}</b></p>
      <p>📏 Depth: <b>${mm} mm</b> (approx)</p>
      <small>Adjust based on rainfall and field condition.</small>
    `;
  });
}

// =========================================================
// DISEASE (simple heuristic + camera)
// =========================================================
function initDisease() {
  const camera = $("#camera");
  const snapshot = $("#snapshot");
  const openCam = $("#openCam");
  const captureBtn = $("#capture");
  const closeCam = $("#closeCam");
  const leafInput = $("#leafInput");
  const diseaseResult = $("#diseaseResult");
  const clearBtn = $("#clearDisease");

  if (!openCam) return;

  let stream = null;

  async function openCamera() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
      });
      camera.srcObject = stream;
      camera.classList.remove("hidden");
      captureBtn.classList.remove("hidden");
      closeCam.classList.remove("hidden");
      openCam.classList.add("hidden");
    } catch (err) {
      alert("Camera not available: " + err.message);
    }
  }

  function closeCamera() {
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
    }
    camera.classList.add("hidden");
    captureBtn.classList.add("hidden");
    closeCam.classList.add("hidden");
    openCam.classList.remove("hidden");
  }

  function capture() {
    const v = camera;
    const c = snapshot;
    const ctx = c.getContext("2d");
    c.width = v.videoWidth || 640;
    c.height = v.videoHeight || 480;
    ctx.drawImage(v, 0, 0);
    c.classList.remove("hidden");
  }

  function clearDisease() {
    leafInput.value = "";
    snapshot.classList.add("hidden");
    diseaseResult.innerHTML = "";
  }

  async function heuristicAnalyze(fileOrCanvas) {
    // works with File or Canvas
    let img = new Image();
    let src;

    if (fileOrCanvas instanceof HTMLCanvasElement) {
      src = fileOrCanvas.toDataURL("image/png");
    } else {
      src = URL.createObjectURL(fileOrCanvas);
    }

    return new Promise((resolve) => {
      img.onload = () => {
        const c = document.createElement("canvas");
        const w = 128;
        const h = (img.height / img.width) * 128;
        c.width = w;
        c.height = h;
        const ctx = c.getContext("2d");
        ctx.drawImage(img, 0, 0, w, h);
        const data = ctx.getImageData(0, 0, w, h).data;
        let sumG = 0;
        let sumR = 0;
        let sumB = 0;
        let darkSpots = 0;
        for (let i = 0; i < data.length; i += 4) {
          const r = data[i];
          const g = data[i + 1];
          const b = data[i + 2];
          sumR += r;
          sumG += g;
          sumB += b;
          if (r < 40 && g < 40 && b < 40) darkSpots++;
        }
        const pixels = data.length / 4;
        const avgR = sumR / pixels;
        const avgG = sumG / pixels;
        const avgB = sumB / pixels;

        let msgs = [];
        if (avgG < (avgR + avgG + avgB) / 3) {
          msgs.push("🟡 Less green → possible nutrient deficiency.");
        }
        if (darkSpots > pixels * 0.02) {
          msgs.push("🔴 Dark spots → possible fungal/bacterial disease.");
        }
        if (!msgs.length) msgs.push("✅ Leaf looks mostly healthy.");
        msgs.push("ℹ Please confirm with a local expert before spraying.");

        resolve(msgs);
        if (!(fileOrCanvas instanceof HTMLCanvasElement)) {
          URL.revokeObjectURL(src);
        }
      };
      img.src = src;
    });
  }

  $("#analyzeLeaf").addEventListener("click", async () => {
    diseaseResult.innerHTML = "🔍 Analyzing...";

    if (leafInput.files && leafInput.files[0]) {
      const msgs = await heuristicAnalyze(leafInput.files[0]);
      diseaseResult.innerHTML = msgs.map((m) => `<div>${m}</div>`).join("");
    } else if (!snapshot.classList.contains("hidden")) {
      const msgs = await heuristicAnalyze(snapshot);
      diseaseResult.innerHTML = msgs.map((m) => `<div>${m}</div>`).join("");
    } else {
      alert("Upload a leaf image or capture from camera first.");
      diseaseResult.innerHTML = "";
    }
  });

  openCam.addEventListener("click", openCamera);
  closeCam.addEventListener("click", closeCamera);
  captureBtn.addEventListener("click", capture);
  clearBtn.addEventListener("click", clearDisease);
}

// =========================================================
// MANDI (sample)
// =========================================================
function initMandi() {
  const sample = {
    wheat: [
      { market: "Kota", price: 2350, grade: "FAQ" },
      { market: "Indore", price: 2425, grade: "A" },
      { market: "Nagpur", price: 2280, grade: "FAQ" },
    ],
    rice: [
      { market: "Karnal", price: 3250, grade: "Basmati" },
      { market: "Raipur", price: 1950, grade: "Common" },
    ],
    cotton: [
      { market: "Rajkot", price: 6700, grade: "Kapas" },
      { market: "Akola", price: 6450, grade: "Kapas" },
    ],
    maize: [
      { market: "Nizamabad", price: 1900, grade: "FAQ" },
      { market: "Davangere", price: 1850, grade: "FAQ" },
    ],
    mustard: [
      { market: "Jaipur", price: 5400, grade: "FAQ" },
      { market: "Alwar", price: 5350, grade: "FAQ" },
    ],
    sugarcane: [
      { market: "Meerut", price: 330, grade: "Common" },
      { market: "Pune", price: 340, grade: "A" },
    ],
    carrot: [
      { market: "Nashik", price: 1200, grade: "Fresh" },
      { market: "Delhi", price: 1400, grade: "Fresh" },
    ],
    brinjal: [
      { market: "Agra", price: 900, grade: "A" },
      { market: "Patna", price: 800, grade: "B" },
    ],
    banana: [
      { market: "Trichy", price: 1200, grade: "Cavendish" },
      { market: "Nagpur", price: 1100, grade: "Local" },
    ],
    potato: [
      { market: "Kanpur", price: 1000, grade: "A" },
      { market: "Indore", price: 950, grade: "B" },
    ],
  };

  const form = $("#mandiForm");
  if (!form) return;

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const crop = $("#mandiCrop").value;
    const data = sample[crop] || [];
    if (!data.length) {
      $("#mandiResult").innerHTML = "<p>No data available (sample only).</p>";
      return;
    }
    const rows = data
      .map(
        (d) =>
          `<tr><td>${d.market}</td><td>₹${d.price}</td><td>${d.grade}</td></tr>`
      )
      .join("");
    $("#mandiResult").innerHTML = `
      <table>
        <thead>
          <tr><th>Market</th><th>Price (₹/quintal)</th><th>Grade</th></tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  });
}

// =========================================================
// WEATHER + CHART (Open-Meteo)
// =========================================================
let weatherChart = null;
let lastWeatherData = null;

function initWeather() {
  const form = $("#weatherForm");
  const wr = $("#weatherResult");
  if (!form) return;

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const city = $("#city").value.trim();
    if (!city) return;
    wr.textContent = "🔄 Fetching...";
    try {
      const geoRes = await fetch(
        `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(
          city
        )}&count=1&language=en&format=json`
      );
      const geo = await geoRes.json();
      if (!geo.results || !geo.results.length) {
        wr.textContent = "❌ City not found";
        return;
      }
      const { latitude, longitude, name, country } = geo.results[0];
      await fetchWeather(latitude, longitude, `${name}, ${country}`);
    } catch (err) {
      wr.textContent = "⚠ Weather error";
    }
  });

  $("#autoWeather").addEventListener("click", () => {
    wr.textContent = "📍 Locating...";
    if (!navigator.geolocation) {
      wr.textContent = "Geolocation not supported.";
      return;
    }
    navigator.geolocation.getCurrentPosition(
      async (pos) => {
        await fetchWeather(
          pos.coords.latitude,
          pos.coords.longitude,
          "Your Location"
        );
      },
      () => {
        wr.textContent = "❌ Location denied";
      }
    );
  });

  async function fetchWeather(lat, lon, place) {
    const api = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=auto`;
    const res = await fetch(api);
    const data = await res.json();
    lastWeatherData = data;
    const c = data.current;
    const d = data.daily;

    wr.innerHTML = `
      <div><b>📍 ${place}</b></div>
      <div>🌡 Temp: ${c.temperature_2m}°C</div>
      <div>💧 Humidity: ${c.relative_humidity_2m}%</div>
      <div>🌬 Wind: ${c.wind_speed_10m} km/h</div>
      <div>🌦 Rain: ${c.precipitation} mm</div>
      <div><small>Updated: ${new Date(c.time).toLocaleString()}</small></div>
    `;

    const labels = d.time.map((t) =>
      new Date(t).toLocaleDateString(undefined, {
        month: "short",
        day: "numeric",
      })
    );
    const maxT = d.temperature_2m_max;
    const minT = d.temperature_2m_min;
    const rain = d.precipitation_sum;

    const ctx = document.getElementById("weatherChart");
    if (!ctx) return;
    if (weatherChart) weatherChart.destroy();

    weatherChart = new Chart(ctx, {
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
          {
            label: "Rain (mm)",
            data: rain,
            type: "bar",
            yAxisID: "y1",
          },
        ],
      },
      options: {
        responsive: true,
        scales: {
          y: { position: "left" },
          y1: {
            position: "right",
            grid: { drawOnChartArea: false },
          },
        },
      },
    });
    updateDashboard();
  }
}

// =========================================================
// AI RECOMMENDATIONS (simple rules)
// =========================================================
function initAIReco() {
  const btn = $("#generateAI");
  if (!btn) return;

  btn.addEventListener("click", () => {
    const out = $("#aiResult");
    const crop = $("#crop")?.value || "wheat";
    const ph = parseFloat($("#ph")?.value || "7");
    const N = parseFloat($("#n")?.value || "300");
    const P = parseFloat($("#p")?.value || "30");
    const K = parseFloat($("#k")?.value || "250");

    let temp = 28;
    let rain = 0;
    let humidity = 60;
    try {
      const w = $("#weatherResult").innerText || "";
      temp = parseFloat(w.match(/Temp: (\d+\.?\d*)/)?.[1] || temp);
      rain = parseFloat(w.match(/Rain: (\d+\.?\d*)/)?.[1] || rain);
      humidity = parseFloat(
        w.match(/Humidity: (\d+\.?\d*)/)?.[1] || humidity
      );
    } catch {
      // ignore
    }

    const tips = [];

    if (ph < 6)
      tips.push(`🌿 Soil acidic (pH ${ph}) → add lime/compost gradually.`);
    else if (ph > 7.5)
      tips.push(
        `🌿 Soil alkaline (pH ${ph}) → add organic manure, avoid excess soda-containing water.`
      );
    else tips.push(`🌿 Soil pH good (${ph}).`);

    if (N < 250) tips.push("🧪 Nitrogen low → apply Urea split in 2–3 doses.");
    if (P < 25)
      tips.push("🧪 Phosphorus low → DAP/SSP as basal dose at sowing.");
    if (K < 200)
      tips.push("🧪 Potassium low → apply MOP, especially for tuber/fruit crops.");

    if (temp > 35)
      tips.push(`☀ High temp (${temp}°C) → irrigate in evening/morning.`);
    if (rain > 10)
      tips.push(`🌧 Rain ${rain}mm → avoid fertilizer just before heavy rain.`);
    if (humidity > 80)
      tips.push(
        `🦠 Humidity ${humidity}% → fungal disease risk, ensure good air movement and avoid wet leaves at night.`
      );

    const score = Math.min(
      100,
      Math.max(
        20,
        Math.round(
          100 - (Math.abs(ph - 7) * 5 + (temp > 35 ? 10 : 0) + (rain > 20 ? 10 : 0))
        )
      )
    );
    tips.push(`<div><b>📈 Crop Health Index:</b> ${score}%</div>`);

    out.innerHTML = tips.map((t) => `<div>${t}</div>`).join("");
    updateDashboard();
  });
}

// =========================================================
// COMPARE
// =========================================================
let compareChart = null;

function initCompare() {
  const form = $("#compareForm");
  if (!form) return;

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const a = $("#crop1").value;
    const b = $("#crop2").value;
    const result = $("#compareResult");
    if (a === b) {
      result.innerHTML = "⚠ Select two different crops.";
      return;
    }

    const data = {
      wheat: { water: 4500, profit: 30000, duration: 120, fertilizer: 110 },
      rice: { water: 7000, profit: 28000, duration: 130, fertilizer: 140 },
      cotton: { water: 5500, profit: 40000, duration: 150, fertilizer: 160 },
      maize: { water: 4000, profit: 32000, duration: 100, fertilizer: 120 },
      mustard: { water: 2500, profit: 25000, duration: 110, fertilizer: 90 },
      sugarcane: { water: 18000, profit: 90000, duration: 300, fertilizer: 250 },
      carrot: { water: 3500, profit: 35000, duration: 90, fertilizer: 80 },
      brinjal: { water: 5000, profit: 50000, duration: 120, fertilizer: 150 },
      banana: { water: 12000, profit: 80000, duration: 300, fertilizer: 200 },
      potato: { water: 4500, profit: 42000, duration: 100, fertilizer: 110 },
    };

    const d1 = data[a];
    const d2 = data[b];

    const waterWin = d1.water < d2.water ? a : b;
    const profitWin = d1.profit > d2.profit ? a : b;
    const fertWin = d1.fertilizer < d2.fertilizer ? a : b;
    const timeWin = d1.duration < d2.duration ? a : b;

    const score1 = (d1.profit / d1.water) * 1000 - d1.fertilizer;
    const score2 = (d2.profit / d2.water) * 1000 - d2.fertilizer;
    const better = score1 > score2 ? a : b;

    result.innerHTML = `
      <table>
        <tr><th>Parameter</th><th>${a}</th><th>${b}</th></tr>
        <tr><td>Water (L/acre)</td><td>${d1.water}</td><td>${d2.water}</td></tr>
        <tr><td>Duration (days)</td><td>${d1.duration}</td><td>${d2.duration}</td></tr>
        <tr><td>Fertilizer (kg/acre)</td><td>${d1.fertilizer}</td><td>${d2.fertilizer}</td></tr>
        <tr><td>Profit (₹/acre)</td><td>${d1.profit}</td><td>${d2.profit}</td></tr>
      </table>
      <p>✅ Low water: <b>${waterWin}</b> • 💹 Profit: <b>${profitWin}</b> • 🧪 Less fert: <b>${fertWin}</b> • ⏱ Faster: <b>${timeWin}</b></p>
      <p class="ok">🌿 AI suggests: <b>${better.toUpperCase()}</b></p>
    `;

    const ctx = document.getElementById("compareChart");
    if (!ctx) return;
    if (compareChart) compareChart.destroy();

    compareChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: [a, b],
        datasets: [
          { label: "Profit (₹/acre)", data: [d1.profit, d2.profit] },
          { label: "Water (L/acre)", data: [d1.water, d2.water] },
        ],
      },
      options: {
        responsive: true,
        scales: { y: { beginAtZero: true } },
      },
    });
  });
}

// =========================================================
// SMART SELECTOR (weather-based)
// =========================================================
function initSmartSelector() {
  const form = $("#smartForm");
  if (!form) return;

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const city = $("#smartCity").value.trim();
    const out = $("#smartResult");
    if (!city) return;
    out.innerHTML = "🔍 Gathering data...";

    try {
      const geoRes = await fetch(
        `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(
          city
        )}&count=1&language=en&format=json`
      );
      const geo = await geoRes.json();
      if (!geo.results || !geo.results.length) {
        out.innerHTML = "❌ City not found.";
        return;
      }
      const { latitude, longitude, name, country } = geo.results[0];
      const wRes = await fetch(
        `https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}&daily=temperature_2m_max,precipitation_sum&timezone=auto`
      );
      const wd = await wRes.json();

      const temps = wd.daily.temperature_2m_max;
      const rains = wd.daily.precipitation_sum;
      const avgTemp =
        temps.reduce((a, b) => a + b, 0) / (temps.length || 1);
      const totalRain = rains.reduce((a, b) => a + b, 0);

      const ph = parseFloat($("#ph")?.value || "7");
      const moisture = parseFloat($("#moisture")?.value || "25");

      const price = {
        wheat: 2400,
        rice: 2300,
        cotton: 6800,
        maize: 2000,
        mustard: 5500,
        sugarcane: 330,
        carrot: 1200,
        brinjal: 900,
        banana: 1100,
        potato: 950,
      };

      const crops = {
        wheat: { temp: [15, 30], rain: [400, 600], fert: 110, water: 4500 },
        rice: { temp: [20, 35], rain: [800, 1200], fert: 140, water: 7000 },
        cotton: { temp: [25, 38], rain: [500, 700], fert: 160, water: 5500 },
        maize: { temp: [18, 35], rain: [400, 600], fert: 120, water: 4000 },
        mustard: { temp: [10, 25], rain: [300, 500], fert: 90, water: 2500 },
        sugarcane: {
          temp: [20, 35],
          rain: [1000, 1500],
          fert: 250,
          water: 18000,
        },
        carrot: { temp: [10, 25], rain: [300, 600], fert: 80, water: 3500 },
        brinjal: { temp: [20, 35], rain: [400, 700], fert: 150, water: 5000 },
        banana: { temp: [20, 35], rain: [900, 1200], fert: 200, water: 12000 },
        potato: { temp: [15, 25], rain: [300, 500], fert: 110, water: 4500 },
      };

      const rows = [];
      let best = null;
      let bestScore = -1;

      Object.entries(crops).forEach(([c, d]) => {
        const tMid = (d.temp[0] + d.temp[1]) / 2;
        const rMid = (d.rain[0] + d.rain[1]) / 2;
        const tScore = 100 - Math.abs(avgTemp - tMid);
        const rScore = 100 - Math.abs(totalRain - rMid) / 10;
        const phScore = 100 - Math.abs(ph - 7) * 10;
        const moistScore = Math.min(100, moisture * 4);
        const econ = (price[c] / d.fert) * 5;
        const sustain = 500 - d.water / 50;
        const total = tScore + rScore + phScore + moistScore + econ + sustain;
        if (total > bestScore) {
          bestScore = total;
          best = c;
        }
        rows.push({
          crop: c,
          total: Math.round(total),
          econ: econ.toFixed(1),
        });
      });

      rows.sort((a, b) => b.total - a.total);

      out.innerHTML = `
        <div>📍 <b>${name}, ${country}</b></div>
        <div>🌡 Avg Temp: ${avgTemp.toFixed(
          1
        )}°C | 🌧 7-day Rain: ${Math.round(totalRain)} mm</div>
        <table>
          <thead><tr><th>Crop</th><th>Suitability</th><th>Economic</th></tr></thead>
          <tbody>
            ${rows
              .slice(0, 5)
              .map(
                (r) =>
                  `<tr><td>${r.crop}</td><td>${r.total}</td><td>${r.econ}</td></tr>`
              )
              .join("")}
          </tbody>
        </table>
        <p class="ok">🌾 Best now: <b>${best.toUpperCase()}</b></p>
      `;
    } catch (err) {
      out.innerHTML = "⚠ Error fetching data.";
    }
  });
}

// =========================================================
// FERTILIZER CALC
// =========================================================
function initFertCalc() {
  const form = $("#fertForm");
  if (!form) return;

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const crop = $("#fertCrop").value;
    const area = parseFloat($("#area").value);
    const out = $("#fertResult");

    const d = {
      wheat: { urea: 120, dap: 60, mop: 30 },
      rice: { urea: 150, dap: 70, mop: 40 },
      maize: { urea: 100, dap: 60, mop: 40 },
      sugarcane: { urea: 300, dap: 120, mop: 100 },
      banana: { urea: 200, dap: 100, mop: 150 },
      potato: { urea: 150, dap: 90, mop: 80 },
      cotton: { urea: 130, dap: 70, mop: 60 },
      mustard: { urea: 80, dap: 40, mop: 30 },
      carrot: { urea: 90, dap: 45, mop: 70 },
      brinjal: { urea: 110, dap: 65, mop: 60 },
    }[crop];

    if (!d) {
      out.innerHTML = "No data for this crop (prototype).";
      return;
    }

    out.innerHTML = `
      <p>🌾 Crop: <b>${crop}</b> • 📏 Area: <b>${area} acre</b></p>
      <p>🧪 Urea: <b>${(d.urea * area).toFixed(
        1
      )} kg</b> • DAP: <b>${(d.dap * area).toFixed(
      1
    )} kg</b> • MOP: <b>${(d.mop * area).toFixed(1)} kg</b></p>
    `;
  });
}

// =========================================================
// PEST INFO
// =========================================================
function initPestInfo() {
  const btn = $("#showDisease");
  if (!btn) return;

  btn.addEventListener("click", () => {
    const crop = $("#diseaseCrop").value;
    const infoMap = {
      wheat: "Rust (yellow/brown). Use Propiconazole 1 ml/L at first symptom.",
      rice: "Blast / BLB – Tricyclazole or Copper oxychloride as per label.",
      maize: "Stem borer / FAW – Emamectin 0.4 g/L, use pheromone traps.",
      banana: "Sigatoka – Mancozeb / Tilt (Propiconazole) 1 ml/L.",
      brinjal: "Fruit & shoot borer – neem oil 5 ml/L or Cypermethrin.",
      potato: "Late blight – Ridomil Gold at 10–12 day interval.",
      sugarcane: "Red rot – rogue infected clumps; Carbendazim seed treatment.",
      carrot: "Alternaria blight – Mancozeb spray every 10 days.",
      cotton: "Bollworm – Spinosad / Emamectin as per recommendation.",
      mustard:
        "Aphids – Imidacloprid 0.3 ml/L; avoid spraying during full flowering.",
    };
    const info = infoMap[crop] || "No data (prototype).";
    $("#diseaseInfo").innerHTML = `<p>${info}</p>`;
  });
}

// =========================================================
// TIPS
// =========================================================
function initTips() {
  const btn = $("#showTip");
  if (!btn) return;

  btn.addEventListener("click", () => {
    const crop = $("#tipCrop").value;
    const tipsMap = {
      wheat: "Sow Nov–Dec; irrigate at tillering & grain filling stages.",
      rice: "Transplant around 25 days; maintain 5 cm water; drain before harvest.",
      sugarcane: "Plant Feb; irrigate every 5–7 days; strip dry leaves.",
      maize: "Sow Jun–Jul; ensure no water stagnation; split nitrogen doses.",
      banana: "Use tissue culture plants; irrigate every 4–5 days; monthly FYM.",
      cotton: "Maintain proper spacing; avoid waterlogging; regular pest scouting.",
      carrot: "Fine seedbed; uniform moisture; avoid excessive late nitrogen.",
      brinjal: "Stake plants; light frequent irrigation; remove infested shoots.",
      mustard: "Sow Oct–Nov; light irrigation at branching & flowering stage.",
      potato: "Cool climate (15–25°C); ridge rows; avoid wet foliage at night.",
    };
    const tips = tipsMap[crop] || "No tips available (prototype).";
    $("#tipResult").innerHTML = `<p>${tips}</p>`;
  });
}

// =========================================================
// ALERTS
// =========================================================
function initAlerts() {
  const btn = $("#checkAlert");
  if (!btn) return;

  btn.addEventListener("click", () => {
    let temp = 30;
    let humidity = 70;
    let rain = 0;

    try {
      const w = $("#weatherResult").innerText;
      temp = parseFloat(w.match(/Temp: (\d+)/)?.[1] || temp);
      humidity = parseFloat(w.match(/Humidity: (\d+)/)?.[1] || humidity);
      rain = parseFloat(w.match(/Rain: (\d+)/)?.[1] || rain);
    } catch {
      // ignore
    }

    let msg = "✅ No major pest alert based on basic weather data.";
    if (humidity > 80 && temp > 25)
      msg =
        "⚠ High humidity & warm temp → fungal disease risk. Use preventive fungicide if needed.";
    else if (rain > 15)
      msg =
        "🌧 Heavy rain → bacterial disease risk in rice; ensure drainage, consider copper fungicide.";
    else if (temp > 35)
      msg = "☀ Heat stress → mites risk; maintain soil moisture, avoid stress.";

    $("#alertResult").innerHTML = `<p>${msg}</p>`;
    updateDashboard();
  });
}

// =========================================================
// PROFIT
// =========================================================
function initProfit() {
  const form = $("#profitForm");
  if (!form) return;

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const crop = $("#profitCrop").value;
    const area = parseFloat($("#profitArea").value);
    const cost = parseFloat($("#cost").value);

    const price = {
      wheat: 2400,
      rice: 2300,
      sugarcane: 330,
      banana: 1100,
      potato: 950,
      maize: 2000,
      cotton: 6800,
      mustard: 5500,
      carrot: 1200,
      brinjal: 900,
    }[crop];

    const yieldA = {
      wheat: 20,
      rice: 22,
      sugarcane: 400,
      banana: 35,
      potato: 30,
      maize: 22,
      cotton: 7,
      mustard: 10,
      carrot: 80,
      brinjal: 120,
    }[crop];

    if (!price || !yieldA) {
      $("#profitResult").innerHTML = "No data (prototype).";
      return;
    }

    const income = price * yieldA * area;
    const totalCost = cost * area;
    const profit = income - totalCost;

    $("#profitResult").innerHTML = `
      <p>🌾 ${crop} • 📏 ${area} acre</p>
      <p>💰 Income: ₹${income.toLocaleString()}</p>
      <p>💸 Cost: ₹${totalCost.toLocaleString()}</p>
      <p class="ok">✅ Estimated Profit: ₹${profit.toLocaleString()}</p>
    `;
    updateDashboard();
  });
}

// =========================================================
// VOICE ASSISTANT
// =========================================================
function initVoiceAssistant() {
  const voiceResult = $("#voiceResult");
  const startBtn = $("#startVoice");
  const stopBtn = $("#stopVoice");
  const langSelect = $("#asrLang");

  if (!startBtn || !voiceResult) return;

  let recognition = null;

  function createRecognition() {
    const SR =
      window.SpeechRecognition || window.webkitSpeechRecognition || null;
    if (!SR) return null;
    return new SR();
  }

  function speak(text, lang) {
    if (!("speechSynthesis" in window)) return;
    const u = new SpeechSynthesisUtterance(text);
    u.lang = lang || "hi-IN";
    window.speechSynthesis.speak(u);
  }

  function reply(q) {
    let r = "माफ कीजिए, कृपया दुबारा पूछें।";

    if (q.includes("खाद") || q.includes("fertilizer"))
      r =
        "Soil tab में NPK भरिए, system आपको DAP/SSP और Urea की सलाह देगा।";
    else if (q.includes("सिंचाई") || q.includes("irrigation"))
      r =
        "Irrigation tab में मिट्टी और stage चुनिए, वहाँ schedule दिखेगा।";
    else if (q.includes("रोग") || q.includes("disease"))
      r = "Disease Check tab में पत्ते की फोटो अपलोड कीजिए।";
    else if (q.includes("मंडी") || q.includes("price"))
      r = "Mandi Prices tab में crop select करके नमूना भाव देख सकते हैं।";

    voiceResult.innerHTML += `<div>🤖 ${r}</div>`;
    speak(r, langSelect.value);
  }

  startBtn.addEventListener("click", () => {
    recognition = createRecognition();
    if (!recognition) {
      voiceResult.textContent =
        "❌ SpeechRecognition not supported in this browser.";
      return;
    }
    recognition.lang = langSelect.value || "hi-IN";
    recognition.continuous = false;
    recognition.interimResults = false;
    voiceResult.textContent = "🎧 Listening...";
    recognition.start();

    recognition.onresult = (e) => {
      const q = e.results[0][0].transcript;
      voiceResult.innerHTML = `🗣 <b>${q}</b>`;
      reply(q.toLowerCase());
    };

    recognition.onerror = (e) => {
      voiceResult.innerHTML = `⚠ ${e.error || "error"}`;
    };
  });

  stopBtn.addEventListener("click", () => {
    try {
      recognition && recognition.stop();
    } catch {
      // ignore
    }
  });
}

// =========================================================
// CHATBOT
// =========================================================
function initChatbot() {
  const chatBox = $("#chatBox");
  const chatInput = $("#chatInput");
  const sendBtn = $("#sendChat");
  const micBtn = $("#micBtn");
  if (!chatBox || !chatInput) return;

  function addMsg(role, text) {
    const d = document.createElement("div");
    d.className = "msg " + role;
    d.textContent = text;
    chatBox.appendChild(d);
    chatBox.scrollTop = chatBox.scrollHeight;
  }

  function speak(text, lang) {
    if (!("speechSynthesis" in window)) return;
    const u = new SpeechSynthesisUtterance(text);
    u.lang = lang || "en-IN";
    speechSynthesis.speak(u);
  }

  function ai(q) {
    const hasHindi = /[ऀ-ॿ]/.test(q);
    const lang = hasHindi ? "hi-IN" : "en-IN";
    let a = "";

    if (q.includes("fertilizer") || q.includes("खाद"))
      a =
        "Use balanced NPK as per soil test – DAP/SSP for P and Urea for N. Check Soil or Fert Calc tab.";
    else if (q.includes("water") || q.includes("सिंचाई") || q.includes("irrigation"))
      a =
        "Irrigate early morning/evening. Frequency depends on soil and crop – see Irrigation tab.";
    else if (q.includes("disease") || q.includes("रोग"))
      a = "Upload a leaf photo in Disease Check to get a quick heuristic analysis.";
    else if (q.includes("weather") || q.includes("मौसम"))
      a =
        "Open the Weather tab and enter your city or use 'Use My Location' for 7-day forecast.";
    else if (q.includes("hello") || q.includes("नमस्ते"))
      a =
        "Namaste किसान! Ask me about fertilizer, irrigation, pest alerts or profit estimation.";
    else
      a =
        "Sorry, I don’t have exact data for that yet. Try asking about fertilizer, irrigation, disease, weather, or profit.";

    addMsg("bot", a);
    speak(a, lang);
  }

  sendBtn.addEventListener("click", () => {
    const q = chatInput.value.trim();
    if (!q) return;
    addMsg("user", q);
    chatInput.value = "";
    ai(q.toLowerCase());
  });

  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      sendBtn.click();
    }
  });

  micBtn.addEventListener("click", () => {
    const SR =
      window.SpeechRecognition || window.webkitSpeechRecognition || null;
    if (!SR) {
      alert("Voice recognition not supported in this browser.");
      return;
    }
    const rec = new SR();
    rec.lang = "hi-IN";
    rec.onresult = (e) => {
      const q = e.results[0][0].transcript;
      addMsg("user", q);
      ai(q.toLowerCase());
    };
    rec.start();
  });
}

// =========================================================
// FORUM (localStorage)
// =========================================================
function initForum() {
  const KEY = "kishan_forum_posts_v1";
  const postForm = $("#postForm");
  const postsList = $("#postsList");
  if (!postForm || !postsList) return;

  function load() {
    let posts = [];
    try {
      posts = JSON.parse(localStorage.getItem(KEY) || "[]");
    } catch {
      posts = [];
    }
    if (!posts.length) {
      postsList.innerHTML = "<em>No posts yet. Be first!</em>";
      return;
    }
    posts.sort((a, b) => b.time - a.time);
    postsList.innerHTML = posts
      .map(
        (p) => `
        <div class="card">
          <strong>${p.name || "Anonymous"}</strong><br>
          ${p.text}<br>
          <small>${new Date(p.time).toLocaleString()}</small>
        </div>
      `
      )
      .join("");
  }

  postForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const name = $("#postName").value.trim();
    const text = $("#postText").value.trim();
    if (!text) return;
    let posts = [];
    try {
      posts = JSON.parse(localStorage.getItem(KEY) || "[]");
    } catch {
      posts = [];
    }
    posts.push({ name, text, time: Date.now() });
    localStorage.setItem(KEY, JSON.stringify(posts));
    $("#postText").value = "";
    load();
  });

  load();
}

// =========================================================
// DASHBOARD SUMMARY
// =========================================================
function updateDashboard() {
  const dashWeather = $("#dashWeather");
  const dashSoil = $("#dashSoil");
  const dashAI = $("#dashAI");
  const dashPest = $("#dashPest");
  const dashProfit = $("#dashProfit");

  if (!dashWeather) return; // dashboard not present

  const wText = $("#weatherResult")?.innerText || "";
  dashWeather.innerHTML = wText
    ? `<small>${wText.split("\n").slice(0, 4).join("<br>")}</small>`
    : "--";

  const ph = $("#ph")?.value || "--";
  const m = $("#moisture")?.value || "--";
  const c = $("#crop")?.value || "--";
  dashSoil.innerHTML = `
    <p>Crop: <b>${c}</b></p>
    <p>pH: <b>${ph}</b></p>
    <p>Moisture: <b>${m}%</b></p>
  `;

  const aiText = $("#aiResult")?.innerText || "";
  const s = aiText.match(/Health Index:\s*(\d+)%/i)?.[1] || "--";
  dashAI.innerHTML = `<p>Health Index: <b>${s}%</b></p>`;

  const alertText = $("#alertResult")?.innerText || "--";
  dashPest.innerHTML = `<small>${alertText}</small>`;

  const pr = $("#profitResult")?.innerText || "";
  const pv = pr.match(/Estimated Profit:\s*₹([0-9,]+)/)?.[1] || "--";
  dashProfit.innerHTML = pv ? `<p>Profit: <b>₹${pv}</b></p>` : "--";
}
