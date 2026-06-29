const sliderIds = ["D", "U", "k", "source_x", "source_y", "sensor_x", "sensor_y", "radius", "total_time"];
let lastParams = null;
let lastPayload = null;

function readParams() {
  const params = {
    preset: document.getElementById("preset").value,
    geometry: document.getElementById("geometry").value,
    boundary: document.getElementById("boundary").value,
    outer_concentration: 1.0,
  };
  for (const id of sliderIds) {
    params[id] = Number(document.getElementById(id).value);
  }
  return params;
}

function updateSliderLabels() {
  for (const id of sliderIds) {
    const input = document.getElementById(id);
    const label = document.getElementById(`${id}Value`);
    label.textContent = input.value;
  }
}

async function loadPresets() {
  const response = await fetch("/api/presets");
  const presets = await response.json();
  const select = document.getElementById("preset");
  select.innerHTML = "";
  for (const preset of presets) {
    const option = document.createElement("option");
    option.value = preset.name;
    option.textContent = preset.title;
    select.appendChild(option);
  }
  select.value = "microchannel_biosensor";
}

async function loadOpenFOAMStatus() {
  const response = await fetch("/api/openfoam/status");
  const status = await response.json();
  const text = document.getElementById("openfoamStatus");
  const button = document.getElementById("openfoamButton");
  text.textContent = status.installed ? `installed: ${status.version}` : "not installed; Python solver enabled";
  button.disabled = !status.installed;
  if (!status.installed) {
    button.classList.add("secondary");
  }
}

async function runSimulation() {
  lastParams = readParams();
  const response = await fetch("/api/simulate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(lastParams),
  });
  lastPayload = await response.json();
  renderPayload(lastPayload);
}

function renderPayload(payload) {
  if (payload.kind === "cartesian") {
    const finalFrame = payload.frames[payload.frames.length - 1];
    Plotly.react("fieldPlot", [{
      z: finalFrame,
      x: payload.x_um,
      y: payload.y_um,
      type: "heatmap",
      colorscale: "Viridis",
      colorbar: { title: "C" },
    }], {
      title: "Concentration field",
      xaxis: { title: "x (um)" },
      yaxis: { title: "y (um)" },
      margin: { t: 44, r: 20, b: 46, l: 54 },
    }, { responsive: true });

    Plotly.react("curvePlot", [
      {
        x: payload.diagnostics.diagnostic_times_s,
        y: payload.diagnostics.sensor_concentration,
        type: "scatter",
        mode: "lines",
        name: "sensor",
      },
      {
        x: payload.diagnostics.diagnostic_times_s,
        y: payload.diagnostics.total_mass,
        type: "scatter",
        mode: "lines",
        name: "mass",
        yaxis: "y2",
      },
    ], {
      title: "Sensor and mass",
      xaxis: { title: "time (s)" },
      yaxis: { title: "sensor" },
      yaxis2: { title: "mass", overlaying: "y", side: "right" },
      margin: { t: 44, r: 58, b: 46, l: 54 },
    }, { responsive: true });
  } else {
    const traces = payload.profiles.map((profile, index) => ({
      x: payload.r_um,
      y: profile,
      type: "scatter",
      mode: "lines",
      name: `${payload.frame_times_s[index].toFixed(2)} s`,
    }));
    Plotly.react("fieldPlot", traces, {
      title: "Radial concentration profiles",
      xaxis: { title: "radius (um)" },
      yaxis: { title: "concentration" },
      margin: { t: 44, r: 20, b: 46, l: 54 },
    }, { responsive: true });

    Plotly.react("curvePlot", [
      {
        x: payload.diagnostics.diagnostic_times_s,
        y: payload.diagnostics.boundary_flux,
        type: "scatter",
        mode: "lines",
        name: "boundary flux",
      },
      {
        x: payload.diagnostics.diagnostic_times_s,
        y: payload.diagnostics.total_mass,
        type: "scatter",
        mode: "lines",
        name: "mass",
        yaxis: "y2",
      },
    ], {
      title: "Flux and mass",
      xaxis: { title: "time (s)" },
      yaxis: { title: "flux" },
      yaxis2: { title: "mass", overlaying: "y", side: "right" },
      margin: { t: 44, r: 58, b: 46, l: 54 },
    }, { responsive: true });
  }
  renderNumbers(payload.diagnostics.dimensionless);
  renderMetadata(payload.diagnostics);
}

function renderNumbers(numbers) {
  const container = document.getElementById("numbers");
  container.innerHTML = Object.entries(numbers).map(([name, item]) => (
    `<div><strong>${name}</strong>: ${Number(item.value).toPrecision(3)}<br><span>${item.interpretation}</span></div>`
  )).join("");
}

function renderMetadata(diagnostics) {
  const metadata = diagnostics.metadata;
  document.getElementById("runMeta").innerHTML = [
    `<strong>${diagnostics.state_label}</strong>`,
    `dt: ${Number(metadata.dt_s).toPrecision(3)} s`,
    `steps: ${metadata.steps}`,
    `warnings: ${diagnostics.warnings.length}`,
  ].join("<br>");
}

async function exportPng() {
  const params = lastParams || readParams();
  const response = await fetch("/api/export_png", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  const data = await response.json();
  const link = document.getElementById("downloadLink");
  link.href = `data:image/png;base64,${data.content_base64}`;
  link.download = data.filename;
  link.hidden = false;
  link.textContent = "Download PNG";
}

async function exportCsv() {
  const params = lastParams || readParams();
  const response = await fetch("/api/export_csv", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  const blob = await response.blob();
  const link = document.getElementById("downloadLink");
  link.href = URL.createObjectURL(blob);
  link.download = `${params.preset}.csv`;
  link.hidden = false;
  link.textContent = "Download CSV";
}

function resetControls() {
  const defaults = { D: 80, U: 200, k: 0.02, source_x: 24, source_y: 45, sensor_x: 170, sensor_y: 45, radius: 20, total_time: 1 };
  for (const [id, value] of Object.entries(defaults)) {
    document.getElementById(id).value = value;
  }
  document.getElementById("preset").value = "microchannel_biosensor";
  document.getElementById("geometry").value = "spherical";
  document.getElementById("boundary").value = "absorbing";
  updateSliderLabels();
}

async function runOpenFOAM() {
  const response = await fetch("/api/run_openfoam", { method: "POST" });
  const data = await response.json();
  document.getElementById("openfoamStatus").textContent = data.message;
}

async function start() {
  updateSliderLabels();
  for (const id of sliderIds) {
    document.getElementById(id).addEventListener("input", updateSliderLabels);
  }
  document.getElementById("runButton").addEventListener("click", runSimulation);
  document.getElementById("exportPngButton").addEventListener("click", exportPng);
  document.getElementById("exportCsvButton").addEventListener("click", exportCsv);
  document.getElementById("resetButton").addEventListener("click", resetControls);
  document.getElementById("openfoamButton").addEventListener("click", runOpenFOAM);
  await loadPresets();
  await loadOpenFOAMStatus();
  await runSimulation();
}

start();
