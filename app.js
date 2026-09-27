const form = document.querySelector("#predict-form");
const submitButton = document.querySelector("#submit");
const formError = document.querySelector("#form-error");
const result = document.querySelector("#result");
const resultTitle = document.querySelector("#result-title");
const resultCopy = document.querySelector("#result-copy");
const meterWrap = document.querySelector("#meter-wrap");
const meter = document.querySelector("#meter");
const meterLabel = document.querySelector("#meter-label");
const neighbors = document.querySelector("#neighbors");
const featureWrap = document.querySelector("#feature-wrap");
const featureTable = document.querySelector("#feature-table");

const presets = {
  low: {
    Age: 40,
    Sex: "M",
    ChestPainType: "ATA",
    RestingBP: 140,
    Cholesterol: 289,
    FastingBS: "0",
    RestingECG: "Normal",
    MaxHR: 172,
    ExerciseAngina: "N",
    Oldpeak: 0,
    ST_Slope: "Up",
  },
  high: {
    Age: 48,
    Sex: "F",
    ChestPainType: "ASY",
    RestingBP: 138,
    Cholesterol: 214,
    FastingBS: "0",
    RestingECG: "Normal",
    MaxHR: 108,
    ExerciseAngina: "Y",
    Oldpeak: 1.5,
    ST_Slope: "Flat",
  },
};

function readForm() {
  const data = new FormData(form);
  return {
    Age: data.get("Age"),
    Sex: data.get("Sex"),
    ChestPainType: data.get("ChestPainType"),
    RestingBP: data.get("RestingBP"),
    Cholesterol: data.get("Cholesterol"),
    FastingBS: Number(data.get("FastingBS")),
    RestingECG: data.get("RestingECG"),
    MaxHR: data.get("MaxHR"),
    ExerciseAngina: data.get("ExerciseAngina"),
    Oldpeak: data.get("Oldpeak"),
    ST_Slope: data.get("ST_Slope"),
  };
}

function applyPreset(name) {
  const preset = presets[name];
  for (const [key, value] of Object.entries(preset)) {
    const fields = form.elements[key];
    if (!fields) continue;
    if (fields instanceof RadioNodeList) {
      fields.value = String(value);
    } else {
      fields.value = value;
    }
  }
  formError.hidden = true;
  form.requestSubmit();
}

function showError(message) {
  formError.hidden = false;
  formError.textContent = message;
}

function renderFeatures(features) {
  const headers = Object.keys(features);
  const head = headers.map((name) => `<th>${name}</th>`).join("");
  const cells = headers.map((name) => `<td>${features[name]}</td>`).join("");
  featureTable.innerHTML = `<thead><tr>${head}</tr></thead><tbody><tr>${cells}</tr></tbody>`;
  featureWrap.hidden = false;
}

function renderResult(payload) {
  const high = payload.prediction === 1;
  result.classList.toggle("is-high", high);
  result.classList.toggle("is-low", !high);
  resultTitle.classList.toggle("is-high", high);
  resultTitle.classList.toggle("is-low", !high);
  resultTitle.textContent = high ? "Higher estimated risk" : "Lower estimated risk";

  const votes = payload.neighbors_positive;
  const total = payload.neighbors;
  const share = Math.round(payload.probability * 100);
  resultCopy.textContent = `${votes} of ${total} nearest patients in the training set were labeled with heart disease.`;
  meter.style.setProperty("--p", `${share}%`);
  meterLabel.textContent = `${share}% of the neighbor vote`;
  meterWrap.hidden = false;

  neighbors.innerHTML = "";
  for (let index = 0; index < total; index += 1) {
    const item = document.createElement("li");
    item.classList.toggle("on", index < votes);
    neighbors.appendChild(item);
  }

  renderFeatures(payload.features);
}

document.querySelectorAll("[data-preset]").forEach((button) => {
  button.addEventListener("click", () => applyPreset(button.dataset.preset));
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!form.reportValidity()) return;

  formError.hidden = true;
  submitButton.disabled = true;
  submitButton.textContent = "Estimating…";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(readForm()),
    });
    const payload = await response.json();
    if (!response.ok) {
      showError(payload.error || "The model could not score this record.");
      return;
    }
    renderResult(payload);
  } catch (error) {
    showError("The prediction server is not running. Start it with python server.py.");
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Estimate risk";
  }
});
