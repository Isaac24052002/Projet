const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const predictBtn = document.getElementById('predictBtn');
const statusEl = document.getElementById('status');
const resultEl = document.getElementById('result');

let currentFile = null;

function setStatus(text) {
  statusEl.textContent = text;
}

function showResult(label, message, isError = false) {
  resultEl.innerHTML = `
    <div class="label">${isError ? 'Erreur' : (label || 'Inconnu')}</div>
    <div>${message}</div>
  `;
}

fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (!file) {
    predictBtn.disabled = true;
    setStatus("En attente d'une image...");
    return;
  }
  currentFile = file;
  predictBtn.disabled = false;
  setStatus('Image chargee. Pret a predire.');

  const reader = new FileReader();
  reader.onload = (e) => {
    preview.innerHTML = `<img src="${e.target.result}" alt="preview" />`;
  };
  reader.readAsDataURL(file);
});

predictBtn.addEventListener('click', async () => {
  if (!currentFile) return;

  const formData = new FormData();
  formData.append('file', currentFile);

  setStatus('Analyse en cours...');
  predictBtn.disabled = true;

  try {
    const response = await fetch('/predict', { method: 'POST', body: formData });
    const data = await response.json();

    if (!response.ok || data.error) {
      throw new Error(data.error || 'Erreur inconnue');
    }

    showResult(data.label, data.message);
    setStatus('Prediction terminee.');
  } catch (err) {
    showResult(null, err.message, true);
    setStatus('Une erreur est survenue.');
  } finally {
    predictBtn.disabled = false;
  }
});
