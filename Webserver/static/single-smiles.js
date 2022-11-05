import { getOptions } from '/static/options.js';
import { hideMoleculeWrapper, displayMoleculeCard } from "./molecule-card.js";
import { clearErrorMessage, displayError } from "./error.js";

const form = document.getElementById('single-smiles-form');
const smilesInput = document.getElementById('smiles-input');
const loadingWrapper = document.querySelector('.loading-wrapper');

function resetElements() {
    hideMoleculeWrapper();
    clearErrorMessage();
}

export function showLoadingWrapper() {
    if (loadingWrapper.className.includes('hidden')) {
        loadingWrapper.classList.remove('hidden');
    }
}

form.onsubmit = (event) => {
    event.preventDefault();

    resetElements();

    showLoadingWrapper();

    loadingWrapper.scrollIntoView({behavior:"smooth"});

    fetch('/smiles', {
        method: 'POST',
        body: JSON.stringify({'smiles': smilesInput.value, 'options': getOptions()}),
        headers: {
            'Content-type': 'application/json; charset=UTF-8',
        }
    })
    .then((response) => (response.ok ? response.json() : Promise.reject(response)))
    .then((data) => displayMoleculeCard(data))
    .catch((err) => (displayError(err)))
}