import { getOptions } from '/static/options.js';
import { displayError, clearErrorMessage } from "./error.js";

const form = document.getElementById('multi-smiles-form')
const textArea = document.getElementById('multi-smiles')
const smilesFile = document.getElementById('smiles-file')

function downloadCSV(blob) {
    let url = window.URL.createObjectURL(blob);
    window.location.assign(url);
    URL.revokeObjectURL(url);
}

smilesFile.addEventListener('change', () => {
    let files = smilesFile.files;

    if (files.length == 0) return;

    const file = files[0];

    let reader = new FileReader();

    reader.onload = (event) => {
        const file = event.target.result;
        const lines = file.split(/\r\n|\n/);
        textArea.value = lines.join(',')
    };

    reader.onerror = (event) => alert(event.target.error.name);
    reader.readAsText(file)
});

form.onsubmit = (event) => {
    event.preventDefault();
    clearErrorMessage();

    fetch('/smiles-csv', {
        method: 'POST',
        // Swap newlines for commas, Remove whitespace, trailing/leading commas, and separate by comma into array
        body: JSON.stringify({'smiles': textArea.value.replace(/\r\n/g, ',').replace(/\n/g, ',').replace(/\s/g, '').replace(/(^,)|(,$)/g, '').split(','), 'options': getOptions()}),
        headers: {
            'Content-type': 'application/json; charset=UTF-8',
        }
    })
    .then(async (response) => (response.ok ? response.blob() : Promise.reject(response)))
    .then((blob) => downloadCSV(blob))
    .catch((err) => (displayError(err)))
}