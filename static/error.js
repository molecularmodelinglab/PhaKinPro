const errorMessage = document.getElementById('error-message');
const errorWrapper = document.querySelector('.error-wrapper');
const smilesInput = document.getElementById('smiles-input');
const smilesForm = document.getElementById('single-smiles-form');
const loadingWrapper = document.querySelector('.loading-wrapper');

function displayClientError(err) {
    errorMessage.innerHTML = `<b>Unexpected Client Error</b> ${err}`;
    throw err;
}

export function hideLoadingWrapper() {
    if (!loadingWrapper.className.includes('hidden')) {
        loadingWrapper.classList.add('hidden');
    }
}

function displayServerError(err) {
    if (err.status === 400) {
        errorMessage.innerHTML = 'Invalid SMILES Input!';
        smilesInput.classList.add('is-invalid');
        smilesForm.classList.add('has-danger');
    } else if (err.status === 413) {
        errorMessage.innerHTML = `To many models and SMILES use the <a href="https://github.com/molecularmodelinglab/ZincRx">local version</a> instead`;
        smilesInput.classList.add('is-invalid');
        smilesForm.classList.add('has-danger');
    } else {
        errorMessage.innerHTML = `<b>Unexpected Server Error</b> (${err.status}): ${err.statusText}`;
    }
    hideLoadingWrapper();
}

export function clearErrorMessage() {
    errorWrapper.classList.add('hidden');
    errorMessage.innerHTML = '';

    if (smilesForm.className.includes('has-danger')) {
        smilesForm.classList.remove('has-danger');
        smilesInput.classList.remove('is-invalid');
    }
}

// add the ability to turn off the loader gif when error is thrown
export function displayError(err) {
    errorWrapper.classList.remove('hidden');
    if (err instanceof Response) {
        displayServerError(err);
    } else {
        displayClientError(err);
    }
}
