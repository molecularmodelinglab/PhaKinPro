const form = document.getElementById('draw-smiles-form');
const loadingWrapper = document.querySelector('.loading-wrapper');
const smileText = document.getElementById('smiles-input')

form.onsubmit = (event) => {
    event.preventDefault();
    smileText.value = document.JME.smiles()
}