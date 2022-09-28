const moleculeWrapper = document.querySelector('.molecule-wrapper');
const moleculeProperties = document.querySelector('.molecule-properties');
const moleculeSVG = document.getElementById('molecule-svg');
const moleculeSMILES = document.getElementById('molecule_smile_string');
const loadingWrapper = document.querySelector('.loading-wrapper');

export function showMoleculeWrapper() {
    if (moleculeWrapper.className.includes('hidden')) {
        moleculeWrapper.classList.remove('hidden');
    }
}

export function hideMoleculeWrapper() {
    if (!moleculeWrapper.className.includes('hidden')) {
        moleculeWrapper.classList.add('hidden');
    }
}

export function hideLoadingWrapper() {
    if (!loadingWrapper.className.includes('hidden')) {
        loadingWrapper.classList.add('hidden');
    }
}

export function displayMoleculeCard(moleculeData) {
    hideLoadingWrapper();
    showMoleculeWrapper();

    moleculeProperties.innerHTML = '';
    moleculeSMILES.innerHTML = moleculeData.SMILES;
    moleculeSVG.innerHTML = moleculeData.svg;

    for (const [modelName, classification, confidence, ad, prob_svg, color, prob_meaning] of moleculeData.pred_data) {

        let wrapper = document.createElement('div');
        wrapper.className = 'option-item custom-control custom-checkbox mb-3';

        let information = document.createElement('span');
        information.id = modelName;
        information.classList.add('model-preds');

        if (ad === "") {
            information.innerHTML = `<h2 style="font-weight: bold">${modelName}</h2> <p style="color:${color};">${classification}</p><p>Confidence: ${confidence}</p>`;
        } else {
            information.innerHTML = `<h2 style="font-weight: bold">${modelName}</h2> <p style="color:${color};">${classification}</p><p>Confidence: ${confidence}</p><p>Applicability Domain: ${ad}</p>`;
        }

        if (prob_svg !== "") {
            let prob_meaning_text = document.createElement('p')
            prob_meaning_text.innerHTML = `${prob_meaning}`
            information.append(prob_meaning_text)
            let prob_svg_element = document.createElement('div');
            prob_svg_element.innerHTML = `<div>${prob_svg}</div>`
            information.append(prob_svg_element)
        }

        wrapper.append(information);
        moleculeProperties.append(wrapper);
    }
}