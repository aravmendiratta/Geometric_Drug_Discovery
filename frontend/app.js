const API_URL = 'http://127.0.0.1:8000';
let viewer = null;
let currentComplex = null;

// DOM Elements
const complexListEl = document.getElementById('complex-list');
const titleEl = document.getElementById('current-complex-title');
const predictBtn = document.getElementById('predict-btn');
const btnText = document.querySelector('.btn-text');
const spinner = document.querySelector('.spinner');
const predValEl = document.getElementById('pred-val');
const actualValEl = document.getElementById('actual-val');

// Initialize 3D Viewer
function initViewer() {
    viewer = $3Dmol.createViewer("3d-viewer", {
        backgroundColor: "rgba(0,0,0,0)",
        antialias: true
    });
}

// Fetch available complexes
async function fetchComplexes() {
    try {
        const res = await fetch(`${API_URL}/complexes`);
        const data = await res.json();
        renderComplexList(data.complexes);
    } catch (err) {
        console.error("Error fetching complexes:", err);
        complexListEl.innerHTML = '<li class="complex-item">Error connecting to API. Ensure backend is running.</li>';
    }
}

// Render Sidebar List
function renderComplexList(complexes) {
    complexListEl.innerHTML = '';
    complexes.forEach(c => {
        const li = document.createElement('li');
        li.className = 'complex-item';
        li.innerText = `Complex ${c.toUpperCase()}`;
        li.onclick = () => selectComplex(c, li);
        complexListEl.appendChild(li);
    });
}

// Handle Selection
async function selectComplex(complexId, listItem) {
    // UI Updates
    document.querySelectorAll('.complex-item').forEach(el => el.classList.remove('active'));
    listItem.classList.add('active');
    titleEl.innerText = `Complex ${complexId.toUpperCase()}`;
    currentComplex = complexId;
    
    // Reset Metrics
    predValEl.innerText = '--';
    predValEl.classList.remove('highlight');
    actualValEl.innerText = '--';
    
    // Enable Predict Button
    predictBtn.disabled = false;
    
    // Fetch structure
    try {
        const res = await fetch(`${API_URL}/structure/${complexId}`);
        const data = await res.json();
        
        // Render in 3Dmol
        viewer.clear();
        
        if (data.protein) {
            viewer.addModel(data.protein, "pdb");
            viewer.setStyle({model: 0}, {cartoon: {color: 'spectrum'}});
        }
        
        if (data.ligand) {
            viewer.addModel(data.ligand, "sdf");
            viewer.setStyle({model: 1}, {stick: {colorscheme: 'cyanCarbon', radius: 0.2}, sphere: {radius: 0.5}});
        }
        
        viewer.zoomTo();
        viewer.render();
        
    } catch (err) {
        console.error("Error loading structure:", err);
    }
}

// Handle Prediction
predictBtn.onclick = async () => {
    if (!currentComplex) return;
    
    // Loading State
    predictBtn.disabled = true;
    btnText.classList.add('hidden');
    spinner.classList.remove('hidden');
    
    try {
        const res = await fetch(`${API_URL}/predict/${currentComplex}`, { method: 'POST' });
        const data = await res.json();
        
        // Animate metrics
        predValEl.innerText = data.predicted_pKd;
        predValEl.classList.add('highlight');
        actualValEl.innerText = data.actual_pKd;
        
    } catch (err) {
        console.error("Error predicting:", err);
        predValEl.innerText = "Err";
    } finally {
        // Reset Button
        predictBtn.disabled = false;
        btnText.classList.remove('hidden');
        spinner.classList.add('hidden');
    }
};

// Start
document.addEventListener('DOMContentLoaded', () => {
    initViewer();
    fetchComplexes();
});
