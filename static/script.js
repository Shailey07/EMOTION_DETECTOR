// Emotion Detection System Frontend
// Made by: Shailendra Meghwal

let currentMode = 'realtime';

function switchMode(mode) {
    currentMode = mode;
    
    // Update buttons
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');
    
    // Update content
    document.querySelectorAll('.mode-content').forEach(content => {
        content.classList.remove('active');
    });
    
    if (mode === 'realtime') {
        document.getElementById('realtime-mode').classList.add('active');
        loadHistory();
    } else {
        document.getElementById('image-mode').classList.add('active');
    }
    
    // Hide results
    document.getElementById('results').style.display = 'none';
}

function captureMoment() {
    const video = document.getElementById('video-feed');
    const canvas = document.createElement('canvas');
    canvas.width = video.width;
    canvas.height = video.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    const imageData = canvas.toDataURL('image/jpeg');
    detectEmotion(imageData);
}

function detectFromImage() {
    const preview = document.getElementById('preview-image');
    if (preview.src) {
        detectEmotion(preview.src);
    }
}

async function detectEmotion(imageData) {
    showLoading();
    
    try {
        const response = await fetch('/detect', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData })
        });
        
        const data = await response.json();
        
        if (data.success && data.results.length > 0) {
            displayResults(data.results);
            saveToHistory(data.results[0]);
        } else {
            showError('No face detected! Please try again.');
        }
    } catch (error) {
        console.error('Error:', error);
        showError('Detection failed. Please check your connection.');
    }
}

function displayResults(results) {
    const resultsDiv = document.getElementById('results-content');
    const resultsSection = document.getElementById('results');
    
    resultsDiv.innerHTML = '';
    
    results.forEach(result => {
        const confidencePercent = (result.confidence * 100).toFixed(1);
        const card = document.createElement('div');
        card.className = 'result-card';
        card.innerHTML = `
            <div class="result-emotion">${result.emotion}</div>
            <div class="result-confidence">${confidencePercent}% confidence</div>
        `;
        resultsDiv.appendChild(card);
    });
    
    resultsSection.style.display = 'block';
    
    // Update current emotion badge if in realtime mode
    if (currentMode === 'realtime' && results[0]) {
        const badge = document.getElementById('current-emotion');
        badge.innerHTML = `${results[0].emotion} (${(results[0].confidence * 100).toFixed(1)}%)`;
    }
}

async function saveToHistory(result) {
    try {
        await fetch('/history', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                emotion: result.emotion,
                confidence: result.confidence
            })
        });
        
        loadHistory();
    } catch (error) {
        console.error('Error saving history:', error);
    }
}

async function loadHistory() {
    try {
        const response = await fetch('/history');
        const data = await response.json();
        
        const historyList = document.getElementById('history-list');
        
        if (data.history && data.history.length > 0) {
            historyList.innerHTML = '';
            data.history.slice(-10).reverse().forEach(item => {
                const time = new Date(item.timestamp).toLocaleTimeString();
                const div = document.createElement('div');
                div.className = 'history-item';
                div.innerHTML = `
                    <span>${item.emotion}</span>
                    <span>${(item.confidence * 100).toFixed(1)}%</span>
                    <span style="color: #666; font-size: 0.9em;">${time}</span>
                `;
                historyList.appendChild(div);
            });
        } else {
            historyList.innerHTML = '<p>No history yet. Start detecting!</p>';
        }
    } catch (error) {
        console.error('Error loading history:', error);
    }
}

function showLoading() {
    const resultsSection = document.getElementById('results');
    const resultsDiv = document.getElementById('results-content');
    resultsDiv.innerHTML = '<div class="loading">🔍 Analyzing... Please wait</div>';
    resultsSection.style.display = 'block';
}

function showError(message) {
    const resultsDiv = document.getElementById('results-content');
    resultsDiv.innerHTML = `<div class="error">❌ ${message}</div>`;
}

// Handle image upload
document.getElementById('image-input')?.addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(event) {
            const preview = document.getElementById('preview-image');
            preview.src = event.target.result;
            document.querySelector('.upload-area').style.display = 'none';
            document.querySelector('.preview-container').style.display = 'block';
        };
        reader.readAsDataURL(file);
    }
});

// Load history on page load
loadHistory();

// Auto-refresh history every 5 seconds
setInterval(loadHistory, 5000);