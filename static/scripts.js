document.addEventListener('DOMContentLoaded', () => {
    handleFormSubmission('#register-form');
    handleFormSubmission('#recognize-form');
});

function handleFormSubmission(formId) {
    const form = document.querySelector(formId);
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(form);
        
        try {
            const response = await fetch(form.action, {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.error) {
                showError(data.error);
            } else {
                showResults(data, formId === '#register-form');
            }
        } catch (error) {
            showError('An error occurred. Please try again.');
        }
    });
}
function showLoading() {
    document.querySelector('.overlay').style.display = 'flex';
    document.querySelector('.loader').style.display = 'block';
    document.querySelectorAll('button[type="submit"]').forEach(btn => {
        btn.disabled = true;
    });
}

function hideLoading() {
    document.querySelector('.overlay').style.display = 'none';
    document.querySelector('.loader').style.display = 'none';
    document.querySelectorAll('button[type="submit"]').forEach(btn => {
        btn.disabled = false;
    });
}

function showStatusMessage(message, isSuccess = true) {
    const statusDiv = document.querySelector('.status-message');
    statusDiv.className = `status-message ${isSuccess ? 'success' : 'error'}`;
    statusDiv.querySelector('.message-text').textContent = message;
    statusDiv.style.display = 'block';
    
    setTimeout(() => {
        statusDiv.style.display = 'none';
    }, 3000);
}

// Modify the handleFormSubmission function
function handleFormSubmission(formId) {
    const form = document.querySelector(formId);
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        showLoading();
        
        try {
            const response = await fetch(form.action, {
                method: 'POST',
                body: new FormData(form)
            });
            
            const data = await response.json();
            hideLoading();
            
            if (data.error) {
                showStatusMessage(data.error, false);
            } else {
                showResults(data, formId === '#register-form');
                if (formId === '#register-form') {
                    showStatusMessage(`Registration complete for ${data.message.split(' ')[2]}!`);
                }
            }
        } catch (error) {
            hideLoading();
            showStatusMessage('An error occurred. Please try again.', false);
        }
    });
}

function showResults(data, isRegistration) {
    clearResults();
    
    // Show annotated image
    if (data.annotated_img) {
        const annotatedContainer = document.createElement('div');
        annotatedContainer.innerHTML = `
            <h3>Processed Image:</h3>
            <img src="/static/uploads/${data.annotated_img}">
        `;
        document.getElementById('annotated-container').appendChild(annotatedContainer);
    }

    // Show crop images
    if (data.crops) {
        const cropsContainer = document.createElement('div');
        cropsContainer.innerHTML = `
            <h3>Detected Muzzles:</h3>
            ${data.crops.map(crop => `
                <img src="/static/uploads/${crop}">
            `).join('')}
        `;
        document.getElementById('crops-container').appendChild(cropsContainer);
    }

    // Show recognition result
    if (!isRegistration) {
        const resultDiv = document.createElement('div');
        resultDiv.innerHTML = data.recognized_cattle_id ?
            `<h3>Recognized: ${data.recognized_cattle_id}</h3>
             <p>Similarity: ${(data.similarity * 100).toFixed(1)}%</p>` :
            `<h3>No match found</h3>
             <p>Highest similarity: ${(data.similarity * 100).toFixed(1)}%</p>`;
        document.getElementById('similarity-result').appendChild(resultDiv);
    }
}

function clearResults() {
    document.getElementById('annotated-container').innerHTML = '';
    document.getElementById('crops-container').innerHTML = '';
    document.getElementById('similarity-result').innerHTML = '';
}

function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error';
    errorDiv.innerHTML = message;
    document.getElementById('results').prepend(errorDiv);
    setTimeout(() => errorDiv.remove(), 3000);
}