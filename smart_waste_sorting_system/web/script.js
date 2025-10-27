// Smart Waste Sorting System - Frontend JavaScript
class WasteSortingSystem {
    constructor() {
        this.camera = null;
        this.stream = null;
        this.isDetecting = false;
        this.detectionCount = 0;
        this.organicCount = 0;
        this.inorganicCount = 0;
        
        this.initializeElements();
        this.bindEvents();
        this.updateStats();
    }

    initializeElements() {
        this.cameraFeed = document.getElementById('cameraFeed');
        this.detectionCanvas = document.getElementById('detectionCanvas');
        this.startCameraBtn = document.getElementById('startCamera');
        this.stopCameraBtn = document.getElementById('stopCamera');
        this.captureImageBtn = document.getElementById('captureImage');
        this.detectionStatus = document.getElementById('detectionStatus');
        this.detectionResults = document.getElementById('detectionResults');
        this.industryInfo = document.getElementById('industryInfo');
        
        // Stats elements
        this.totalDetections = document.getElementById('totalDetections');
        this.organicCountEl = document.getElementById('organicCount');
        this.inorganicCountEl = document.getElementById('inorganicCount');
        this.recyclingRate = document.getElementById('recyclingRate');
    }

    bindEvents() {
        this.startCameraBtn.addEventListener('click', () => this.startCamera());
        this.stopCameraBtn.addEventListener('click', () => this.stopCamera());
        this.captureImageBtn.addEventListener('click', () => this.captureAndAnalyze());
    }

    async startCamera() {
        try {
            this.updateStatus('Starting camera...', 'warning');
            
            const constraints = {
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'environment'
                }
            };

            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.cameraFeed.srcObject = this.stream;
            
            this.startCameraBtn.disabled = true;
            this.stopCameraBtn.disabled = false;
            this.captureImageBtn.disabled = false;
            
            this.updateStatus('Camera active - Ready to detect', 'success');
            
        } catch (error) {
            console.error('Error accessing camera:', error);
            this.updateStatus('Camera access denied or not available', 'error');
            this.showNotification('Camera access is required for waste detection', 'error');
        }
    }

    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
            this.cameraFeed.srcObject = null;
        }
        
        this.startCameraBtn.disabled = false;
        this.stopCameraBtn.disabled = true;
        this.captureImageBtn.disabled = true;
        
        this.updateStatus('Camera stopped', 'info');
        this.clearDetectionResults();
    }

    async captureAndAnalyze() {
        if (!this.stream) {
            this.showNotification('Please start the camera first', 'warning');
            return;
        }

        try {
            this.updateStatus('Analyzing waste...', 'warning');
            this.captureImageBtn.disabled = true;
            
            // Capture image from video feed
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = this.cameraFeed.videoWidth;
            canvas.height = this.cameraFeed.videoHeight;
            ctx.drawImage(this.cameraFeed, 0, 0);
            
            const imageData = canvas.toDataURL('image/jpeg', 0.8);
            
            // Send to backend for analysis
            const response = await fetch('/api/detect-waste', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageData })
            });

            if (!response.ok) {
                throw new Error('Detection failed');
            }

            const results = await response.json();
            this.displayDetectionResults(results);
            this.updateStats();
            
        } catch (error) {
            console.error('Detection error:', error);
            this.updateStatus('Detection failed - Please try again', 'error');
            this.showNotification('Failed to analyze waste. Please try again.', 'error');
        } finally {
            this.captureImageBtn.disabled = false;
        }
    }

    displayDetectionResults(results) {
        this.clearDetectionResults();
        
        if (!results.detections || results.detections.length === 0) {
            this.detectionResults.innerHTML = `
                <div class="no-results">
                    <i class="fas fa-search"></i>
                    <p>No waste items detected in the image</p>
                </div>
            `;
            this.industryInfo.innerHTML = `
                <div class="no-industry-info">
                    <i class="fas fa-building"></i>
                    <p>No industry applications available</p>
                </div>
            `;
            return;
        }

        // Display detection results
        const detectionsHtml = results.detections.map(detection => {
            const confidence = Math.round(detection.confidence * 100);
            const category = detection.category;
            
            return `
                <div class="detection-item ${category}">
                    <h3>${detection.class}</h3>
                    <p><strong>Category:</strong> ${category.charAt(0).toUpperCase() + category.slice(1)}</p>
                    <p><strong>Confidence:</strong> ${confidence}%</p>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${confidence}%"></div>
                    </div>
                </div>
            `;
        }).join('');

        this.detectionResults.innerHTML = detectionsHtml;

        // Display industry information
        const industryHtml = results.industry_applications.map(app => `
            <div class="industry-item">
                <h3>${app.waste_type}</h3>
                <div class="industry-type">${app.industry_type}</div>
                <div class="applications">${app.applications}</div>
            </div>
        `).join('');

        this.industryInfo.innerHTML = industryHtml;

        // Update counters
        results.detections.forEach(detection => {
            this.detectionCount++;
            if (detection.category === 'organic') {
                this.organicCount++;
            } else if (detection.category === 'inorganic') {
                this.inorganicCount++;
            }
        });

        this.updateStatus(`Detected ${results.detections.length} waste item(s)`, 'success');
    }

    clearDetectionResults() {
        this.detectionResults.innerHTML = `
            <div class="no-results">
                <i class="fas fa-info-circle"></i>
                <p>Start camera and capture an image to detect waste items</p>
            </div>
        `;
        
        this.industryInfo.innerHTML = `
            <div class="no-industry-info">
                <i class="fas fa-building"></i>
                <p>Industry information will appear here after waste detection</p>
            </div>
        `;
    }

    updateStatus(message, type = 'info') {
        const statusIcon = this.detectionStatus.querySelector('i');
        const statusText = this.detectionStatus.querySelector('span') || this.detectionStatus;
        
        // Update icon based on status type
        statusIcon.className = this.getStatusIcon(type);
        statusIcon.style.color = this.getStatusColor(type);
        
        // Update text
        if (statusText.tagName === 'SPAN') {
            statusText.textContent = message;
        } else {
            this.detectionStatus.innerHTML = `${statusIcon.outerHTML} ${message}`;
        }
    }

    getStatusIcon(type) {
        const icons = {
            'success': 'fas fa-check-circle',
            'warning': 'fas fa-exclamation-triangle',
            'error': 'fas fa-times-circle',
            'info': 'fas fa-info-circle'
        };
        return icons[type] || icons['info'];
    }

    getStatusColor(type) {
        const colors = {
            'success': '#27ae60',
            'warning': '#f39c12',
            'error': '#e74c3c',
            'info': '#3498db'
        };
        return colors[type] || colors['info'];
    }

    updateStats() {
        this.totalDetections.textContent = this.detectionCount;
        this.organicCountEl.textContent = this.organicCount;
        this.inorganicCountEl.textContent = this.inorganicCount;
        
        const recyclingRate = this.detectionCount > 0 
            ? Math.round((this.inorganicCount / this.detectionCount) * 100)
            : 0;
        this.recyclingRate.textContent = `${recyclingRate}%`;
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <i class="fas ${this.getStatusIcon(type)}"></i>
            <span>${message}</span>
        `;
        
        // Add styles
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${this.getStatusColor(type)};
            color: white;
            padding: 15px 20px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            z-index: 1000;
            display: flex;
            align-items: center;
            gap: 10px;
            animation: slideInRight 0.3s ease;
        `;
        
        document.body.appendChild(notification);
        
        // Remove after 3 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.3s ease';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }
}

// Add CSS animations for notifications
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Initialize the system when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new WasteSortingSystem();
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        // Page is hidden, could pause camera if needed
        console.log('Page hidden');
    } else {
        // Page is visible again
        console.log('Page visible');
    }
});
