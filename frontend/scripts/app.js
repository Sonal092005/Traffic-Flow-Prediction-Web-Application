// Initialize chart and history
let trafficChart = null;
let predictionHistory = [];

// Initialize Chart.js
const ctx = document.getElementById('trafficChart').getContext('2d');
trafficChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Traffic Flow',
            data: [],
            borderColor: '#08d9d6',
            backgroundColor: 'rgba(8, 217, 214, 0.1)',
            borderWidth: 2,
            tension: 0.4
        }]
    },
    options: {
        responsive: true,
        scales: {
            y: {
                beginAtZero: true,
                max: 40
            }
        }
    }
});

// Download button functionality
document.getElementById('downloadPredictions').addEventListener('click', function () {
    if (predictionHistory.length === 0) {
        alert('No predictions to download yet!');
        return;
    }
    const dataStr = JSON.stringify(predictionHistory, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'traffic_predictions.json';
    link.click();
    URL.revokeObjectURL(url);
});

// Form submission
document.getElementById('predictForm').addEventListener('submit', function (e) {
    e.preventDefault();
    const hour = document.getElementById('hour').value;
    const weekday = document.getElementById('weekday').value;
    const junction = document.getElementById('junction').value;
    let resultSection = document.getElementById('resultSection');
    resultSection.innerHTML = '<div class="spinner"></div>';

    if (hour < 0 || weekday < 0 || junction < 0) {
        resultSection.innerHTML = `<p style="color:red;">Negative values not allowed for prediction.</p>`;
        return;
    }

    fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            hour: Number(hour),
            weekday: Number(weekday),
            junction: Number(junction)
        })
    })
    .then(res => res.json())
    .then(res => {
        if (res.error) {
            resultSection.innerHTML = `<p style="color:red;">${res.error}</p>`;
        } else {
            const pred = Number(res.predicted_traffic).toFixed(2);
            const level = res.traffic_level;
            const junctionName = res.junction_name;
            const timeLabel = res.time_label;

            resultSection.innerHTML = `
                <h2>Predicted Traffic Flow</h2>
                <p style="font-size:2.5rem;color:#08d9d6;text-shadow:0 0 10px #08d9d6;">
                    ${pred}
                </p>
                <p style="font-size:1.2rem;color:#ffffff;margin:0.4rem 0;">
                    Level: <strong>${level}</strong>
                </p>
                <p style="font-size:1rem;color:#cccccc;margin:0.2rem 0;">
                    ${junctionName} &nbsp; | &nbsp; ${timeLabel}
                </p>
            `;

            let now = new Date().toLocaleTimeString();
            if (trafficChart) {
                trafficChart.data.labels.push(now);
                trafficChart.data.datasets[0].data.push(Number(pred));

                if (trafficChart.data.labels.length > 10) {
                    trafficChart.data.labels.shift();
                    trafficChart.data.datasets[0].data.shift();
                }
                trafficChart.update();
            }

            predictionHistory.push({
                timestamp: now,
                hour,
                weekday,
                junction,
                predicted_traffic: pred,
                traffic_level: level,
                junction_name: junctionName,
                time_label: timeLabel
            });
        }
    })
    .catch(() => {
        resultSection.innerHTML = `<p style="color:red;">Prediction failed. Please try again later.</p>`;
    });
});
