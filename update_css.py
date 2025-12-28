css_content = """html, body {
    height: 100%;
    scroll-behavior: smooth;
}
body {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    font-family: 'Montserrat', 'Segoe UI', Arial, sans-serif;
    margin: 0;
    padding: 0;
    min-height: 100vh;
}
.main-scroll-container {
    scroll-snap-type: y mandatory;
    overflow-y: auto;
    height: 100vh;
}
.section-fullscreen {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    scroll-snap-align: start;
}
.welcome-text {
    font-size: 3rem;
    font-family: 'Montserrat', 'Segoe UI', Arial, sans-serif;
    color: #00d4ff;
    letter-spacing: 2.5px;
    filter: drop-shadow(0 4px 12px rgba(0, 212, 255, 0.3));
    animation: fadeIn 1.2s ease;
    font-weight: 700;
}
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(20px);}
  to { opacity: 1; transform: translateY(0);}
}
.header h1 {
    font-size: 3.5rem;
    font-family: 'Montserrat', 'Segoe UI', Arial, sans-serif;
    background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    letter-spacing: 2px;
    margin-bottom: 0.85em;
    text-align: center;
    filter: drop-shadow(0 4px 15px rgba(0, 212, 255, 0.3));
    width: 100vw;
}

.header h3 {
    font-size: 1.5rem;
    letter-spacing: 1.2px;
    margin-bottom: 2rem;
    color: #e0f7ff;
    text-align: center;
    font-weight: 500;
}

.weather-section {
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.15) 0%, rgba(44, 83, 100, 0.15) 100%);
    border-radius: 18px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    padding: 2.4rem 2.2rem 1.4rem 2.2rem;
    color: #fff;
    text-align: center;
    border: 1px solid rgba(0, 212, 255, 0.2);
    -webkit-backdrop-filter: blur(10px);
    backdrop-filter: blur(10px);
}
.weather-section h2 {
    color: #00d4ff;
    font-size: 1.35rem;
    font-weight: 600;
}
.weather-card {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    justify-content: center;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(15, 32, 39, 0.8) 0%, rgba(32, 58, 67, 0.8) 100%);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    padding: 2.3rem 2.8rem 2.3rem 2.3rem;
    min-width: 310px;
    max-width: 380px;
    margin: 0 auto;
    border: 1px solid rgba(0, 212, 255, 0.3);
}
.weather-main {
    display: flex;
    align-items: center;
    margin-bottom: 1.7rem;
}
#weather-icon-big img {
    width: 85px;
    height: 85px;
    margin-right: 25px;
}
.weather-temp-big {
    font-size: 2.9rem;
    font-weight: 700;
    color: #fff;
    text-shadow: 0 2px 10px rgba(0, 212, 255, 0.3);
}
.temperature-unit {
    font-size: 1.25rem;
    color: #b3e5fc;
}
.weather-status {
    font-size: 1.35rem;
    color: #e0f7ff;
    margin-bottom: 0.35em;
    font-weight: 500;
}
.weather-extra {
    font-size: 1.12rem;
    color: #b3e5fc;
    margin-bottom: 0.28em;
}
.weather-location {
    color: #00d4ff;
    font-size: 1.18rem;
    text-align: left;
    margin-left: 2.3rem;
    margin-top: 0.85em;
    font-weight: 600;
    letter-spacing: 0.08em;
}
.input-section {
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.12) 0%, rgba(44, 83, 100, 0.12) 100%);
    border-radius: 18px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    padding: 2.1rem 2.3rem 1.7rem 2.3rem;
    color: #fff;
    text-align: center;
    border: 1px solid rgba(0, 212, 255, 0.25);
    -webkit-backdrop-filter: blur(10px);
    backdrop-filter: blur(10px);
}
.input-section input {
    width: 92%;
    padding: 0.9rem;
    margin-bottom: 1.2rem;
    border: 1px solid rgba(0, 212, 255, 0.3);
    border-radius: 8px;
    background: rgba(15, 32, 39, 0.6);
    font-size: 1.1rem;
    color: #e0f7ff;
    font-weight: 500;
    transition: all 0.3s ease;
}
.input-section input:focus { 
    box-shadow: 0 0 15px rgba(0, 212, 255, 0.5);
    outline: none;
    background: rgba(15, 32, 39, 0.8);
    border-color: #00d4ff;
}
.input-section select {
    width: 100%;
    padding: 0.9rem;
    margin-bottom: 1.2rem;
    border: 1px solid rgba(0, 212, 255, 0.3);
    border-radius: 8px;
    background: rgba(15, 32, 39, 0.6);
    font-size: 1.08rem;
    color: #e0f7ff;
    font-weight: 500;
    transition: all 0.3s ease;
}
.input-section select:focus { 
    box-shadow: 0 0 15px rgba(0, 212, 255, 0.5);
    outline: none;
    background: rgba(15, 32, 39, 0.8);
    border-color: #00d4ff;
}
.full-width { width: 100%; }
.mt-1-2 { margin-top: 1.2em; }
.glow-btn {
    background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
    color: #fff;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    padding: 1.1em 2.3em;
    font-family: 'Montserrat', 'Segoe UI', Arial, sans-serif;
    font-size: 1.2em;
    margin-top: 1.3em;
    box-shadow: 0 6px 20px rgba(0, 212, 255, 0.4);
    cursor: pointer;
    transition: all 0.3s ease;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.glow-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(0, 212, 255, 0.6);
    background: linear-gradient(135deg, #0099cc 0%, #00d4ff 100%);
}
.glow-btn:active {
    transform: translateY(0);
}
.result-section {
    min-height: 50px;
    font-size: 1.4rem;
    color: #00d4ff;
    text-align: center;
    margin-bottom: 1.5em;
    letter-spacing: 1.1px;
    font-weight: 600;
}
.chart-container {
    display: flex;
    justify-content: center;
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(44, 83, 100, 0.1));
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
    padding: 1.4rem 1.1rem;
    margin: 0 auto 2rem auto;
    border: 1px solid rgba(0, 212, 255, 0.2);
    -webkit-backdrop-filter: blur(10px);
    backdrop-filter: blur(10px);
}
.download-section {
    display: flex;
    justify-content: center;
    margin-bottom: 2em;
}
.footer-section {
    text-align: center;
    color: #b3e5fc;
    font-size: 1.08rem;
    letter-spacing: 1.1px;
    margin-top: 2.7rem;
    font-weight: 500;
}

/* Spinner animation */
.spinner {
    border: 4px solid rgba(0, 212, 255, 0.2);
    border-top: 4px solid #00d4ff;
    border-radius: 50%;
    width: 45px;
    height: 45px;
    animation: spin 1s linear infinite;
    margin: 22px auto;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
"""

with open('frontend/main.css', 'w', encoding='utf-8') as f:
    f.write(css_content)
    
print("CSS file updated with clean professional design!")
print("The design now has:")
print("- Dark elegant gradient background (teal/gray tones)")
print("- Cyan accent color (#00d4ff)")
print("- Subtle glow effects")
print("- Smooth animations")
print("- Professional glassmorphism cards")
