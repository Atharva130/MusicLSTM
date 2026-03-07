# app.py

import gradio as gr
import sys
sys.path.append("src")

from src.generate import generate, save_midi
from src.midi_to_audio import midi_to_mp3

MOODS = ["calm", "sad", "romantic", "energetic", "dark"]

MOOD_INFO = {
    "calm":      {"emoji": "🌿", "bpm": "76",  "composer": "Bach",      "key": "Major"},
    "sad":       {"emoji": "🌧️", "bpm": "60",  "composer": "Chopin",    "key": "Minor"},
    "romantic":  {"emoji": "🌹", "bpm": "72",  "composer": "Beethoven", "key": "Major"},
    "energetic": {"emoji": "⚡",  "bpm": "140", "composer": "Beethoven", "key": "Major"},
    "dark":      {"emoji": "🌑", "bpm": "88",  "composer": "Bach",      "key": "Minor"},
}

def generate_music(mood, length, temperature):
    if not mood or mood not in MOODS:
        mood = "dark"
    try:
        tokens    = generate(mood=mood, length=int(length), temperature=float(temperature))
        midi_path = save_midi(tokens, mood=mood)
        wav_path  = midi_to_mp3(midi_path, "data/audio/output.wav")
        info      = MOOD_INFO[mood]
        track_html = f"""
        <div class='np-row'>Track: <span class='np-val' style='color: #fff;'>Generated {mood.capitalize()} Composition</span></div>
        <div class='np-row'>Artist: <span class='np-val' style='color: #fff;'>{info['composer']}</span></div>
        <div class='np-row'>Key: <span class='np-val' style='color: #fff;'>{info['key']} · {info['bpm']} BPM</span></div>
        <div class='np-notes'>{len(tokens)} notes generated ✓</div>
        """
        return wav_path, track_html
    except Exception as e:
        return None, f"<div style='color:#ef4444;'>❌ {str(e)}</div>"


custom_theme = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="emerald",
    neutral_hue="slate"
).set(
    body_background_fill="#070a0f",
    block_background_fill="#10141a",
    block_border_width="0px",
    slider_color="#00f2fe"
)

css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    font-family: 'Inter', sans-serif !important;
    background: #070a0f !important;
    color: #fff !important;
    position: relative;
    overflow-x: hidden;
}

.gradio-container { max-width: 100% !important; padding: 0 !important; }
footer { display: none !important; }

:root, .dark, body {
    --color-accent: #00f2fe !important;
    --color-accent-soft: rgba(0, 242, 254, 0.2) !important;
    --slider-color: #00f2fe !important;
}

input[type="range"] { accent-color: #00f2fe !important; }

.bg-clef {
    position: absolute;
    font-size: 280px;
    color: rgba(16, 185, 129, 0.04);
    top: 20px; left: 3%;
    z-index: 0;
    user-select: none;
    pointer-events: none;
}

.bg-strings-right {
    position: absolute;
    right: 5%; top: 0; bottom: 0;
    width: 45px;
    z-index: 0;
    pointer-events: none;
    background: repeating-linear-gradient(90deg, transparent 0, transparent 8px, rgba(0, 242, 254, 0.08) 8px, rgba(0, 242, 254, 0.08) 10px);
}

/* ── PIANO HERO ── */
.piano-hero {
    position: relative; width: 100%; height: 380px; display: flex;
    align-items: center; justify-content: center; overflow: hidden;
    z-index: 2;
    border-bottom: 1px solid rgba(0, 242, 254, 0.1);
}
.piano-hero::before {
    content: ''; position: absolute; inset: 0;
    background: repeating-linear-gradient(
        to right,
        #111 0px, #111 2px, #1a1a1a 2px, #1a1a1a 52px,
        #111 52px, #111 54px, #1a1a1a 54px, #1a1a1a 104px,
        #111 104px, #111 106px, #1a1a1a 106px, #1a1a1a 156px,
        #111 156px, #111 158px, #1a1a1a 158px, #1a1a1a 208px,
        #111 208px, #111 210px, #1a1a1a 210px, #1a1a1a 260px,
        #111 260px, #111 262px, #1a1a1a 262px, #1a1a1a 312px,
        #111 312px, #111 314px, #1a1a1a 314px, #1a1a1a 364px
    );
}
.piano-hero::after {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 58%;
    background: repeating-linear-gradient(
        to right,
        transparent 0px, transparent 33px, #030406 33px, #030406 63px,
        transparent 63px, transparent 85px, #030406 85px, #030406 115px,
        transparent 115px, transparent 155px, #030406 155px, #030406 185px,
        transparent 185px, transparent 207px, #030406 207px, #030406 237px,
        transparent 237px, transparent 363px, #030406 363px, #030406 393px,
        transparent 393px, transparent 415px, #030406 415px, #030406 445px,
        transparent 445px, transparent 623px
    );
    background-size: 624px 100%;
}
.hero-overlay {
    position: absolute; inset: 0;
    background: linear-gradient(to bottom, rgba(7,10,15,0.2) 0%, rgba(7,10,15,0.6) 50%, rgba(7,10,15,1) 100%);
    z-index: 1;
}
.hero-content { position: relative; z-index: 2; text-align: center; padding: 0 2rem; }
.hero-title {
    font-size: 2.6rem; font-weight: 700; line-height: 1.2;
    letter-spacing: -0.02em; margin-bottom: 1.8rem;
    background: linear-gradient(135deg, #00f2fe 0%, #4facfe 50%, #10b981 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 4px 30px rgba(0, 242, 254, 0.3);
}

/* ── VISUALIZER ── */
.visualizer { display: flex; align-items: flex-end; justify-content: center; gap: 6px; height: 48px; margin-top: 10px; }
.v-bar { width: 8px; border-radius: 5px 5px 3px 3px; animation: vpulse 1.3s ease-in-out infinite; }
.v-bar:nth-child(1) { height: 16px; background: #00f2fe; animation-delay: 0.00s; }
.v-bar:nth-child(2) { height: 38px; background: #10b981; animation-delay: 0.15s; }
.v-bar:nth-child(3) { height: 22px; background: #4facfe; animation-delay: 0.30s; }
.v-bar:nth-child(4) { height: 44px; background: #00f2fe; animation-delay: 0.08s; box-shadow: 0 0 15px rgba(0,242,254,0.6); }
.v-bar:nth-child(5) { height: 28px; background: #10b981; animation-delay: 0.22s; }
@keyframes vpulse { 0%,100% { transform: scaleY(0.3); opacity: 0.6; } 50% { transform: scaleY(1); opacity: 1; } }

/* ── MOOD CARDS ── */
.mood-section { max-width: 960px; margin: -55px auto 0; padding: 0 2rem; position: relative; z-index: 10; }
.mood-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; margin-bottom: 2.5rem; }
.mood-card {
    background: #10141a; border: 1.5px solid #1a202c; border-radius: 20px;
    padding: 1.6rem 0.5rem 1.3rem; text-align: center; cursor: pointer;
    transition: all 0.2s ease; user-select: none;
}
.mood-card:hover { background: #151a23; border-color: #00f2fe; transform: translateY(-4px); }
.mood-card.active {
    background: #0f1c24; border-color: #10b981;
    box-shadow: 0 0 0 1px #10b981, 0 0 25px rgba(16, 185, 129, 0.3);
    transform: translateY(-4px);
}
.m-emoji { font-size: 2.4rem; display: block; margin-bottom: 0.7rem; filter: drop-shadow(0 2px 5px rgba(0,0,0,0.5)); }
.m-name  { font-size: 0.95rem; font-weight: 700; color: #f0f0f0; margin-bottom: 0.3rem; }
.m-meta  { font-size: 0.72rem; color: #718096; }

/* ── LAYOUT ── */
.main-content {
    max-width: 900px; margin: 0 auto; padding: 0 2rem 3rem;
    position: relative; z-index: 10;
    display: flex; flex-direction: column; gap: 1.5rem;
}
.custom-panel {
    background: #10141a !important; border: 1px solid #1a202c !important;
    border-radius: 22px !important; padding: 2rem 2.5rem !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.6);
}
.player-panel {
    background: #0c0f14 !important;
    border-top: 3px solid #00f2fe !important;
    padding: 2.5rem 3rem !important;
}
.panel-hdr {
    font-size: 1.15rem; font-weight: 700; color: #e2e8f0;
    margin-bottom: 1.5rem; display: flex; align-items: center;
    gap: 0.5rem; letter-spacing: 0.02em;
}

.gradio-slider { background: transparent !important; border: none !important; box-shadow: none !important; margin-bottom: 0.5rem !important; }
.gradio-slider label > span { font-size: 0.75rem !important; font-weight: 600 !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; color: #718096 !important; }

#gen-btn {
    background: linear-gradient(135deg, #00f2fe 0%, #10b981 100%) !important;
    border: none !important; border-radius: 12px !important; color: #fff !important;
    font-weight: 700 !important; font-size: 1.05rem !important; padding: 1rem !important;
    width: 100% !important; margin-top: 1rem !important;
    box-shadow: 0 6px 20px rgba(0, 242, 254, 0.3) !important;
    transition: all 0.2s !important; letter-spacing: 0.05em !important;
    text-shadow: 0 1px 3px rgba(0,0,0,0.3);
}
#gen-btn:hover { transform: translateY(-3px) !important; box-shadow: 0 8px 25px rgba(16, 185, 129, 0.4) !important; }

.player-info-container { display: flex; flex-direction: column; gap: 8px; margin-bottom: 25px; }
.np-row  { font-size: 0.95rem; color: #718096; }
.np-val  { font-weight: 600; margin-left: 5px; }
.np-notes {
    font-size: 0.9rem; color: #00f2fe; margin-top: 10px; font-weight: 700;
    background: rgba(0, 242, 254, 0.1); padding: 5px 12px; border-radius: 50px;
    display: inline-block; width: fit-content; border: 1px solid rgba(0, 242, 254, 0.2);
}

/* ── AUDIO PLAYER — kill the scrollbar inside waveform ── */
.gradio-audio { background: transparent !important; border: none !important; margin-top: 1rem !important; }

/* Target the waveform scroll container specifically */
.waveform-container,
[data-testid="waveform-container"],
.waveform,
.scroll-hide,
.overflow-x-auto,
.overflow-auto,
.overflow-scroll {
    overflow: hidden !important;
    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
}

.waveform-container::-webkit-scrollbar,
[data-testid="waveform-container"]::-webkit-scrollbar,
.waveform::-webkit-scrollbar,
.scroll-hide::-webkit-scrollbar,
.overflow-x-auto::-webkit-scrollbar,
.overflow-auto::-webkit-scrollbar,
.overflow-scroll::-webkit-scrollbar {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
}

/* Kill ALL scrollbars everywhere as nuclear option */
*::-webkit-scrollbar { display: none !important; width: 0 !important; height: 0 !important; }
* { scrollbar-width: none !important; -ms-overflow-style: none !important; }

audio {
    border-radius: 12px !important; border: 1px solid #1a202c !important;
    background: #0c0f14 !important; width: 100% !important;
    height: auto !important; padding: 10px 0; outline: none !important;
}

button[aria-label="Share"], button[title="Share"], .share-button, a[title="Share"] { display: none !important; }
.hidden-control { display: none !important; }

/* ── STATS ── */
.stats-bar {
    display: flex; justify-content: center; gap: 4rem;
    padding: 3rem 0; border-top: 1px solid #1a202c;
    max-width: 960px; margin: 2rem auto 0; position: relative; z-index: 10;
}
.stat-val { font-size: 1.8rem; font-weight: 800; color: #fff; text-align: center; }
.stat-val em { color: #00f2fe; font-style: normal; }
.stat-lbl { font-size: 0.65rem; color: #718096; text-transform: uppercase; letter-spacing: 0.12em; text-align: center; margin-top: 5px; font-weight: 600; }
"""

js = """
() => {
    function setupCardBridge() {
        const cards = document.querySelectorAll('.mood-card');
        if (cards.length === 0) {
            setTimeout(setupCardBridge, 300);
            return;
        }
        cards.forEach(card => {
            card.addEventListener('click', () => {
                cards.forEach(c => c.classList.remove('active'));
                card.classList.add('active');
                const mood = card.getAttribute('data-mood');
                const inputArea = document.querySelector('#hidden-mood-input textarea') || document.querySelector('#hidden-mood-input input');
                if (inputArea) {
                    inputArea.value = mood;
                    inputArea.dispatchEvent(new Event('input', { bubbles: true }));
                }
            });
        });
    }
    setupCardBridge();
}
"""

with gr.Blocks(title="Piano Music Generator") as demo:

    gr.HTML('<div class="bg-clef">&#119070;</div><div class="bg-strings-right"></div>')

    with gr.Column(elem_classes="hidden-control"):
        mood_input = gr.Textbox(value="dark", elem_id="hidden-mood-input")

    gr.HTML("""
    <div class="piano-hero">
        <div class="hero-overlay"></div>
        <div class="hero-content">
            <div class="hero-title">🎶 Turn Emotions Into Music with AI</div>
            <div class="visualizer">
                <div class="v-bar"></div><div class="v-bar"></div><div class="v-bar"></div>
                <div class="v-bar"></div><div class="v-bar"></div>
            </div>
        </div>
    </div>
    """)

    gr.HTML("""
    <div class="mood-section">
        <div class="mood-grid">
            <div class="mood-card" data-mood="calm">
                <span class="m-emoji">🌿</span><div class="m-name">Calm</div><div class="m-meta">76 BPM · Bach</div>
            </div>
            <div class="mood-card" data-mood="sad">
                <span class="m-emoji">🌧️</span><div class="m-name">Sad</div><div class="m-meta">60 BPM · Chopin</div>
            </div>
            <div class="mood-card" data-mood="romantic">
                <span class="m-emoji">🌹</span><div class="m-name">Romantic</div><div class="m-meta">72 BPM · Beethoven</div>
            </div>
            <div class="mood-card" data-mood="energetic">
                <span class="m-emoji">⚡</span><div class="m-name">Energetic</div><div class="m-meta">140 BPM · Beethoven</div>
            </div>
            <div class="mood-card active" data-mood="dark">
                <span class="m-emoji">🌑</span><div class="m-name">Dark</div><div class="m-meta">88 BPM · Bach</div>
            </div>
        </div>
    </div>
    """)

    with gr.Column(elem_classes="main-content"):

        with gr.Column(elem_classes="custom-panel"):
            gr.HTML('<div class="panel-hdr">⚙️ Generation Settings</div>')
            length_input = gr.Slider(minimum=100, maximum=500, value=150, step=50,  label="Notes Length")
            temp_input   = gr.Slider(minimum=0.5, maximum=1.5, value=1.2, step=0.1, label="Creativity")
            btn = gr.Button("🎵 Compose Track", elem_id="gen-btn", size="lg")

        with gr.Column(elem_classes="custom-panel player-panel"):
            gr.HTML('<div class="panel-hdr">🎧 Now Playing</div>')
            status_out = gr.HTML("""
                <div class="player-info-container">
                    <div class="np-row">Track: <span class="np-val" style="color:#fff;">Waiting for generation...</span></div>
                    <div class="np-row">Artist: <span class="np-val" style="color:#fff;">-</span></div>
                    <div class="np-row">Key: <span class="np-val" style="color:#fff;">-</span></div>
                </div>
            """)
            audio_out = gr.Audio(label="", type="filepath", interactive=False, show_label=False)

    gr.HTML("""
    <div class="stats-bar">
        <div><div class="stat-val">501</div><div class="stat-lbl">MIDI Files</div></div>
        <div><div class="stat-val"><em>3.8</em>M</div><div class="stat-lbl">Parameters</div></div>
        <div><div class="stat-val">577</div><div class="stat-lbl">Vocab Size</div></div>
        <div><div class="stat-val"><em>1.89</em></div><div class="stat-lbl">Best Loss</div></div>
        <div><div class="stat-val">5</div><div class="stat-lbl">Moods</div></div>
    </div>
    """)

    btn.click(fn=generate_music, inputs=[mood_input, length_input, temp_input], outputs=[audio_out, status_out])
    demo.load(None, None, None, js=js)

if __name__ == "__main__":
    demo.launch(theme=custom_theme, css=css)