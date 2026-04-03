import numpy as np
import sounddevice as sd
import pygame
import threading
 
# ─────────────────────────────────────────────
# 1. AUDIO CONSTANTS
# ─────────────────────────────────────────────
SAMPLE_RATE = 44100   # samples per second — CD quality audio
BUFFER_SIZE = 1024    # how many samples per audio chunk (larger = less crackling)
AMPLITUDE   = 0.3     # volume (0.0 → 1.0). Keep below 1.0 to avoid clipping
 
# ─────────────────────────────────────────────
# 2. KEY → FREQUENCY MAP  (your job: add more!)
# ─────────────────────────────────────────────
# These follow the standard musical scale (A4 = 440 Hz).
# Formula for equal temperament: f = 440 * 2^(n/12)
# where n is the number of semitones from A4.
KEY_FREQ = {
    pygame.K_a: 261.63,  # C4  (middle C)
    pygame.K_s: 293.66,  # D4
    pygame.K_d: 329.63,  # E4
    pygame.K_f: 349.23,  # F4
    pygame.K_g: 392.00,  # G4
    pygame.K_h: 440.00,  # A4
    pygame.K_j: 493.88,  # B4
    pygame.K_k: 523.25,  # C5
 
    # Add sharps / higher octave here ↓
    pygame.K_w: 277.18,  # C#4
    pygame.K_e: 311.13,  # D#4
    pygame.K_t: 369.99,  # F#4
    pygame.K_y: 415.30,  # G#4
    pygame.K_u: 466.16,  # A#4
}
 
# ─────────────────────────────────────────────
# 3. SHARED STATE  (thread-safe with a lock)
# ─────────────────────────────────────────────
active_keys = set()   # which keys are currently held down
state_lock  = threading.Lock()
 
# Phase accumulator per key: keeps track of where in the wave cycle each
# frequency is. Without this, each buffer would restart at 0 → audible click.
phase_accumulators = {key: 0.0 for key in KEY_FREQ}
 
# ─────────────────────────────────────────────
# 4. SINE WAVE GENERATOR
# ─────────────────────────────────────────────
def generate_sine(freq: float, num_samples: int, phase: float) -> tuple[np.ndarray, float]:
    """
    Returns (wave, new_phase).
 
    wave       — numpy array of `num_samples` float32 samples
    new_phase  — the phase at the END of this buffer (pass back in next call)
 
    Why track phase?
        A sine wave is: sin(2π * f * t)
        If t resets to 0 each buffer, you get a discontinuity → click/pop.
        Instead we track cumulative phase so the wave is perfectly continuous.
    """
    t         = np.arange(num_samples) / SAMPLE_RATE
    wave      = AMPLITUDE * np.sin(2 * np.pi * freq * t + phase)
    new_phase = (phase + 2 * np.pi * freq * num_samples / SAMPLE_RATE) % (2 * np.pi)
    return wave.astype(np.float32), new_phase
 
# ─────────────────────────────────────────────
# 5. AUDIO CALLBACK  (called by sounddevice on a background thread)
# ─────────────────────────────────────────────
def audio_callback(outdata: np.ndarray, frames: int, time, status):
    """
    sounddevice calls this function every ~23ms asking for `frames` samples.
    We must fill `outdata` with the audio signal or we get silence/glitches.
 
    outdata shape: (frames, channels)  — we use 1 channel (mono)
    """
    mixed = np.zeros(frames, dtype=np.float32)  # start silent
 
    with state_lock:
        keys = set(active_keys)  # snapshot so the main thread can keep writing
 
    for key in keys:
        freq = KEY_FREQ[key]
        wave, phase_accumulators[key] = generate_sine(freq, frames, phase_accumulators[key])
        mixed += wave   # SUPERPOSITION: just add the waves together
 
    # Normalize to prevent clipping when many keys are pressed
    if len(keys) > 1:
        mixed /= len(keys)
 
    outdata[:, 0] = mixed   # write mono signal into left channel
 
# ─────────────────────────────────────────────
# 6. WAVEFORM DATA FOR VISUALIZATION
# ─────────────────────────────────────────────
VIZ_SAMPLES = 512   # how many samples to draw on screen (one screen-width worth)
 
def get_viz_data() -> dict:
    """
    Build one frame of visualization data.
    Returns a dict with each active frequency's wave + the summed wave.
    """
    with state_lock:
        keys = set(active_keys)
 
    waves      = {}
    superposed = np.zeros(VIZ_SAMPLES, dtype=np.float32)
 
    for key in keys:
        freq        = KEY_FREQ[key]
        # Use a fresh t for display only — we don't care about continuity here
        t           = np.linspace(0, VIZ_SAMPLES / SAMPLE_RATE, VIZ_SAMPLES)
        wave        = np.sin(2 * np.pi * freq * t).astype(np.float32)
        waves[key]  = wave
        superposed += wave
 
    if len(keys) > 1:
        superposed /= len(keys)
 
    return {"individual": waves, "superposed": superposed}
 
# ─────────────────────────────────────────────
# 7. PYGAME DRAWING HELPERS
# ─────────────────────────────────────────────
WIDTH, HEIGHT = 1000, 700
COLORS = [
    (0,   200, 255),   # cyan
    (255, 100,   0),   # orange
    (100, 255, 100),   # green
    (255,  50, 150),   # pink
    (200, 150, 255),   # lavender
]
 
def draw_waveform(surface, wave: np.ndarray, y_center: int, height: int, color, label=""):
    """Draw a single waveform centered at y_center."""
    if len(wave) == 0:
        return
 
    points = []
    for i, sample in enumerate(wave):
        x = int(i * WIDTH / len(wave))
        y = int(y_center - sample * height * 0.45)   # scale to fit
        points.append((x, y))
 
    if len(points) > 1:
        pygame.draw.lines(surface, color, False, points, 2)
 
    if label:
        font  = pygame.font.SysFont("monospace", 14)
        text  = font.render(label, True, color)
        surface.blit(text, (10, y_center - height // 2))
 
def draw_grid(surface, y_center: int, height: int):
    """Faint horizontal grid lines."""
    pygame.draw.line(surface, (40, 40, 40), (0, y_center), (WIDTH, y_center), 1)
    pygame.draw.line(surface, (30, 30, 30), (0, y_center - height // 2), (WIDTH, y_center - height // 2), 1)
    pygame.draw.line(surface, (30, 30, 30), (0, y_center + height // 2), (WIDTH, y_center + height // 2), 1)
 
# ─────────────────────────────────────────────
# 8. MAIN LOOP
# ─────────────────────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Keyboard Oscilloscope — Day 07")
    clock  = pygame.time.Clock()
 
    title_font = pygame.font.SysFont("monospace", 18, bold=True)
    hint_font  = pygame.font.SysFont("monospace", 13)
 
    # Start the audio stream — sounddevice will call audio_callback in background
    stream = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BUFFER_SIZE,
        channels=1,
        dtype="float32",
        callback=audio_callback,
    )
    stream.start()
 
    running = True
    while running:
        # ── Event handling ──────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in KEY_FREQ:
                    with state_lock:
                        active_keys.add(event.key)
            elif event.type == pygame.KEYUP:
                if event.key in KEY_FREQ:
                    with state_lock:
                        active_keys.discard(event.key)
 
        # ── Get data ────────────────────────────────────
        viz = get_viz_data()
        individual = viz["individual"]
        superposed = viz["superposed"]
 
        # ── Draw ────────────────────────────────────────
        screen.fill((10, 10, 15))   # near-black background
 
        # Title
        title = title_font.render("KEYBOARD OSCILLOSCOPE", True, (180, 180, 180))
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 12))
 
        # Layout: individual waves in top 60%, superposed in bottom 40%
        num_channels   = max(len(individual), 1)
        channel_height = int(HEIGHT * 0.55 / num_channels)
 
        for i, (key, wave) in enumerate(individual.items()):
            freq    = KEY_FREQ[key]
            y_mid   = int(channel_height * (i + 0.5)) + 50
            color   = COLORS[i % len(COLORS)]
            draw_grid(screen, y_mid, channel_height)
            draw_waveform(screen, wave, y_mid, channel_height, color, f"CH{i+1}  {freq:.1f} Hz")
 
        # Divider line
        divider_y = int(HEIGHT * 0.62)
        pygame.draw.line(screen, (60, 60, 60), (0, divider_y), (WIDTH, divider_y), 1)
 
        # Superposed (combined) wave
        super_y = int(HEIGHT * 0.62 + HEIGHT * 0.20)
        draw_grid(screen, super_y, int(HEIGHT * 0.32))
        draw_waveform(screen, superposed, super_y, int(HEIGHT * 0.32), (255, 255, 100), "∑ SUPERPOSITION")
 
        # Hint
        hint = hint_font.render("Hold keys: A S D F G H J K   |   Sharps: W E T Y U   |   ESC to quit", True, (70, 70, 70))
        screen.blit(hint, (WIDTH // 2 - hint.get_width() // 2, HEIGHT - 24))
 
        pygame.display.flip()
        clock.tick(60)   # 60 fps cap
 
    stream.stop()
    stream.close()
    pygame.quit()
 
if __name__ == "__main__":
    main()