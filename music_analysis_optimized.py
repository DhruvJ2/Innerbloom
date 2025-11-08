"""
Enhanced Music Analysis with Lyrics Detection (Hindi & English)
Optimized with beautiful terminal output
"""

import librosa
import numpy as np
import soundfile as sf
import os
import json
import time
import argparse
import warnings
from multiprocessing import Pool, cpu_count
import logging

# Beautiful terminal output
try:
    from termcolor import colored, cprint
    from pyfiglet import figlet_format
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False
    print("Install termcolor and pyfiglet for beautiful output: pip install termcolor pyfiglet")

# Lyrics detection libraries (optional)
try:
    from lyricsgenius import Genius
    GENIUS_AVAILABLE = True
except ImportError:
    GENIUS_AVAILABLE = False
    print("Note: lyricsgenius not installed. Install with: pip install lyricsgenius")

try:
    import speech_recognition as sr  # type: ignore
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    sr = None  # type: ignore
    print("Note: speech_recognition not installed. Install with: pip install SpeechRecognition")

# ============================================================================
# SETUP
# ============================================================================

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# BEAUTIFUL TERMINAL OUTPUT HELPERS
# ============================================================================

def print_banner(text, color='cyan'):
    """Print beautiful ASCII banner."""
    if COLORS_AVAILABLE:
        banner = figlet_format(text, font='slant')
        cprint(banner, color, attrs=['bold'])
    else:
        print(f"\n{'='*80}\n{text}\n{'='*80}\n")

def print_section(text, color='yellow'):
    """Print section header."""
    if COLORS_AVAILABLE:
        cprint(f"\n{'='*80}", color)
        cprint(f"  {text}", color, attrs=['bold'])
        cprint(f"{'='*80}\n", color)
    else:
        print(f"\n{'='*80}\n{text}\n{'='*80}\n")

def print_info(label, value, color='green'):
    """Print formatted info."""
    if COLORS_AVAILABLE:
        print(f"  {colored('●', 'blue')} {colored(label + ':', 'white', attrs=['bold'])} {colored(value, color)}")
    else:
        print(f"  • {label}: {value}")

def print_track_header(num, total, filename, color='cyan'):
    """Print track processing header."""
    if COLORS_AVAILABLE:
        print(f"\n{colored('═'*80, 'blue')}")
        print(f"{colored('♫', 'magenta', attrs=['bold'])} Track [{num}/{total}] {colored(filename, color, attrs=['bold'])}")
        print(f"{colored('─'*80, 'blue')}")
    else:
        print(f"\n[{num}/{total}] {filename}")

def print_progress(message, color='yellow'):
    """Print progress message."""
    if COLORS_AVAILABLE:
        print(f"  {colored('⟳', 'yellow', attrs=['bold'])} {colored(message, color)}")
    else:
        print(f"  ⟳ {message}")

def print_success(message, color='green'):
    """Print success message."""
    if COLORS_AVAILABLE:
        print(f"  {colored('✓', 'green', attrs=['bold'])} {colored(message, color)}")
    else:
        print(f"  ✓ {message}")

def print_error(message):
    """Print error message."""
    if COLORS_AVAILABLE:
        print(f"  {colored('✗', 'red', attrs=['bold'])} {colored(message, 'red')}")
    else:
        print(f"  ✗ {message}")

# ============================================================================
# LYRICS DETECTION & ANALYSIS
# ============================================================================

def detect_language(text):
    """Detect if text is Hindi or English."""
    # Check for Devanagari Unicode range (Hindi)
    hindi_chars = sum(1 for char in text if '\u0900' <= char <= '\u097F')
    total_chars = len([c for c in text if c.isalpha()])
    
    if total_chars == 0:
        return "unknown"
    
    hindi_ratio = hindi_chars / total_chars
    return "hindi" if hindi_ratio > 0.3 else "english"

def extract_lyrics_from_audio(audio_file, language='auto'):
    """
    Extract lyrics from audio using speech recognition.
    Note: This extracts spoken words, not embedded lyrics.
    """
    if not SPEECH_RECOGNITION_AVAILABLE:
        return None, None
    
    try:
        print_progress("Attempting speech-to-text (may take a moment)...")
        
        recognizer = sr.Recognizer()
        
        # Load audio (limited to first 60 seconds for speed)
        y, sr_rate = librosa.load(audio_file, sr=16000, duration=60)
        
        # Save temporary WAV for speech recognition
        temp_wav = "temp_speech.wav"
        sf.write(temp_wav, y, sr_rate)
        
        with sr.AudioFile(temp_wav) as source:
            audio_data = recognizer.record(source, duration=30)
        
        # Try recognition based on language
        detected_text = None
        detected_lang = language
        
        if language == 'auto' or language == 'english':
            try:
                detected_text = recognizer.recognize_google(audio_data, language='en-US')
                detected_lang = 'english'
            except:
                pass
        
        if (detected_text is None) and (language == 'auto' or language == 'hindi'):
            try:
                detected_text = recognizer.recognize_google(audio_data, language='hi-IN')
                detected_lang = 'hindi'
            except:
                pass
        
        # Cleanup
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        
        if detected_text:
            # Verify language
            actual_lang = detect_language(detected_text)
            return detected_text, actual_lang
        
        return None, None
        
    except Exception as e:
        logger.debug(f"Speech recognition failed: {e}")
        return None, None

def analyze_lyrics_sentiment(text, language='english'):
    """
    Analyze sentiment and themes from lyrics text.
    Returns mood, themes, and sentiment scores.
    """
    if not text:
        return None
    
    text_lower = text.lower()
    
    # Multi-language keyword sets
    if language == 'hindi':
        # Hindi sentiment keywords (transliterated for code compatibility)
        positive_keywords = ['प्यार', 'खुश', 'सुंदर', 'जीवन', 'दिल', 'मोहब्बत', 'आशा', 
                           'sapna', 'khushi', 'pyar', 'dil', 'zindagi', 'mohabbat']
        negative_keywords = ['दर्द', 'गम', 'रोना', 'टूटा', 'अकेला', 'आंसू',
                           'dard', 'gam', 'toot', 'akela', 'aansu']
        energetic_keywords = ['नाच', 'धूम', 'पार्टी', 'डांस', 'nach', 'dhoom', 'party', 'dance']
        calm_keywords = ['शांत', 'आराम', 'सुकून', 'shant', 'sukoon', 'aram']
    else:
        # English keywords
        positive_keywords = ['love', 'happy', 'joy', 'beautiful', 'amazing', 'wonderful', 
                           'dream', 'feel', 'heaven', 'bright', 'light', 'shine', 
                           'dance', 'celebrate', 'life', 'hope', 'smile', 'peace', 'free']
        negative_keywords = ['sad', 'dark', 'lost', 'broken', 'hurt', 'pain', 'alone', 
                           'fall', 'dying', 'tears', 'shadow', 'fear', 'cold', 
                           'empty', 'night', 'storm', 'fade', 'cry']
        energetic_keywords = ['fire', 'energy', 'power', 'rock', 'wild', 'run', 'beat', 
                             'electric', 'pulse', 'drive', 'rush', 'jump', 'loud', 'fast']
        calm_keywords = ['calm', 'soft', 'gentle', 'quiet', 'slow', 'peace', 'still', 
                        'flow', 'breeze', 'whisper', 'ambient', 'serene']
    
    # Count keyword occurrences
    positive_count = sum(1 for word in positive_keywords if word in text_lower)
    negative_count = sum(1 for word in negative_keywords if word in text_lower)
    energetic_count = sum(1 for word in energetic_keywords if word in text_lower)
    calm_count = sum(1 for word in calm_keywords if word in text_lower)
    
    # Calculate scores
    if positive_count + negative_count > 0:
        sentiment_score = (positive_count - negative_count) / (positive_count + negative_count)
    else:
        sentiment_score = 0.0
    
    if energetic_count + calm_count > 0:
        energy_score = (energetic_count - calm_count) / (energetic_count + calm_count)
    else:
        energy_score = 0.0
    
    # Determine mood
    if sentiment_score > 0.5:
        mood = "Uplifting/Joyful" if energy_score > 0 else "Peaceful/Content"
    elif sentiment_score > 0:
        mood = "Positive/Hopeful" if energy_score > 0 else "Calm/Reflective"
    elif sentiment_score > -0.5:
        mood = "Intense/Passionate" if energy_score > 0 else "Contemplative/Pensive"
    else:
        mood = "Melancholic/Somber" if energy_score <= 0 else "Angry/Aggressive"
    
    # Extract themes (top keywords mentioned)
    all_keywords = positive_keywords + negative_keywords + energetic_keywords + calm_keywords
    mentioned_themes = [word for word in all_keywords if word in text_lower]
    
    return {
        'mood': mood,
        'sentiment_score': round(sentiment_score, 2),
        'energy_score': round(energy_score, 2),
        'language': language,
        'themes': mentioned_themes[:5],  # Top 5 themes
        'positive_ratio': round(positive_count / max(1, positive_count + negative_count), 2),
        'word_count': len(text.split())
    }

def get_lyrics_from_filename(filename):
    """Extract song info and attempt mood analysis from filename."""
    name = os.path.splitext(filename)[0]
    name = name.replace('_', ' ').replace('-', ' ').replace('.', ' ')
    
    # Detect language from filename
    language = detect_language(name)
    
    # Basic sentiment from filename
    sentiment = analyze_lyrics_sentiment(name, language)
    
    return {
        'source': 'filename',
        'text_snippet': name[:100],
        'language': language,
        'sentiment': sentiment
    }

# ============================================================================
# OPTIMIZED AUDIO LOADING
# ============================================================================

def load_audio_optimized(audio_path, sr=44100):
    """Optimized audio loading with format detection."""
    try:
        file_ext = os.path.splitext(audio_path)[1].lower()
        
        if file_ext in ['.wav', '.flac', '.ogg']:
            with sf.SoundFile(audio_path) as f:
                y = f.read().T if f.channels > 1 else f.read()
                sr_loaded = f.samplerate
                
                if sr_loaded != sr:
                    y = librosa.resample(y, orig_sr=sr_loaded, target_sr=sr, res_type='kaiser_fast')
                    sr_loaded = sr
        else:
            y, sr_loaded = librosa.load(audio_path, sr=sr, mono=True)
        
        return y, sr_loaded
    
    except Exception as e:
        logger.error(f"Failed to load {audio_path}: {e}")
        return None, None

# ============================================================================
# COMPREHENSIVE MUSIC FEATURE EXTRACTION
# ============================================================================

def extract_all_features_with_lyrics(audio_path, sr=44100, extract_lyrics=True):
    """Extract all music features plus lyrics analysis."""
    try:
        print_progress(f"Loading audio: {os.path.basename(audio_path)}")
        
        y, sr_loaded = load_audio_optimized(audio_path, sr)
        if y is None:
            return None
        
        filename = os.path.basename(audio_path)
        features = {
            'file': audio_path,
            'filename': filename,
            'duration': len(y) / sr_loaded
        }
        
        print_progress("Analyzing tempo and rhythm...")
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr_loaded)
        features['tempo'] = float(tempo) if not isinstance(tempo, np.ndarray) else float(tempo.item())
        features['beat_frames'] = beat_frames.tolist()[:500]  # Limit for JSON size
        
        print_progress("Extracting spectral features...")
        S = librosa.feature.melspectrogram(y=y, sr=sr_loaded, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        energy = np.mean(S_db, axis=0)
        features['energy_mean'] = float(np.mean(energy))
        features['energy_std'] = float(np.std(energy))
        features['energy_max'] = float(np.max(energy))
        
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr_loaded)[0]
        features['spectral_centroid_mean'] = float(np.mean(spec_centroid))
        features['spectral_centroid_std'] = float(np.std(spec_centroid))
        
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zero_crossing_rate_mean'] = float(np.mean(zcr))
        
        print_progress("Analyzing harmonic content...")
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        harmonic_energy = np.sum(y_harmonic ** 2)
        percussive_energy = np.sum(y_percussive ** 2)
        total_energy = harmonic_energy + percussive_energy + 1e-10
        
        features['harmonic_ratio'] = float(harmonic_energy / total_energy)
        features['percussive_ratio'] = float(percussive_energy / total_energy)
        
        print_progress("Detecting musical key...")
        chroma_cqt = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr_loaded)
        chroma_mean = np.mean(chroma_cqt, axis=1)
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key_idx = np.argmax(chroma_mean)
        key = notes[key_idx]
        
        major_pattern = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        minor_pattern = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])
        major_corr = np.correlate(chroma_mean, np.roll(major_pattern, key_idx))[0]
        minor_corr = np.correlate(chroma_mean, np.roll(minor_pattern, key_idx))[0]
        mode = "Major" if major_corr > minor_corr else "Minor"
        
        features['key'] = f"{key} {mode}"
        features['key_confidence'] = float(max(major_corr, minor_corr))
        
        print_progress("Calculating RMS energy...")
        rms = librosa.feature.rms(y=y)[0]
        features['rms_mean'] = float(np.mean(rms))
        features['rms_std'] = float(np.std(rms))
        
        # Lyrics detection and analysis
        if extract_lyrics:
            print_progress("Analyzing lyrics (this may take a moment)...")
            
            # Try filename-based analysis first
            lyrics_info = get_lyrics_from_filename(filename)
            
            # Optionally try speech recognition (disabled by default for speed)
            # Uncomment below to enable:
            # lyrics_text, detected_lang = extract_lyrics_from_audio(audio_path)
            # if lyrics_text:
            #     lyrics_sentiment = analyze_lyrics_sentiment(lyrics_text, detected_lang)
            #     lyrics_info = {
            #         'source': 'speech_recognition',
            #         'text_snippet': lyrics_text[:200],
            #         'language': detected_lang,
            #         'sentiment': lyrics_sentiment
            #     }
            
            features['lyrics_analysis'] = lyrics_info
            
            if lyrics_info and lyrics_info.get('sentiment'):
                print_success(f"Lyrics mood: {lyrics_info['sentiment']['mood']} ({lyrics_info['language']})")
        
        # Store audio for mixing
        features['audio'] = y
        features['sr'] = sr_loaded
        
        return features
    
    except Exception as e:
        print_error(f"Error processing {audio_path}: {e}")
        return None

# ============================================================================
# INTELLIGENT TRACK ORDERING
# ============================================================================

def calculate_transition_score(track1, track2):
    """Calculate smooth transition score between tracks."""
    score = 0.0
    
    # Tempo similarity (30%)
    tempo_diff = abs(track1['tempo'] - track2['tempo'])
    tempo_score = max(0, 100 - tempo_diff)
    score += tempo_score * 0.3
    
    # Energy similarity (25%)
    energy_diff = abs(track1['energy_mean'] - track2['energy_mean'])
    energy_score = max(0, 100 - energy_diff)
    score += energy_score * 0.25
    
    # Key compatibility (20%)
    if track1.get('key') and track2.get('key'):
        key1_note = track1['key'].split()[0]
        key2_note = track2['key'].split()[0]
        key_circle = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F']
        try:
            idx1 = key_circle.index(key1_note)
            idx2 = key_circle.index(key2_note)
            key_distance = min(abs(idx1 - idx2), 12 - abs(idx1 - idx2))
            key_score = (1 - key_distance / 6) * 100
            score += key_score * 0.2
        except:
            score += 50 * 0.2
    
    # Harmonic similarity (15%)
    harmonic_diff = abs(track1['harmonic_ratio'] - track2['harmonic_ratio'])
    harmonic_score = (1 - harmonic_diff) * 100
    score += harmonic_score * 0.15
    
    # Mood compatibility (10%) - if lyrics analysis available
    if (track1.get('lyrics_analysis') and track2.get('lyrics_analysis') and
        track1['lyrics_analysis'].get('sentiment') and track2['lyrics_analysis'].get('sentiment')):
        
        mood1 = track1['lyrics_analysis']['sentiment']['energy_score']
        mood2 = track2['lyrics_analysis']['sentiment']['energy_score']
        mood_diff = abs(mood1 - mood2)
        mood_score = (1 - mood_diff) * 100
        score += mood_score * 0.1
    
    return score

def find_optimal_order(tracks, strategy='energy_progression'):
    """Find optimal track ordering."""
    print_progress(f"Optimizing track order using '{strategy}' strategy...")
    
    if strategy == 'energy_progression':
        # Sort by increasing energy and tempo
        sorted_tracks = sorted(tracks, key=lambda x: (x['tempo'], x['energy_mean']))
    
    elif strategy == 'smooth_transitions':
        # Greedy algorithm for smooth transitions
        if len(tracks) <= 1:
            return tracks
        
        sorted_tracks = [tracks[0]]
        remaining = tracks[1:]
        
        while remaining:
            last_track = sorted_tracks[-1]
            best_idx = max(range(len(remaining)),
                          key=lambda i: calculate_transition_score(last_track, remaining[i]))
            sorted_tracks.append(remaining.pop(best_idx))
    
    else:
        sorted_tracks = tracks
    
    return sorted_tracks

# ============================================================================
# PRISTINE MIXING
# ============================================================================

def constant_power_crossfade(y1, y2, sr, crossfade_duration=5):
    """Constant power crossfade."""
    crossfade_samples = int(sr * crossfade_duration)
    
    if len(y1) < crossfade_samples or len(y2) < crossfade_samples:
        crossfade_samples = min(len(y1), len(y2))
    
    t = np.linspace(0, np.pi/2, crossfade_samples)
    fade_out = np.cos(t)
    fade_in = np.sin(t)
    
    y1_end = y1[-crossfade_samples:]
    y2_start = y2[:crossfade_samples]
    
    mixed_section = y1_end * fade_out + y2_start * fade_in
    y_out = np.concatenate([y1[:-crossfade_samples], mixed_section, y2[crossfade_samples:]])
    
    return y_out

def find_optimal_crossfade_point(y, sr, beat_frames, min_duration=60):
    """Find optimal crossfade point."""
    min_samples = int(min_duration * sr)
    
    if len(y) < min_samples:
        return len(y)
    
    beat_samples = librosa.frames_to_samples(beat_frames, hop_length=512)
    valid_beats = beat_samples[beat_samples >= min_samples]
    
    if len(valid_beats) > 0:
        return int(valid_beats[0])
    return min_samples

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def mix_playlist_enhanced(dir_path, min_track_duration=60, target_sr=44100,
                         crossfade_duration=5, sort_strategy='energy_progression',
                         extract_lyrics=True):
    """Enhanced seamless mixing with lyrics analysis."""
    
    # Beautiful banner
    print_banner("MUSIC MIXER", 'cyan')
    
    audio_files = sorted([
        os.path.join(dir_path, f) for f in os.listdir(dir_path)
        if f.lower().endswith(('.mp3', '.wav', '.flac', '.m4a', '.ogg'))
    ])
    
    if not audio_files:
        print_error("No audio files found!")
        return None
    
    print_section("CONFIGURATION", 'yellow')
    print_info("Audio Files Found", str(len(audio_files)), 'cyan')
    print_info("Crossfade Duration", f"{crossfade_duration}s", 'green')
    print_info("Min Track Duration", f"{min_track_duration}s", 'green')
    print_info("Sort Strategy", sort_strategy, 'magenta')
    print_info("Lyrics Analysis", "Enabled" if extract_lyrics else "Disabled", 'yellow')
    
    print_section("ANALYZING TRACKS", 'cyan')
    
    start_time = time.time()
    track_data = []
    
    for i, file in enumerate(audio_files, 1):
        print_track_header(i, len(audio_files), os.path.basename(file))
        
        features = extract_all_features_with_lyrics(file, target_sr, extract_lyrics)
        
        if features:
            track_data.append(features)
            print_success(f"Tempo: {features['tempo']:.1f} BPM | Key: {features['key']} | Duration: {features['duration']:.1f}s")
            print_success(f"Energy: {features['energy_mean']:.1f}dB | Harmonic Ratio: {features['harmonic_ratio']:.2f}")
        else:
            print_error("Failed to process track")
    
    analysis_time = time.time() - start_time
    print_success(f"Analysis completed in {analysis_time:.1f}s")
    
    if not track_data:
        print_error("No valid tracks processed!")
        return None
    
    # Sort tracks
    print_section("OPTIMIZING TRACK ORDER", 'magenta')
    track_data = find_optimal_order(track_data, strategy=sort_strategy)
    
    for i, track in enumerate(track_data, 1):
        print_info(f"Position {i}", track['filename'], 'cyan')
    
    # Create mix
    print_section("CREATING SEAMLESS MIX", 'green')
    
    mixed_audio = np.array([])
    sr = target_sr
    
    for i, track in enumerate(track_data, 1):
        print_track_header(i, len(track_data), track['filename'], 'green')
        
        y = track['audio']
        beat_frames = np.array(track['beat_frames'])
        
        crossfade_point = find_optimal_crossfade_point(y, sr, beat_frames, min_track_duration)
        y_segment = y[:crossfade_point]
        
        if len(mixed_audio) > 0:
            print_progress(f"Applying {crossfade_duration}s crossfade...")
            mixed_audio = constant_power_crossfade(mixed_audio, y_segment, sr, crossfade_duration)
        else:
            mixed_audio = y_segment
        
        print_success(f"Added {len(y_segment)/sr:.1f}s segment")
    
    # Save files
    output_audio = os.path.join(dir_path, 'seamless_mix_enhanced.wav')
    print_progress("Saving audio file...")
    sf.write(output_audio, mixed_audio, sr)
    
    # Save analysis
    analysis_file = os.path.join(dir_path, 'music_analysis_complete.json')
    analysis_data = {
        'mix_info': {
            'output_file': output_audio,
            'total_tracks': len(track_data),
            'total_duration': len(mixed_audio) / sr,
            'crossfade_duration': crossfade_duration,
            'sort_strategy': sort_strategy
        },
        'tracks': [
            {
                'track_number': i + 1,
                'filename': t['filename'],
                'tempo': round(t['tempo'], 1),
                'key': t['key'],
                'duration': round(t['duration'], 1),
                'energy_mean': round(t['energy_mean'], 1),
                'spectral_centroid': round(t['spectral_centroid_mean'], 0),
                'harmonic_ratio': round(t['harmonic_ratio'], 2),
                'beat_frames': t['beat_frames'],
                'lyrics_analysis': t.get('lyrics_analysis', {})
            }
            for i, t in enumerate(track_data)
        ]
    }
    
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, indent=2, ensure_ascii=False)
    
    # Final output
    print_section("SUCCESS!", 'green')
    print_info("Audio Output", output_audio, 'cyan')
    print_info("Analysis Data", analysis_file, 'cyan')
    print_info("Total Duration", f"{len(mixed_audio)/sr:.1f}s ({len(mixed_audio)/sr/60:.2f} min)", 'yellow')
    print_info("Processing Time", f"{time.time() - start_time:.1f}s", 'yellow')
    
    print_banner("COMPLETE!", 'green')
    
    return track_data, output_audio, analysis_file

# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enhanced Music Mixer with Lyrics Analysis')
    parser.add_argument('--dir', default='youtube_music_audio', help='Music directory')
    parser.add_argument('--crossfade', type=float, default=7, help='Crossfade duration (seconds)')
    parser.add_argument('--min-duration', type=int, default=90, help='Min track duration (seconds)')
    parser.add_argument('--strategy', default='energy_progression', 
                       choices=['energy_progression', 'smooth_transitions', 'original'],
                       help='Track ordering strategy')
    parser.add_argument('--no-lyrics', action='store_true', help='Disable lyrics analysis')
    
    args = parser.parse_args()
    
    try:
        mix_playlist_enhanced(
            args.dir,
            min_track_duration=args.min_duration,
            crossfade_duration=args.crossfade,
            sort_strategy=args.strategy,
            extract_lyrics=not args.no_lyrics
        )
    except KeyboardInterrupt:
        print_error("\nProcess interrupted by user")
    except Exception as e:
        print_error(f"Error: {e}")
        import traceback
        traceback.print_exc()