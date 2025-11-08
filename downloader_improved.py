from ytmusicapi import YTMusic
import subprocess
import os
from multiprocessing.pool import ThreadPool

def download_audio(video_url, save_dir):
    """
    Download audio from YouTube in absolute best quality available.
    No compression, no quality loss - pristine audio.

    Args:
        video_url: YouTube video URL
        save_dir: Directory to save the downloaded audio
    """
    command = [
        'yt-dlp',
        '-f', 'bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio',  # Best available format
        '--audio-quality', '0',  # Highest quality (0 = best, 10 = worst)
        '--embed-thumbnail',  # Embed thumbnail as album art
        '--add-metadata',  # Add metadata (title, artist, etc.)
        '--no-post-overwrites',  # Don't overwrite existing files
        '-o', os.path.join(save_dir, '%(title)s.%(ext)s'),  # Output format
        '--no-playlist',  # Don't download playlists (single video only)
        '--ignore-errors',  # Continue on errors
        '--no-warnings',  # Suppress warnings
        video_url
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"✓ Downloaded: {video_url}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error downloading {video_url}: {e.stderr}")

def download_playlist_audio_parallel(auth_file, playlist_index=0, save_dir='youtube_music_audio', num_workers=6):
    """
    Download all tracks from a YouTube Music playlist in parallel.
    Uses best available audio quality - no compression or quality loss.

    Args:
        auth_file: Path to YTMusic authentication JSON file
        playlist_index: Index of playlist to download (0 = first playlist)
        save_dir: Directory to save downloaded audio files
        num_workers: Number of parallel download threads
    """
    # Initialize YTMusic API
    ytmusic = YTMusic(auth_file)
    playlists = ytmusic.get_library_playlists()

    if playlist_index >= len(playlists):
        print(f"❌ Playlist index {playlist_index} out of range.")
        print(f"Available playlists: {len(playlists)}")
        for i, pl in enumerate(playlists):
            print(f"  [{i}] {pl['title']}")
        return

    # Get selected playlist
    playlist = playlists[playlist_index]
    playlist_id = playlist['playlistId']

    print("\n" + "=" * 70)
    print(f"🎵 DOWNLOADING PLAYLIST: {playlist['title']}")
    print("=" * 70)
    print(f"Format: Best Available (M4A/WebM/Opus - highest bitrate)")
    print(f"Quality: Maximum (no compression)")
    print(f"Save Directory: {save_dir}")
    print(f"Parallel Workers: {num_workers}")
    print("=" * 70 + "\n")

    # Get all tracks from playlist
    tracks = ytmusic.get_playlist(playlist_id, limit=None)['tracks']
    print(f"Found {len(tracks)} tracks in playlist\n")

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Build video URLs
    video_urls = [f"https://www.youtube.com/watch?v={t['videoId']}" for t in tracks]

    # Download in parallel
    print("Starting parallel downloads...\n")
    with ThreadPool(num_workers) as pool:
        pool.starmap(download_audio, [(url, save_dir) for url in video_urls])

    print("\n" + "=" * 70)
    print("✅ ALL DOWNLOADS COMPLETED!")
    print("=" * 70)
    print(f"Files saved in: {save_dir}")
    print(f"Format: Best Available Quality (256kbps+ Opus or AAC)")

if __name__ == '__main__':
    # Configuration
    auth_path = 'browser.json'  # Path to your YTMusic auth file
    playlist_index = 1  # Which playlist to download (0 = first, 1 = second, etc.)
    output_dir = 'youtube_music_audio'  # Output directory
    workers = 6  # Number of parallel downloads

    # Start download
    download_playlist_audio_parallel(
        auth_file=auth_path,
        playlist_index=playlist_index,
        save_dir=output_dir,
        num_workers=workers
    )
