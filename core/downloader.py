import os
import time
import glob
import yt_dlp
import config
from utils.logger import logger
from utils.helpers import get_video_id_from_url, sanitize_filename

class YouTubeDownloader:
    def __init__(self, output_dir=None, max_retries=3, retry_delay=2):
        """Initialize the YouTube downloader
        
        Args:
            output_dir: Directory to save downloaded videos (defaults to config.INPUTS_DIR)
            max_retries: Maximum number of download attempts
            retry_delay: Delay between retry attempts in seconds
        """
        self.output_dir = output_dir or config.INPUTS_DIR
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        os.makedirs(self.output_dir, exist_ok=True)
    
    def download(self, url):
        """Download a video from YouTube with retry logic using yt-dlp
        
        Args:
            url: YouTube video URL
            
        Returns:
            Path to downloaded video file or None if download failed
        """
        # Configure yt-dlp options
        # We want mp4 format, best quality
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': os.path.join(self.output_dir, '%(title)s_%(id)s.%(ext)s'),
            'restrictfilenames': True,  # Ensure safe filenames
            'noplaylist': True,
            'quiet': True,
            'no_warnings': True,
        }

        for attempt in range(self.max_retries):
            try:
                logger.info(f"Downloading video from {url} (Attempt {attempt+1}/{self.max_retries})")
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    # Extract info first to get filename and check if exists
                    info = ydl.extract_info(url, download=False)
                    video_title = info.get('title', 'video')
                    video_id = info.get('id', 'unknown')
                    
                    # Prepare expected filename pattern (yt-dlp sanitizes it)
                    # We let yt-dlp handle the downloading and naming, then we find the file
                    
                    # Perform download
                    logger.info(f"Downloading '{video_title}'...")
                    error_code = ydl.download([url])
                    
                    if error_code != 0:
                        raise Exception(f"yt-dlp returned error code {error_code}")
                    
                    # Find the downloaded file
                    # We use the prepare_filename method to know perfectly where it is
                    filename = ydl.prepare_filename(info)
                    
                    # Verify it exists
                    if os.path.exists(filename):
                        logger.info(f"Download complete: {filename}")
                        return filename
                    else:
                        # Fallback search if something went wrong with name prediction
                        # (this is rare with prepare_filename but good for safety)
                        search_pattern = os.path.join(self.output_dir, f"*{video_id}*.mp4")
                        files = glob.glob(search_pattern)
                        if files:
                            logger.info(f"Download complete: {files[0]}")
                            return files[0]
                                            
                    return None
                
            except Exception as e:
                logger.warning(f"Error downloading video (attempt {attempt+1}): {str(e)}")
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Failed to download after {self.max_retries} attempts: {str(e)}")
                    return None