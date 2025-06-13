import os
import time
from pytube import YouTube
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
        """Download a video from YouTube with retry logic
        
        Args:
            url: YouTube video URL
            
        Returns:
            Path to downloaded video file or None if download failed
        """
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Downloading video from {url} (Attempt {attempt+1}/{self.max_retries})")
                yt = YouTube(url)
                
                # Get video ID for filename
                video_id = get_video_id_from_url(url) or yt.video_id
                
                # Generate filename
                title = sanitize_filename(yt.title)
                filename = f"{title}_{video_id}.mp4"
                output_path = os.path.join(self.output_dir, filename)
                
                # If file already exists, return the path
                if os.path.exists(output_path):
                    logger.info(f"Video already downloaded: {output_path}")
                    return output_path
                
                # Download video
                logger.info(f"Downloading '{yt.title}' at {config.YT_DOWNLOAD_RESOLUTION}")
                
                # Try to get the desired resolution
                stream = (yt.streams
                          .filter(progressive=True, file_extension="mp4")
                          .order_by("resolution")
                          .desc()
                          .first())
                
                if not stream:
                    logger.warning(f"No suitable stream found for {url}")
                    return None
                    
                downloaded_path = stream.download(
                    output_path=self.output_dir,
                    filename=filename
                )
                
                logger.info(f"Download complete: {downloaded_path}")
                return downloaded_path
                
            except Exception as e:
                logger.warning(f"Error downloading video (attempt {attempt+1}): {str(e)}")
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Failed to download after {self.max_retries} attempts")
                    return None