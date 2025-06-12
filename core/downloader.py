import os
from pytube import YouTube
import config
from utils.logger import logger
from utils.helpers import get_video_id_from_url, sanitize_filename

class YouTubeDownloader:
    def __init__(self, output_dir=None):
        """Initialize the YouTube downloader
        
        Args:
            output_dir: Directory to save downloaded videos (defaults to config.INPUTS_DIR)
        """
        self.output_dir = output_dir or config.INPUTS_DIR
        os.makedirs(self.output_dir, exist_ok=True)
    
    def download(self, url):
        """Download a video from YouTube
        
        Args:
            url: YouTube video URL
            
        Returns:
            Path to downloaded video file or None if download failed
        """
        try:
            logger.info(f"Downloading video from {url}")
            yt = YouTube(url)
            
            # Get video ID for filename
            video_id = get_video_id_from_url(url) or yt.video_id
            
            # Generate filename
            title = sanitize_filename(yt.title)
            filename = f"{title}_{video_id}.mp4"
            output_path = os.path.join(self.output_dir, filename)
            
            # Download video
            logger.info(f"Downloading '{yt.title}' at {config.YT_DOWNLOAD_RESOLUTION}")
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
            logger.error(f"Error downloading video: {str(e)}")
            return None