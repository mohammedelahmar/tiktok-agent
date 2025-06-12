from moviepy.editor import VideoFileClip, CompositeVideoClip, vfx
import config
from utils.logger import logger
from utils.helpers import get_output_path

class VideoFormatter:
    def __init__(self):
        """Initialize video formatter"""
        self.target_aspect_ratio = config.OUTPUT_RESOLUTION[1] / config.OUTPUT_RESOLUTION[0]  # height/width (9:16)
    
    def format_to_9_16(self, video_path, method="crop", output_path=None):
        """Convert a video to 9:16 aspect ratio
        
        Args:
            video_path: Path to the video file
            method: Formatting method ('crop', 'blur', 'bars')
            output_path: Path to save the formatted video (optional)
            
        Returns:
            str: Path to the formatted video
        """
        try:
            with VideoFileClip(video_path) as clip:
                original_aspect = clip.h / clip.w
                
                if original_aspect >= self.target_aspect_ratio:
                    # Video is already tall enough, just need to crop width
                    formatted_clip = self._crop_horizontally(clip)
                else:
                    # Video is too wide, need to use selected method
                    if method == "crop":
                        formatted_clip = self._crop_to_9_16(clip)
                    elif method == "blur":
                        formatted_clip = self._blur_background(clip)
                    elif method == "bars":
                        formatted_clip = self._add_bars(clip)
                    else:
                        logger.warning(f"Unknown format method: {method}, using crop as default")
                        formatted_clip = self._crop_to_9_16(clip)
                
                # Resize to target resolution
                target_width, target_height = config.OUTPUT_RESOLUTION
                formatted_clip = formatted_clip.resize(height=target_height, width=target_width)
                
                # Generate output path if not provided
                if output_path is None:
                    output_path = get_output_path(video_path, suffix=f"9x16_{method}")
                
                # Write the formatted clip
                formatted_clip.write_videofile(
                    output_path,
                    codec="libx264",
                    audio_codec="aac"
                )
                
                logger.info(f"Video formatted and saved to {output_path}")
                return output_path
                
        except Exception as e:
            logger.error(f"Error formatting video: {str(e)}")
            return None
    
    def _crop_to_9_16(self, clip):
        """Crop the video to 9:16 aspect ratio, focusing on the center"""
        target_width = clip.h / self.target_aspect_ratio
        
        # Calculate crop dimensions
        crop_x1 = (clip.w - target_width) / 2
        crop_x2 = crop_x1 + target_width
        
        return clip.crop(x1=crop_x1, y1=0, x2=crop_x2, y2=clip.h)
    
    def _crop_horizontally(self, clip):
        """Crop a tall video horizontally to make it exactly 9:16"""
        target_height = clip.w * self.target_aspect_ratio
        
        # Calculate crop dimensions
        crop_y1 = (clip.h - target_height) / 2
        crop_y2 = crop_y1 + target_height
        
        return clip.crop(x1=0, y1=crop_y1, x2=clip.w, y2=crop_y2)
    
    def _blur_background(self, clip):
        """Add blurred, scaled background to make the video 9:16"""
        # Calculate dimensions for the main clip in the 9:16 frame
        target_height = clip.w * self.target_aspect_ratio
        scale_factor = target_height / clip.h
        
        # Scale the original clip to fit height
        main_clip = clip.resize(scale_factor)
        
        # Create a blurred, scaled background
        background = (clip
                     .resize(height=target_height)
                     .crop(x1=0, y1=0, x2=main_clip.h / self.target_aspect_ratio, y2=target_height)
                     .fx(vfx.blur, sigma=20))
        
        # Place the main clip in the center
        x_pos = (background.w - main_clip.w) / 2
        main_clip = main_clip.set_position((x_pos, 0))
        
        # Composite the clips
        return CompositeVideoClip([background, main_clip], size=(background.w, background.h))
    
    def _add_bars(self, clip):
        """Add black bars to make the video 9:16"""
        # Calculate target height for 9:16
        target_height = clip.w * self.target_aspect_ratio
        
        # Calculate bar height
        bar_height = (target_height - clip.h) / 2
        
        # Create a new clip with the right dimensions
        new_clip = CompositeVideoClip(
            [clip.set_position((0, bar_height))],
            size=(clip.w, target_height)
        )
        
        return new_clip