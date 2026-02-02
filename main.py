import argparse
import sys
import os
import logging
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import torch

import config
from utils.logger import logger, set_log_level
from core.downloader import YouTubeDownloader
from core.file_loader import VideoFileLoader
from core.viral_clip_extractor import ViralClipExtractor
from core.formatter import VideoFormatter
from utils.cloud_storage import CloudStorage
from utils.interactive import interactive_mode


def main():
    """Main entry point for the TikTok Agent"""
    parser = argparse.ArgumentParser(description="TikTok Agent - Extract viral clips from videos")
    
    # Input source options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--youtube", "-yt", help="YouTube video URL")
    input_group.add_argument("--file", "-f", help="Path to local video file")
    
    # Clip options
    parser.add_argument("--duration", "-d", type=float, default=config.DEFAULT_CLIP_DURATION,
                        help=f"Duration of clip in seconds (default: {config.DEFAULT_CLIP_DURATION})")
    parser.add_argument("--num-clips", "-n", type=int, default=1,
                        help="Number of viral clips to extract (default: 1)")
    parser.add_argument("--min-gap", "-g", type=float, default=1.0,
                        help="Minimum gap between multiple clips in seconds (default: 1.0)")
    
    # Format options
    parser.add_argument("--format", "-fmt", choices=["crop", "blur", "bars"], default="crop",
                        help="Method for formatting to 9:16 ratio (default: crop)")
    
    # Watermark options
    watermark_group = parser.add_argument_group('Watermark Options')
    watermark_group.add_argument("--watermark", "-wm", action="store_true", default=config.WATERMARK_ENABLED,
                        help="Add watermark to videos")
    watermark_group.add_argument("--watermark-type", choices=["text", "image"], default=config.WATERMARK_TYPE,
                        help="Type of watermark (default: text)")
    watermark_group.add_argument("--watermark-text", default=config.WATERMARK_TEXT,
                        help="Text for watermark (default: @YourUsername)")
    watermark_group.add_argument("--watermark-image", default=config.WATERMARK_IMAGE,
                        help="Path to image for watermark")
    watermark_group.add_argument("--watermark-position", choices=["top-left", "top-right", "bottom-left", "bottom-right", "center"],
                        default=config.WATERMARK_POSITION, help="Position of watermark (default: bottom-right)")
    watermark_group.add_argument("--watermark-opacity", type=float, default=config.WATERMARK_OPACITY,
                        help="Opacity of watermark, 0.0-1.0 (default: 0.7)")
    watermark_group.add_argument("--watermark-padding", type=int, default=config.WATERMARK_PADDING,
                        help="Padding from edges in pixels (default: 20)")
    watermark_group.add_argument("--watermark-text-size", type=int, default=config.WATERMARK_TEXT_SIZE,
                        help="Font size for text watermark (default: 40)")
    watermark_group.add_argument("--watermark-text-color", default=config.WATERMARK_TEXT_COLOR,
                        help="Color for text watermark (default: white)")
    
    # Output options
    parser.add_argument("--output", "-o", help="Output file path (for single clip) or prefix (for multiple clips)")
    
    # Performance options
    parser.add_argument("--workers", "-w", type=int, default=config.MAX_WORKERS,
                       help=f"Number of parallel workers for processing (default: {config.MAX_WORKERS})")
    parser.add_argument("--face-detector", choices=["opencv", "mediapipe", "none"], 
                       default=config.FACE_DETECTOR,
                       help=f"Face detection method (default: {config.FACE_DETECTOR})")
    parser.add_argument("--use-gpu", action="store_true", default=config.CUDA_AVAILABLE,
                       help="Use GPU acceleration if available")
    
    # Logging options
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        default=os.environ.get("TIKTOK_AGENT_LOG_LEVEL", "INFO"),
                        help="Set logging level (default: INFO)")
    
    # Cloud storage options
    cloud_storage_group = parser.add_argument_group('Cloud Storage Options')
    cloud_storage_group.add_argument("--upload-cloud", action="store_true", default=config.CLOUD_STORAGE_ENABLED,
                         help="Upload output clips to cloud storage")
    cloud_storage_group.add_argument("--cloud-provider", choices=["gdrive", "s3"], default=config.CLOUD_STORAGE_PROVIDER,
                         help="Cloud storage provider (default: gdrive)")
    cloud_storage_group.add_argument("--generate-thumbnail", action="store_true", default=False,
                         help="Generate a thumbnail for each extracted clip")

    # Google Drive options
    gdrive_group = parser.add_argument_group('Google Drive Options')
    gdrive_group.add_argument("--gdrive-credentials", default=config.GDRIVE_CREDENTIALS_PATH,
                         help="Path to Google Drive credentials.json file")
    gdrive_group.add_argument("--gdrive-token", default=config.GDRIVE_TOKEN_PATH,
                         help="Path to Google Drive token.json file (will be created if it doesn't exist)")
    gdrive_group.add_argument("--gdrive-folder", default=config.GDRIVE_FOLDER_ID,
                         help="Google Drive folder ID to upload to")

    # AWS S3 options
    s3_group = parser.add_argument_group('AWS S3 Options')
    s3_group.add_argument("--s3-bucket", default=config.S3_BUCKET_NAME,
                         help="AWS S3 bucket name")
    s3_group.add_argument("--s3-region", default=config.S3_REGION,
                         help="AWS S3 region")
    s3_group.add_argument("--s3-access-key", default=config.S3_ACCESS_KEY,
                         help="AWS S3 access key ID")
    s3_group.add_argument("--s3-secret-key", default=config.S3_SECRET_KEY,
                         help="AWS S3 secret access key")
                         
    # Parse arguments
    args = parser.parse_args()
    
    # Update config based on arguments
    config.MAX_WORKERS = args.workers
    config.FACE_DETECTOR = args.face_detector
    config.USE_GPU = args.use_gpu and torch.cuda.is_available()
    
    # Set logging level
    log_level = getattr(logging, args.log_level, logging.INFO)
    set_log_level(log_level)
    
    # If no input method specified, enter interactive mode
    if not args.youtube and not args.file:
        args = interactive_mode(args)
    
    try:
        # Step 1: Get the video
        video_path = get_video(args)
        if not video_path:
            logger.error("Failed to get video. Exiting.")
            return 1
        
        # Step 2: Extract clips
        clip_extractor = ViralClipExtractor()
        
        # Prepare watermark options if enabled
        watermark_options = None
        if args.watermark:
            watermark_options = {
                'enabled': True,
                'type': args.watermark_type,
                'text': args.watermark_text,
                'image': args.watermark_image,
                'position': args.watermark_position,
                'opacity': args.watermark_opacity,
                'padding': args.watermark_padding,
                'text_size': args.watermark_text_size,
                'text_color': args.watermark_text_color
            }
        
        if args.num_clips <= 1:
            # Extract a single clip
            clip_path, start_time, end_time, score = clip_extractor.extract_best_clip(
                video_path, 
                clip_duration=args.duration,
                output_path=args.output
            )
            
            if not clip_path:
                logger.error("Failed to extract clip. Exiting.")
                return 1
                
            logger.info(f"Extracted clip from {start_time:.2f}s to {end_time:.2f}s with score {score:.4f}")

            # Generate thumbnail if requested
            if hasattr(args, 'generate_thumbnail') and args.generate_thumbnail:
                 # Check if we have the clipper instance available or access it through the extractor
                if hasattr(clip_extractor, 'clipper'):
                    thumbnail_time = (start_time + end_time) / 2
                    thumbnail_path = f"{str(Path(clip_path).with_suffix(''))}_thumbnail.jpg"
                    clip_extractor.clipper.generate_thumbnail(video_path, thumbnail_path, thumbnail_time)
            
            # Format the clip to 9:16
            formatter = VideoFormatter()
            formatted_path = formatter.format_to_9_16(
                clip_path,
                method=args.format,
                output_path=args.output,
                watermark_options=watermark_options
            )
            
            if not formatted_path:
                logger.error("Failed to format video. Exiting.")
                return 1
                
            logger.info(f"Successfully created TikTok video: {formatted_path}")
        else:
            # Extract multiple clips
            clips = clip_extractor.extract_multiple_clips(
                video_path, 
                num_clips=args.num_clips,
                clip_duration=args.duration,
                min_gap=args.min_gap,
                output_prefix=args.output
            )
            
            if not clips:
                logger.error("Failed to extract any clips. Exiting.")
                return 1
                
            # Format each clip
            formatter = VideoFormatter()
            formatted_paths = []
            
            for i, (clip_path, start_time, end_time, score) in enumerate(clips):
                # Generate output path for formatted clip
                if args.output:
                    base_path = Path(args.output)
                    fmt_output = str(base_path.parent / f"{base_path.stem}_{i+1}_9x16{base_path.suffix}")
                else:
                    fmt_output = None
                
                # Generate thumbnail if requested
                if hasattr(args, 'generate_thumbnail') and args.generate_thumbnail:
                     if hasattr(clip_extractor, 'clipper'):
                        thumbnail_time = (start_time + end_time) / 2
                        thumbnail_path = f"{str(Path(clip_path).with_suffix(''))}_thumbnail.jpg"
                        clip_extractor.clipper.generate_thumbnail(video_path, thumbnail_path, thumbnail_time)

                # Format the clip
                formatted_path = formatter.format_to_9_16(
                    clip_path,
                    method=args.format,
                    output_path=fmt_output,
                    watermark_options=watermark_options
                )
                
                if formatted_path:
                    formatted_paths.append(formatted_path)
                    logger.info(f"Successfully created TikTok video {i+1}: {formatted_path}")
            
            logger.info(f"Successfully formatted {len(formatted_paths)} out of {len(clips)} clips")
        
        # Upload to cloud storage if enabled
        if args.upload_cloud:
            # Function to handle cloud uploads
            def upload_to_cloud(file_path):
                if args.cloud_provider == "gdrive":
                    result = CloudStorage.upload_to_google_drive(
                        file_path,
                        credentials_path=args.gdrive_credentials,
                        token_path=args.gdrive_token,
                        folder_id=args.gdrive_folder if args.gdrive_folder else None
                    )
                    if result:
                        logger.info(f"Successfully uploaded to Google Drive: {result.get('webViewLink')}")
                    else:
                        logger.error(f"Failed to upload to Google Drive: {file_path}")
                        
                elif args.cloud_provider == "s3":
                    url = CloudStorage.upload_to_aws_s3(
                        file_path,
                        bucket_name=args.s3_bucket,
                        aws_access_key=args.s3_access_key if args.s3_access_key else None,
                        aws_secret_key=args.s3_secret_key if args.s3_secret_key else None,
                        region=args.s3_region if args.s3_region else None
                    )
                    if url:
                        logger.info(f"Successfully uploaded to AWS S3: {url}")
                    else:
                        logger.error(f"Failed to upload to AWS S3: {file_path}")
            
            # Single clip case
            if args.num_clips <= 1:
                if formatted_path:
                    logger.info(f"Uploading clip to {args.cloud_provider}...")
                    upload_to_cloud(formatted_path)
            # Multiple clips case
            else:
                logger.info(f"Uploading {len(formatted_paths)} clips to {args.cloud_provider}...")
                for formatted_path in formatted_paths:
                    upload_to_cloud(formatted_path)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        return 1


def get_video(args):
    """Get the video from YouTube or local file"""
    if args.youtube:
        # Download from YouTube
        downloader = YouTubeDownloader()
        return downloader.download(args.youtube)
    else:
        # Load local file
        loader = VideoFileLoader()
        return loader.load(args.file)


if __name__ == "__main__":
    sys.exit(main())

