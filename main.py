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
                    from pathlib import Path
                    base_path = Path(args.output)
                    fmt_output = str(base_path.parent / f"{base_path.stem}_{i+1}_9x16{base_path.suffix}")
                else:
                    fmt_output = None
                
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


def interactive_mode(args):
    """Interactive mode for selecting input options"""
    print("\n========== TikTok Agent ==========")
    print("Extract viral clips from videos and format for TikTok")
    print("====================================\n")
    
    # Select input source
    print("Select input source:")
    print("1. Local video file")
    print("2. YouTube video")
    
    choice = ""
    while choice not in ["1", "2"]:
        choice = input("Enter your choice (1/2): ")
    
    # Process choice
    if choice == "1":
        try:
            # Initialize tkinter without showing a window
            root = tk.Tk()
            root.attributes("-topmost", True)  # Make sure dialog appears on top
            root.withdraw()  # Hide the root window
            
            # Force focus to ensure dialog appears
            root.focus_force()
            
            print("Opening file selector dialog... (might appear behind current window)")
            
            # Show the file dialog
            file_path = filedialog.askopenfilename(
                parent=root,
                title="Select Video File",
                filetypes=[
                    ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm"),
                    ("All files", "*.*")
                ]
            )
            
            # Destroy the tkinter instance
            root.destroy()
            
            if not file_path:
                print("No file selected. Exiting.")
                sys.exit(0)
                
            args.file = file_path
            print(f"Selected file: {file_path}")
            
        except Exception as e:
            print(f"Error opening file dialog: {str(e)}")
            print("Please enter the file path manually:")
            args.file = input("File path: ")
        
    else:  # YouTube
        url = input("Enter YouTube URL: ")
        args.youtube = url
    
    # Number of clips
    num_clips_str = input("Enter number of clips to extract (default: 1): ")
    if num_clips_str.strip():
        try:
            args.num_clips = int(num_clips_str)
            if args.num_clips < 1:
                print("Number of clips must be at least 1. Setting to 1.")
                args.num_clips = 1
        except ValueError:
            print("Invalid number of clips. Using default: 1")
    
    # Duration
    duration_str = input(f"Enter clip duration in seconds (default {config.DEFAULT_CLIP_DURATION}): ")
    if duration_str.strip():
        try:
            args.duration = float(duration_str)
        except ValueError:
            print(f"Invalid duration. Using default: {config.DEFAULT_CLIP_DURATION}")
    
    # Format method
    print("\nSelect format method:")
    print("1. Crop (center crop to 9:16)")
    print("2. Blur (blurred background)")
    print("3. Bars (black bars)")
    
    format_choice = ""
    while format_choice not in ["1", "2", "3"]:
        format_choice = input("Enter your choice (1/2/3) [default: 1]: ") or "1"
    
    format_map = {"1": "crop", "2": "blur", "3": "bars"}
    args.format = format_map[format_choice]
    
    # Watermark options
    watermark_choice = input("\nAdd watermark to video? (y/n) [default: n]: ").lower() or "n"
    args.watermark = watermark_choice in ('y', 'yes')
    
    if args.watermark:
        print("\nSelect watermark type:")
        print("1. Text")
        print("2. Image")
        
        wm_type_choice = input("Enter your choice (1/2) [default: 1]: ") or "1"
        args.watermark_type = "text" if wm_type_choice == "1" else "image"
        
        if args.watermark_type == "text":
            args.watermark_text = input(f"Enter watermark text [default: {config.WATERMARK_TEXT}]: ") or config.WATERMARK_TEXT
            args.watermark_text_size = int(input(f"Enter text size [default: {config.WATERMARK_TEXT_SIZE}]: ") or config.WATERMARK_TEXT_SIZE)
            args.watermark_text_color = input(f"Enter text color [default: {config.WATERMARK_TEXT_COLOR}]: ") or config.WATERMARK_TEXT_COLOR
        else:
            # For image watermark
            try:
                root = tk.Tk()
                root.attributes("-topmost", True)
                root.withdraw()
                root.focus_force()
                
                image_path = filedialog.askopenfilename(
                    parent=root,
                    title="Select Watermark Image",
                    filetypes=[
                        ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp"),
                        ("All files", "*.*")
                    ]
                )
                
                root.destroy()
                
                if image_path:
                    args.watermark_image = image_path
                    print(f"Selected watermark image: {image_path}")
                else:
                    print("No image selected. Watermark will be disabled.")
                    args.watermark = False
                    
            except Exception as e:
                print(f"Error opening file dialog: {str(e)}")
                image_path = input("Enter path to watermark image: ")
                if os.path.exists(image_path):
                    args.watermark_image = image_path
                else:
                    print("Invalid image path. Watermark will be disabled.")
                    args.watermark = False
        
        if args.watermark:
            # Position
            print("\nSelect watermark position:")
            print("1. Top-left")
            print("2. Top-right")
            print("3. Bottom-left")
            print("4. Bottom-right (default)")
            print("5. Center")
            
            pos_choice = input("Enter your choice (1-5) [default: 4]: ") or "4"
            pos_map = {
                "1": "top-left",
                "2": "top-right",
                "3": "bottom-left", 
                "4": "bottom-right",
                "5": "center"
            }
            args.watermark_position = pos_map.get(pos_choice, "bottom-right")
            
            # Opacity
            opacity_str = input(f"Enter watermark opacity (0.0-1.0) [default: {config.WATERMARK_OPACITY}]: ")
            if opacity_str.strip():
                try:
                    opacity = float(opacity_str)
                    args.watermark_opacity = max(0.0, min(1.0, opacity))
                except ValueError:
                    print(f"Invalid opacity. Using default: {config.WATERMARK_OPACITY}")
    
    if args.num_clips > 1:
        # Min gap option for multiple clips
        min_gap_str = input("Enter minimum gap between clips in seconds (default: 1.0): ")
        if min_gap_str.strip():
            try:
                args.min_gap = float(min_gap_str)
                if args.min_gap < 0:
                    print("Gap must be non-negative. Setting to 0.")
                    args.min_gap = 0
            except ValueError:
                print("Invalid gap value. Using default: 1.0")
    
    # Face detection method
    print("\nSelect face detection method:")
    print("1. MediaPipe (default, more accurate but may be slower)")
    print("2. OpenCV (faster but less accurate)")
    print("3. None (disable face detection)")
    
    face_choice = input("Enter your choice (1/2/3) [default: 1]: ") or "1"  # Default to "1" if empty
    face_map = {"1": "mediapipe", "2": "opencv", "3": "none"}
    
    if face_choice not in face_map:
        print(f"Invalid choice: {face_choice}. Using default: MediaPipe")
        face_choice = "1"
        
    args.face_detector = face_map[face_choice]
    config.FACE_DETECTOR = args.face_detector
    print(f"Using face detection method: {args.face_detector}")
    
    # GPU acceleration
    if torch.cuda.is_available():
        gpu_choice = input("\nUse GPU acceleration if available? (y/n) [default: y]: ").lower() or "y"
        args.use_gpu = gpu_choice not in ('n', 'no')
        config.USE_GPU = args.use_gpu
    else:
        print("\nGPU acceleration not available on this system.")
        args.use_gpu = False
        config.USE_GPU = False
    
    # Workers for parallel processing
    cpu_count = os.cpu_count() or 4
    workers_str = input(f"\nNumber of parallel workers (1-{cpu_count}) [default: {config.MAX_WORKERS}]: ") or str(config.MAX_WORKERS)
    try:
        args.workers = int(workers_str)
        if args.workers < 1:
            args.workers = 1
            print("Workers must be at least 1. Setting to 1.")
        elif args.workers > cpu_count:
            args.workers = cpu_count
            print(f"Workers limited to available CPU count: {args.workers}")
    except ValueError:
        print(f"Invalid worker count. Using default: {config.MAX_WORKERS}")
        args.workers = config.MAX_WORKERS
        
    config.MAX_WORKERS = args.workers
    print(f"Using {args.workers} parallel workers")
    
    # Advanced logging options
    show_advanced = input("\nShow advanced logging options? (y/n) [default: n]: ").lower() or "n"
    if show_advanced in ('y', 'yes'):
        print("\nSelect logging level:")
        print("1. DEBUG (very verbose)")
        print("2. INFO (default)")
        print("3. WARNING (fewer messages)")
        print("4. ERROR (only errors)")
        print("5. CRITICAL (only critical errors)")
        
        log_choice = input("Enter your choice (1-5) [default: 2]: ") or "2"
            
        log_map = {"1": "DEBUG", "2": "INFO", "3": "WARNING", "4": "ERROR", "5": "CRITICAL"}
        if log_choice not in log_map:
            print(f"Invalid choice: {log_choice}. Using default: INFO")
            log_choice = "2"
            
        args.log_level = log_map[log_choice]
        log_level = getattr(logging, args.log_level)
        set_log_level(log_level)
        print(f"Logging level set to: {args.log_level}")
    
    # Cloud storage options
    print("\n=== Cloud Storage Options ===")
    upload_choice = input("Upload output clips to cloud storage? (y/n) [default: n]: ").lower() or "n"
    args.upload_cloud = upload_choice in ('y', 'yes')

    if args.upload_cloud:
        print("\nSelect cloud storage provider:")
        print("1. Google Drive")
        print("2. AWS S3")
        
        provider_choice = input("Enter your choice (1/2) [default: 1]: ") or "1"
        args.cloud_provider = "gdrive" if provider_choice == "1" else "s3"
        
        if args.cloud_provider == "gdrive":
            print("\n=== Google Drive Options ===")
            
            # Check for existing credentials
            default_creds = os.path.abspath(config.GDRIVE_CREDENTIALS_PATH)
            if not os.path.exists(default_creds):
                default_creds = str(config.PROJECT_ROOT / "credentials.json")
                
            if os.path.exists(default_creds):
                use_existing = input(f"Use existing credentials file? ({default_creds}) (y/n) [default: y]: ").lower() or "y"
                if use_existing in ("y", "yes"):
                    args.gdrive_credentials = default_creds
                else:
                    # Ask for credentials file path
                    creds_path = input("Enter path to Google Drive credentials.json file: ")
                    args.gdrive_credentials = os.path.abspath(creds_path) if creds_path else default_creds
            else:
                print("No default credentials file found.")
                creds_path = input("Enter path to Google Drive credentials.json file: ")
                args.gdrive_credentials = os.path.abspath(creds_path)
            
                # Ask for folder ID with the default value shown
            default_folder = config.GDRIVE_FOLDER_ID
            folder_prompt = f"Enter Google Drive folder ID [default: {default_folder}]: " if default_folder else "Enter Google Drive folder ID (leave empty for root folder): "
            args.gdrive_folder = input(folder_prompt) or default_folder
            
        else:  # AWS S3
            print("\n=== AWS S3 Options ===")
            args.s3_bucket = input("Enter S3 bucket name: ")
            args.s3_region = input("Enter AWS region (e.g., us-west-1) [optional]: ") or ""
            
            use_env_creds = input("Use AWS credentials from environment/config files? (y/n) [default: y]: ").lower() or "y"
            if use_env_creds not in ("y", "yes"):
                args.s3_access_key = input("Enter AWS access key ID: ")
                args.s3_secret_key = input("Enter AWS secret access key: ")

    # Summary of settings
    print("\n=== Configuration Summary ===")
    print(f"Face detection: {args.face_detector}")
    print(f"GPU acceleration: {'Enabled' if args.use_gpu else 'Disabled'}")
    print(f"Parallel workers: {args.workers}")
    print(f"Clip count: {args.num_clips}")
    print(f"Clip duration: {args.duration}s")
    print(f"Format method: {args.format}")
    print(f"Watermark: {'Enabled' if args.watermark else 'Disabled'}")
    if args.watermark:
        print(f"  Type: {args.watermark_type}")
        if args.watermark_type == "text":
            print(f"  Text: {args.watermark_text}")
        else:
            print(f"  Image: {args.watermark_image}")
        print(f"  Position: {args.watermark_position}")
    print(f"Cloud storage: {'Enabled' if args.upload_cloud else 'Disabled'}")
    if args.upload_cloud:
        print(f"  Provider: {args.cloud_provider}")
        if args.cloud_provider == "gdrive":
            print(f"  Credentials path: {args.gdrive_credentials}")
            print(f"  Folder ID: {args.gdrive_folder or 'Root folder'}")
        else:
            print(f"  S3 bucket: {args.s3_bucket}")
            print(f"  Region: {args.s3_region or 'Default'}")
    print("==========================\n")
    
    return args


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