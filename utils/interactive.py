import os
import sys
import tkinter as tk
from tkinter import filedialog
import torch
import logging
from utils.logger import set_log_level
import config

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
    
            if opacity_str.strip():
                try:
                    opacity = float(opacity_str)
                    args.watermark_opacity = max(0.0, min(1.0, opacity))
                except ValueError:
                    print(f"Invalid opacity. Using default: {config.WATERMARK_OPACITY}")
    
    # Thumbnail option
    thumbnail_choice = input("\nGenerate thumbnail for each clip? (y/n) [default: y]: ").lower() or "y"
    args.generate_thumbnail = thumbnail_choice in ('y', 'yes')

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
