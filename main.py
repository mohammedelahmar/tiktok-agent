import argparse
import sys
import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import config
from utils.logger import logger
from core.downloader import YouTubeDownloader
from core.file_loader import VideoFileLoader
from core.viral_clip_extractor import ViralClipExtractor
from core.formatter import VideoFormatter


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
    
    # Format options
    parser.add_argument("--format", "-fmt", choices=["crop", "blur", "bars"], default="crop",
                        help="Method for formatting to 9:16 ratio (default: crop)")
    
    # Output options
    parser.add_argument("--output", "-o", help="Output file path")
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no input method specified, enter interactive mode
    if not args.youtube and not args.file:
        args = interactive_mode(args)
    
    try:
        # Step 1: Get the video
        video_path = get_video(args)
        if not video_path:
            logger.error("Failed to get video. Exiting.")
            return 1
        
        # Step 2: Extract the best clip
        clip_extractor = ViralClipExtractor()
        clip_path, start_time, end_time, score = clip_extractor.extract_best_clip(
            video_path, 
            clip_duration=args.duration
        )
        
        if not clip_path:
            logger.error("Failed to extract clip. Exiting.")
            return 1
            
        logger.info(f"Extracted clip from {start_time:.2f}s to {end_time:.2f}s with score {score:.4f}")
        
        # Step 3: Format to 9:16
        formatter = VideoFormatter()
        output_path = args.output
        formatted_path = formatter.format_to_9_16(
            clip_path,
            method=args.format,
            output_path=output_path
        )
        
        if not formatted_path:
            logger.error("Failed to format video. Exiting.")
            return 1
            
        logger.info(f"Successfully created TikTok video: {formatted_path}")
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
                    ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
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