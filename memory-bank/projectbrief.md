# Project Brief: Reverse Archive Search

## Overview
A Python-based desktop application that enables reverse image search within Discord channel archives, specifically designed for the 2b2t Minecraft server's map art community.

## Core Problem
Users often see map art images and want to find the original Discord message that contains that image within the massive map art archive. Manual searching through thousands of messages is impractical.

## Solution
A Tkinter GUI application that:
1. Accepts user image input
2. Processes Discord channel JSON data
3. Downloads and analyzes all attached images from messages
4. Uses computer vision to find the closest matching image
5. Returns the original message with metadata

## Key Features
- **Image Input**: User provides an image file for searching
- **JSON Processing**: Reads Discord channel export JSON files
- **Image Comparison**: Uses CLIP or similar CV models for semantic image matching
- **Result Display**: Shows the original message, author, timestamp, and context
- **GUI Interface**: User-friendly Tkinter desktop application

## Success Criteria
- Accurately identifies source messages for uploaded images
- Handles large Discord archives (99MB+ JSON files)
- Provides fast, reliable image matching
- Intuitive user interface for non-technical users

## Target Users
- 2b2t map art community members
- Discord archive researchers
- Image provenance investigators 