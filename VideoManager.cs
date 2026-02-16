// =============================================================================
// VideoManager.cs
// Purpose: Manages sequential playback of multiple 360° videos on Quest Pro
// Attach to: DataCollectionManager (same GameObject as DataCollector)
// Project: STAR-VP Quest Pro Data Collection
// Unity: 2022.3 LTS + Meta XR All-in-One SDK
//
// HOW IT WORKS:
//   1. Loads videos from a list of file URLs (stored on Quest Pro's internal storage)
//   2. Plays them one after another
//   3. Notifies DataCollector when switching videos (so it creates new CSV files)
//   4. Provides a configurable pause between videos
//
// VIDEO FILE LOCATION ON QUEST PRO:
//   Copy your .mp4 files to: /sdcard/Movies/VRStudy/
//   Then set URLs as: file:///sdcard/Movies/VRStudy/video_0.mp4
// =============================================================================

using UnityEngine;
using UnityEngine.Video;
using System.Collections;
using System.Collections.Generic;

public class VideoManager : MonoBehaviour
{
    // =========================================================================
    // INSPECTOR REFERENCES
    // =========================================================================
    [Header("References")]
    [Tooltip("Drag the VideoSphere (which has VideoPlayer) here")]
    public VideoPlayer videoPlayer;

    [Tooltip("Drag the DataCollectionManager (which has DataCollector) here")]
    public DataCollector dataCollector;

    // =========================================================================
    // VIDEO CONFIGURATION
    // =========================================================================
    [Header("Video List")]
    [Tooltip("Full URLs to video files on Quest Pro storage")]
    public List<string> videoURLs = new List<string>();

    [Tooltip("Video IDs matching videoURLs (same order, same count)")]
    public List<string> videoIDs = new List<string>();

    [Header("Playback Settings")]
    [Tooltip("Seconds to wait between videos (rest period)")]
    public float pauseBetweenVideos = 3.0f;

    [Tooltip("Auto-start first video on scene load")]
    public bool autoStart = true;

    // =========================================================================
    // STATE
    // =========================================================================
    private int currentVideoIndex = -1;
    private bool isTransitioning = false;
    private bool allVideosComplete = false;
    private bool videoPreparing = false;
    private bool videoEndDetected = false;

    // =========================================================================
    // UNITY LIFECYCLE
    // =========================================================================

    void Start()
    {
        // Validate configuration
        if (videoPlayer == null)
        {
            Debug.LogError("[VideoManager] VideoPlayer not assigned!");
            return;
        }

        if (dataCollector == null)
        {
            Debug.LogError("[VideoManager] DataCollector not assigned!");
            return;
        }

        if (videoURLs.Count == 0)
        {
            Debug.LogWarning("[VideoManager] No videos configured! Add URLs in Inspector.");
            // Add default test videos if none configured
            SetupDefaultVideos();
        }

        if (videoURLs.Count != videoIDs.Count)
        {
            Debug.LogError($"[VideoManager] Mismatch: {videoURLs.Count} URLs but {videoIDs.Count} IDs!");
            // Auto-generate IDs if missing
            while (videoIDs.Count < videoURLs.Count)
            {
                videoIDs.Add($"video_{videoIDs.Count}");
            }
        }

        // Configure video player
        videoPlayer.source = VideoSource.Url;
        videoPlayer.playOnAwake = false;
        videoPlayer.renderMode = VideoRenderMode.MaterialOverride;
        videoPlayer.skipOnDrop = true;
        videoPlayer.isLooping = false;

        // Register event callbacks
        videoPlayer.prepareCompleted += OnVideoPrepared;
        videoPlayer.errorReceived += OnVideoError;
        videoPlayer.loopPointReached += OnVideoFinished;

        Debug.Log($"[VideoManager] Configured with {videoURLs.Count} videos.");

        // Auto-start
        if (autoStart && videoURLs.Count > 0)
        {
            StartCoroutine(PlayNextVideo());
        }
    }

    void Update()
    {
        // Backup end detection (loopPointReached doesn't always fire on Quest)
        if (videoPlayer != null && !isTransitioning && !allVideosComplete &&
            currentVideoIndex >= 0 && !videoPreparing)
        {
            if (videoPlayer.isPrepared && !videoPlayer.isPlaying &&
                videoPlayer.time > 1.0 && videoPlayer.length > 0 &&
                videoPlayer.time >= videoPlayer.length - 0.5)
            {
                if (!videoEndDetected)
                {
                    videoEndDetected = true;
                    Debug.Log($"[VideoManager] Video end detected via Update() for video {currentVideoIndex}");
                    OnVideoFinished(videoPlayer);
                }
            }
        }
    }

    void OnDestroy()
    {
        // Clean up event handlers
        if (videoPlayer != null)
        {
            videoPlayer.prepareCompleted -= OnVideoPrepared;
            videoPlayer.errorReceived -= OnVideoError;
            videoPlayer.loopPointReached -= OnVideoFinished;
        }
    }

    // =========================================================================
    // VIDEO PLAYBACK CONTROL
    // =========================================================================

    /// <summary>
    /// Plays the next video in the sequence.
    /// </summary>
    private IEnumerator PlayNextVideo()
    {
        isTransitioning = true;
        videoEndDetected = false;
        currentVideoIndex++;

        if (currentVideoIndex >= videoURLs.Count)
        {
            // All videos complete
            allVideosComplete = true;
            isTransitioning = false;
            Debug.Log("[VideoManager] === ALL VIDEOS COMPLETE ===");

            // Stop data collection
            if (dataCollector != null)
            {
                dataCollector.StopRecording();
            }
            yield break;
        }

        // Pause between videos (not before first video)
        if (currentVideoIndex > 0)
        {
            Debug.Log($"[VideoManager] Pausing {pauseBetweenVideos}s before next video...");
            yield return new WaitForSeconds(pauseBetweenVideos);
        }

        // Notify data collector to prepare for new video
        string videoID = videoIDs[currentVideoIndex];
        if (dataCollector != null)
        {
            dataCollector.PrepareForNewVideo(videoID);
            dataCollector.recordData = true;
        }

        // Load and prepare video
        string url = videoURLs[currentVideoIndex];
        Debug.Log($"[VideoManager] Loading video {currentVideoIndex + 1}/{videoURLs.Count}: {videoID}");
        Debug.Log($"[VideoManager] URL: {url}");

        videoPreparing = true;
        videoPlayer.url = url;
        videoPlayer.Prepare();

        // Wait for preparation (with timeout)
        float prepareTimeout = 30f;
        float prepareStart = Time.time;
        while (!videoPlayer.isPrepared && (Time.time - prepareStart) < prepareTimeout)
        {
            yield return null;
        }

        if (!videoPlayer.isPrepared)
        {
            Debug.LogError($"[VideoManager] Video preparation timed out for: {url}");
            videoPreparing = false;
            isTransitioning = false;
            // Try next video
            StartCoroutine(PlayNextVideo());
            yield break;
        }

        videoPreparing = false;
        isTransitioning = false;
    }

    // =========================================================================
    // EVENT CALLBACKS
    // =========================================================================

    private void OnVideoPrepared(VideoPlayer vp)
    {
        Debug.Log($"[VideoManager] Video prepared: {videoIDs[currentVideoIndex]} " +
                  $"(Duration: {vp.length:F1}s, Resolution: {vp.width}x{vp.height})");
        vp.Play();
    }

    private void OnVideoFinished(VideoPlayer vp)
    {
        if (isTransitioning || allVideosComplete) return;

        Debug.Log($"[VideoManager] Video finished: {videoIDs[currentVideoIndex]} " +
                  $"(Played: {vp.time:F1}s / {vp.length:F1}s)");

        // Stop data collection for this video
        if (dataCollector != null)
        {
            dataCollector.StopRecording();
        }

        // Move to next video
        StartCoroutine(PlayNextVideo());
    }

    private void OnVideoError(VideoPlayer vp, string message)
    {
        Debug.LogError($"[VideoManager] Video error: {message}");
        Debug.LogError($"[VideoManager] Failed URL: {vp.url}");

        // Skip to next video on error
        if (!isTransitioning)
        {
            StartCoroutine(PlayNextVideo());
        }
    }

    // =========================================================================
    // DEFAULT VIDEO SETUP
    // =========================================================================

    /// <summary>
    /// Sets up default video paths. Modify these to match your actual video files.
    /// Videos should be copied to /sdcard/Movies/VRStudy/ on the Quest Pro.
    /// </summary>
    private void SetupDefaultVideos()
    {
        // These match the Wu_MMSys_17 video numbering
        // IMPORTANT: Change these to match your actual video filenames!

        string basePath = "file:///sdcard/Movies/VRStudy/";

        string[] defaultVideos = new string[]
        {
            "video_0.mp4",
            "video_1.mp4",
            "video_2.mp4",
            "video_3.mp4",
            "video_4.mp4",
            "video_5.mp4",
            "video_6.mp4",
            "video_7.mp4",
            "video_8.mp4"
        };

        videoURLs.Clear();
        videoIDs.Clear();

        for (int i = 0; i < defaultVideos.Length; i++)
        {
            videoURLs.Add(basePath + defaultVideos[i]);
            videoIDs.Add($"video_{i}");
        }

        Debug.Log($"[VideoManager] Setup {videoURLs.Count} default videos.");
    }

    // =========================================================================
    // PUBLIC UTILITY METHODS
    // =========================================================================

    /// <summary>
    /// Get current progress info.
    /// </summary>
    public string GetProgressInfo()
    {
        if (allVideosComplete) return "All videos complete";
        if (currentVideoIndex < 0) return "Not started";
        return $"Video {currentVideoIndex + 1}/{videoURLs.Count}: {videoIDs[currentVideoIndex]}";
    }

    /// <summary>
    /// Get total number of videos.
    /// </summary>
    public int GetTotalVideos()
    {
        return videoURLs.Count;
    }

    /// <summary>
    /// Get current video index (0-based, -1 if not started).
    /// </summary>
    public int GetCurrentVideoIndex()
    {
        return currentVideoIndex;
    }
}
