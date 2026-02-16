// =============================================================================
// DataCollector.cs
// Purpose: Records head tracking + eye tracking data from Quest Pro at 90 Hz
// Attach to: DataCollectionManager (empty GameObject)
// Project: STAR-VP Quest Pro Data Collection
// Unity: 2022.3 LTS + Meta XR All-in-One SDK
//
// OUTPUT FORMAT:
//   Head CSV columns match Wu_MMSys_17 format (Timestamp, PlaybackTime,
//   UnitQuaternion.x/y/z/w, HmdPosition.x/y/z) plus extra Euler & velocity.
//   Eye CSV is new data unique to Quest Pro.
//   Combined CSV merges both for convenience.
//
// DEPENDENCIES:
//   - Meta XR Core SDK (OVRPlugin, OVRManager, OVREyeGaze)
//   - OVRCameraRig in scene with OVRManager configured for eye tracking
// =============================================================================

using UnityEngine;
using UnityEngine.Video;
using System.IO;
using System;
using System.Text;
using System.Collections.Generic;

public class DataCollector : MonoBehaviour
{
    // =========================================================================
    // INSPECTOR REFERENCES — Drag these in the Unity Inspector
    // =========================================================================
    [Header("Scene References")]
    [Tooltip("Drag CenterEyeAnchor from OVRCameraRig/TrackingSpace here")]
    public Transform vrCamera;

    [Tooltip("Drag VideoSphere (which has VideoPlayer component) here")]
    public VideoPlayer videoPlayer;

    [Tooltip("Drag OVRCameraRig root object here")]
    public OVRCameraRig ovrCameraRig;

    // =========================================================================
    // SETTINGS — Configurable per participant/video
    // =========================================================================
    [Header("Session Settings")]
    [Tooltip("Set this for each participant: P001, P002, etc.")]
    public string participantID = "P001";

    [Tooltip("Current video ID — set by VideoManager automatically")]
    public string videoID = "video_0";

    [Tooltip("Check to start/stop recording")]
    public bool recordData = true;

    [Header("Advanced Settings")]
    [Tooltip("Write to disk every N frames (lower = safer but slower)")]
    public int flushInterval = 90; // Flush every ~1 second at 90 Hz

    [Tooltip("Fixation velocity threshold in degrees/second")]
    public float fixationThreshold = 30f;

    // =========================================================================
    // PRIVATE STATE
    // =========================================================================
    private string basePath;
    private StreamWriter headTrackingFile;
    private StreamWriter eyeTrackingFile;
    private StreamWriter combinedFile;

    private float sessionStartTime;
    private int frameCount = 0;
    private bool isRecording = false;
    private bool filesOpen = false;

    // Previous frame data for velocity/fixation calculation
    private Vector3 prevHeadEuler;
    private Vector3 prevGazeDir;
    private float prevTimestamp;
    private float fixationStartTime;
    private bool wasFixating = false;

    // String builder for performance (avoid GC allocations)
    private StringBuilder sb = new StringBuilder(512);

    // Eye tracking state from OVRPlugin
    private OVRPlugin.EyeGazesState _eyeGazesState;

    // =========================================================================
    // UNITY LIFECYCLE
    // =========================================================================

    void Start()
    {
        // Set up output directory
        basePath = Path.Combine(Application.persistentDataPath, "DataCollection");
        if (!Directory.Exists(basePath))
        {
            Directory.CreateDirectory(basePath);
        }

        Debug.Log($"[DataCollector] Data output directory: {basePath}");
        Debug.Log($"[DataCollector] Participant: {participantID}, Video: {videoID}");

        // Check references
        if (vrCamera == null)
        {
            Debug.LogError("[DataCollector] vrCamera is not assigned! Drag CenterEyeAnchor here.");
            recordData = false;
            return;
        }

        if (ovrCameraRig == null)
        {
            Debug.LogWarning("[DataCollector] ovrCameraRig not assigned. Trying to find one...");
            ovrCameraRig = FindObjectOfType<OVRCameraRig>();
            if (ovrCameraRig == null)
            {
                Debug.LogError("[DataCollector] No OVRCameraRig found in scene!");
                recordData = false;
                return;
            }
        }

        // Start eye tracking via OVRPlugin
        bool eyeTrackingStarted = OVRPlugin.StartEyeTracking();
        Debug.Log($"[DataCollector] Eye tracking started: {eyeTrackingStarted}");

        // Load participant ID from PlayerPrefs if set by ParticipantSetup scene
        string savedID = PlayerPrefs.GetString("ParticipantID", "");
        if (!string.IsNullOrEmpty(savedID))
        {
            participantID = savedID;
            Debug.Log($"[DataCollector] Loaded participant ID from prefs: {participantID}");
        }
    }

    void Update()
    {
        if (!recordData) return;

        // Start recording on first frame or when new video starts
        if (!isRecording)
        {
            StartNewRecording();
        }

        frameCount++;
        float currentTimestamp = Time.realtimeSinceStartup - sessionStartTime;

        // Calculate delta time for velocities
        float deltaTime = currentTimestamp - prevTimestamp;
        if (deltaTime <= 0f) deltaTime = 0.0111f; // Fallback: ~90Hz

        // =====================================================================
        // HEAD TRACKING
        // =====================================================================
        Vector3 headPos = vrCamera.position;
        Quaternion headQuat = vrCamera.rotation;
        Vector3 headEuler = vrCamera.eulerAngles;

        // Normalize Euler angles to [-180, 180] range
        float headYaw = NormalizeAngle(headEuler.y);
        float headPitch = NormalizeAngle(headEuler.x);
        float headRoll = NormalizeAngle(headEuler.z);

        // Angular velocity (degrees/second)
        float velYaw = NormalizeAngle(headEuler.y - prevHeadEuler.y) / deltaTime;
        float velPitch = NormalizeAngle(headEuler.x - prevHeadEuler.x) / deltaTime;
        float velRoll = NormalizeAngle(headEuler.z - prevHeadEuler.z) / deltaTime;

        // Video playback time (synced with video player)
        float playbackTime = 0f;
        if (videoPlayer != null && videoPlayer.isPlaying)
        {
            playbackTime = (float)videoPlayer.time;
        }

        // Write head tracking CSV row
        // FORMAT: Matches Wu_MMSys_17 first 9 columns, then extras
        sb.Clear();
        sb.AppendFormat("{0:F4},{1:F3},", currentTimestamp, playbackTime);
        sb.AppendFormat("{0:F6},{1:F6},{2:F6},{3:F6},", headQuat.x, headQuat.y, headQuat.z, headQuat.w);
        sb.AppendFormat("{0:F6},{1:F6},{2:F6},", headPos.x, headPos.y, headPos.z);
        sb.AppendFormat("{0:F4},{1:F4},{2:F4},", headYaw, headPitch, headRoll);
        sb.AppendFormat("{0:F4},{1:F4},{2:F4}", velYaw, velPitch, velRoll);

        if (headTrackingFile != null)
        {
            headTrackingFile.WriteLine(sb.ToString());
        }

        // =====================================================================
        // EYE TRACKING
        // =====================================================================
        bool hasEyeData = false;
        Vector3 gazeDir = Vector3.forward;
        Vector3 gazeOrigin = Vector3.zero;
        float leftPupilDiam = 0f;
        float rightPupilDiam = 0f;
        float leftOpenness = 1f;
        float rightOpenness = 1f;
        float leftConfidence = 0f;
        float rightConfidence = 0f;
        bool bothEyesValid = false;

        // Use OVRPlugin to get eye gaze data
        bool gotEyeState = OVRPlugin.GetEyeGazesState(OVRPlugin.Step.Render, -1, ref _eyeGazesState);

        if (gotEyeState)
        {
            OVRPlugin.EyeGazeState leftEye = _eyeGazesState.EyeGazes[0];
            OVRPlugin.EyeGazeState rightEye = _eyeGazesState.EyeGazes[1];

            bool leftValid = leftEye.IsValid;
            bool rightValid = rightEye.IsValid;
            bothEyesValid = leftValid && rightValid;

            if (leftValid || rightValid)
            {
                hasEyeData = true;

                // Convert OVRPlugin pose to Unity vectors
                if (leftValid && rightValid)
                {
                    // Average both eyes for combined gaze
                    Vector3 leftGazeDir = OVRPluginPoseToDirection(leftEye.Pose);
                    Vector3 rightGazeDir = OVRPluginPoseToDirection(rightEye.Pose);
                    gazeDir = ((leftGazeDir + rightGazeDir) * 0.5f).normalized;

                    Vector3 leftOrigin = OVRPluginPoseToPosition(leftEye.Pose);
                    Vector3 rightOrigin = OVRPluginPoseToPosition(rightEye.Pose);
                    gazeOrigin = (leftOrigin + rightOrigin) * 0.5f;

                    leftConfidence = leftEye.Confidence;
                    rightConfidence = rightEye.Confidence;
                }
                else if (leftValid)
                {
                    gazeDir = OVRPluginPoseToDirection(leftEye.Pose);
                    gazeOrigin = OVRPluginPoseToPosition(leftEye.Pose);
                    leftConfidence = leftEye.Confidence;
                }
                else
                {
                    gazeDir = OVRPluginPoseToDirection(rightEye.Pose);
                    gazeOrigin = OVRPluginPoseToPosition(rightEye.Pose);
                    rightConfidence = rightEye.Confidence;
                }

                // Eye openness (0=closed, 1=open) — available on Quest Pro
                leftOpenness = leftValid ? leftEye.Confidence : 0f;
                rightOpenness = rightValid ? rightEye.Confidence : 0f;

                // Transform gaze direction to world space
                Transform trackingSpace = ovrCameraRig.trackingSpace;
                if (trackingSpace != null)
                {
                    gazeDir = trackingSpace.TransformDirection(gazeDir);
                    gazeOrigin = trackingSpace.TransformPoint(gazeOrigin);
                }
            }
        }

        // Convert gaze to spherical coordinates (yaw/pitch)
        float gazeYaw = 0f;
        float gazePitch = 0f;
        if (hasEyeData)
        {
            gazeYaw = Mathf.Atan2(gazeDir.x, gazeDir.z) * Mathf.Rad2Deg;
            gazePitch = Mathf.Asin(Mathf.Clamp(gazeDir.y, -1f, 1f)) * Mathf.Rad2Deg;
        }

        // Fixation detection (I-VT: velocity threshold)
        bool isFixating = false;
        float fixationDuration = 0f;
        if (hasEyeData)
        {
            float gazeAngularVelocity = Vector3.Angle(gazeDir, prevGazeDir) / deltaTime;
            isFixating = gazeAngularVelocity < fixationThreshold;

            if (isFixating && wasFixating)
            {
                fixationDuration = (currentTimestamp - fixationStartTime) * 1000f; // ms
            }
            else if (isFixating && !wasFixating)
            {
                fixationStartTime = currentTimestamp;
                fixationDuration = 0f;
            }
            wasFixating = isFixating;
        }

        // Write eye tracking CSV row
        sb.Clear();
        sb.AppendFormat("{0:F4},{1:F3},", currentTimestamp, playbackTime);
        sb.AppendFormat("{0:F6},{1:F6},{2:F6},", gazeDir.x, gazeDir.y, gazeDir.z);
        sb.AppendFormat("{0:F6},{1:F6},{2:F6},", gazeOrigin.x, gazeOrigin.y, gazeOrigin.z);
        sb.AppendFormat("{0:F4},{1:F4},", gazeYaw, gazePitch);
        sb.AppendFormat("{0:F4},{1:F4},", leftPupilDiam, rightPupilDiam);
        sb.AppendFormat("{0:F4},{1:F4},", leftOpenness, rightOpenness);
        sb.AppendFormat("{0:F4},{1:F4},", leftConfidence, rightConfidence);
        sb.AppendFormat("{0},{1:F1},{2}", isFixating ? 1 : 0, fixationDuration, bothEyesValid ? 1 : 0);

        if (eyeTrackingFile != null)
        {
            eyeTrackingFile.WriteLine(sb.ToString());
        }

        // =====================================================================
        // COMBINED DATA
        // =====================================================================
        float gazeRelativeH = hasEyeData ? NormalizeAngle(gazeYaw - headYaw) : 0f;
        float gazeRelativeV = hasEyeData ? NormalizeAngle(gazePitch - headPitch) : 0f;
        float avgPupil = (leftPupilDiam + rightPupilDiam) / 2f;
        float eyeHeadOffset = Mathf.Sqrt(gazeRelativeH * gazeRelativeH + gazeRelativeV * gazeRelativeV);

        // Absolute gaze in world (for viewport prediction)
        float absoluteGazeYaw = hasEyeData ? gazeYaw : headYaw;
        float absoluteGazePitch = hasEyeData ? gazePitch : headPitch;

        sb.Clear();
        sb.AppendFormat("{0:F4},{1:F3},", currentTimestamp, playbackTime);
        sb.AppendFormat("{0:F4},{1:F4},{2:F4},", headYaw, headPitch, headRoll);
        sb.AppendFormat("{0:F4},{1:F4},", gazeYaw, gazePitch);
        sb.AppendFormat("{0:F4},{1:F4},", gazeRelativeH, gazeRelativeV);
        sb.AppendFormat("{0:F4},{1:F4},", avgPupil, eyeHeadOffset);
        sb.AppendFormat("{0:F4},{1:F4}", absoluteGazeYaw, absoluteGazePitch);

        if (combinedFile != null)
        {
            combinedFile.WriteLine(sb.ToString());
        }

        // =====================================================================
        // UPDATE PREVIOUS FRAME STATE
        // =====================================================================
        prevHeadEuler = headEuler;
        prevGazeDir = gazeDir;
        prevTimestamp = currentTimestamp;

        // Periodic flush to prevent data loss on crash
        if (frameCount % flushInterval == 0)
        {
            FlushAllFiles();
        }

        // =====================================================================
        // CHECK IF VIDEO ENDED
        // =====================================================================
        if (videoPlayer != null && !videoPlayer.isPlaying &&
            videoPlayer.time > 0.5 && videoPlayer.time >= videoPlayer.length - 0.5)
        {
            StopRecording();
        }
    }

    // =========================================================================
    // PUBLIC METHODS (called by VideoManager)
    // =========================================================================

    /// <summary>
    /// Call this when switching to a new video.
    /// Closes current files and prepares for new recording.
    /// </summary>
    public void PrepareForNewVideo(string newVideoID)
    {
        // Close current recording if active
        if (isRecording)
        {
            StopRecording();
        }

        videoID = newVideoID;
        Debug.Log($"[DataCollector] Prepared for video: {videoID}");
    }

    /// <summary>
    /// Starts recording. Called automatically on first Update frame.
    /// </summary>
    public void StartNewRecording()
    {
        if (isRecording) return;

        string dateStamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        string prefix = $"{participantID}_{videoID}_{dateStamp}";

        // Create participant directory
        string participantPath = Path.Combine(basePath, participantID);
        if (!Directory.Exists(participantPath))
        {
            Directory.CreateDirectory(participantPath);
        }

        // ---- Head tracking file ----
        string headPath = Path.Combine(participantPath, $"head_{prefix}.csv");
        headTrackingFile = new StreamWriter(headPath, false, Encoding.UTF8);
        // Wu_MMSys_17 compatible columns + extras
        headTrackingFile.WriteLine(
            "Timestamp,PlaybackTime," +
            "UnitQuaternion.x,UnitQuaternion.y,UnitQuaternion.z,UnitQuaternion.w," +
            "HmdPosition.x,HmdPosition.y,HmdPosition.z," +
            "EulerYaw,EulerPitch,EulerRoll," +
            "VelYaw,VelPitch,VelRoll");

        // ---- Eye tracking file ----
        string eyePath = Path.Combine(participantPath, $"eye_{prefix}.csv");
        eyeTrackingFile = new StreamWriter(eyePath, false, Encoding.UTF8);
        eyeTrackingFile.WriteLine(
            "Timestamp,PlaybackTime," +
            "GazeDir.x,GazeDir.y,GazeDir.z," +
            "GazeOrigin.x,GazeOrigin.y,GazeOrigin.z," +
            "GazeYaw,GazePitch," +
            "LeftPupilDiam,RightPupilDiam," +
            "LeftOpenness,RightOpenness," +
            "LeftConfidence,RightConfidence," +
            "IsFixating,FixationDurationMs,BothEyesValid");

        // ---- Combined file ----
        string combinedPath = Path.Combine(participantPath, $"combined_{prefix}.csv");
        combinedFile = new StreamWriter(combinedPath, false, Encoding.UTF8);
        combinedFile.WriteLine(
            "Timestamp,PlaybackTime," +
            "HeadYaw,HeadPitch,HeadRoll," +
            "GazeYaw,GazePitch," +
            "GazeRelativeH,GazeRelativeV," +
            "AvgPupil,EyeHeadOffset," +
            "AbsoluteGazeYaw,AbsoluteGazePitch");

        // Initialize state
        sessionStartTime = Time.realtimeSinceStartup;
        frameCount = 0;
        prevTimestamp = 0f;
        prevHeadEuler = vrCamera.eulerAngles;
        prevGazeDir = Vector3.forward;
        fixationStartTime = 0f;
        wasFixating = false;
        isRecording = true;
        filesOpen = true;

        Debug.Log($"[DataCollector] Recording started: {headPath}");
    }

    /// <summary>
    /// Stops recording and closes all files safely.
    /// </summary>
    public void StopRecording()
    {
        if (!filesOpen) return;

        isRecording = false;
        recordData = false;

        FlushAllFiles();
        CloseAllFiles();

        Debug.Log($"[DataCollector] Recording stopped. {frameCount} frames saved for {participantID}/{videoID}");
        Debug.Log($"[DataCollector] Files at: {Path.Combine(basePath, participantID)}");
    }

    // =========================================================================
    // HELPER METHODS
    // =========================================================================

    /// <summary>
    /// Normalize angle to [-180, +180] range.
    /// </summary>
    private float NormalizeAngle(float angle)
    {
        while (angle > 180f) angle -= 360f;
        while (angle < -180f) angle += 360f;
        return angle;
    }

    /// <summary>
    /// Convert OVRPlugin.Posef orientation to a forward direction vector.
    /// OVRPlugin uses a right-handed coordinate system; Unity is left-handed.
    /// The gaze direction is the forward vector of the pose rotation.
    /// </summary>
    private Vector3 OVRPluginPoseToDirection(OVRPlugin.Posef pose)
    {
        // Convert OVRPlugin quaternion to Unity quaternion
        // OVRPlugin: right-handed (x-right, y-up, z-backward)
        // Unity: left-handed (x-right, y-up, z-forward)
        Quaternion rotation = new Quaternion(
            pose.Orientation.x,
            pose.Orientation.y,
            -pose.Orientation.z,
            -pose.Orientation.w
        );

        // The gaze direction is the forward vector of this rotation
        return rotation * Vector3.forward;
    }

    /// <summary>
    /// Convert OVRPlugin.Posef position to Unity Vector3.
    /// </summary>
    private Vector3 OVRPluginPoseToPosition(OVRPlugin.Posef pose)
    {
        // Flip Z axis for Unity's left-handed coordinate system
        return new Vector3(
            pose.Position.x,
            pose.Position.y,
            -pose.Position.z
        );
    }

    private void FlushAllFiles()
    {
        try
        {
            headTrackingFile?.Flush();
            eyeTrackingFile?.Flush();
            combinedFile?.Flush();
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[DataCollector] Flush error: {e.Message}");
        }
    }

    private void CloseAllFiles()
    {
        try
        {
            if (headTrackingFile != null) { headTrackingFile.Close(); headTrackingFile = null; }
            if (eyeTrackingFile != null) { eyeTrackingFile.Close(); eyeTrackingFile = null; }
            if (combinedFile != null) { combinedFile.Close(); combinedFile = null; }
            filesOpen = false;
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[DataCollector] Close error: {e.Message}");
        }
    }

    // =========================================================================
    // UNITY CALLBACKS — Ensure data is saved even on unexpected exit
    // =========================================================================

    void OnApplicationPause(bool pauseStatus)
    {
        if (pauseStatus && isRecording)
        {
            FlushAllFiles();
            Debug.Log("[DataCollector] App paused — data flushed.");
        }
    }

    void OnApplicationQuit()
    {
        StopRecording();
    }

    void OnDestroy()
    {
        StopRecording();
    }
}
