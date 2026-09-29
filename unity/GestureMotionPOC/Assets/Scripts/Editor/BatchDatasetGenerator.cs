using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

/// <summary>
/// Batch renderer for synthetic gesture-recognition training data.
///
/// Loop order (outer -> inner), matching the intended data layout:
///   for each Environment:
///     for each Character:
///       for each Motion clip:
///         render frames -> BatchOutput/<env>__<char>__<motion>/frame_00000.png ...
///
/// This does NOT call ffmpeg itself. After a run, encode each
/// "<env>__<char>__<motion>" frame folder into an MP4 (Claude/you can do this
/// with a one-line ffmpeg command per folder, then delete the frames).
///
/// Add more Environments / Characters / Motions to the three lists in the
/// window at any time -- no code changes needed to scale up.
/// </summary>
public class BatchDatasetGenerator : EditorWindow
{
    public List<GameObject> environments = new List<GameObject>();
    public List<GameObject> characters = new List<GameObject>();
    public List<AnimationClip> motions = new List<AnimationClip>();

    public int frameRate = 30;
    public int renderWidth = 1280;
    public int renderHeight = 720;

    // Camera rig: mounted ~2.5m up, close, angled down at the character.
    public float cameraHeight = 2.5f;
    public float cameraForwardOffset = 1.6f;
    public float cameraTargetHeight = 1.3f;

    public string outputFolderName = "BatchOutput";

    private SerializedObject so;
    private SerializedProperty envProp, charProp, motionProp;

    [MenuItem("GestureData/Batch Dataset Generator")]
    public static void ShowWindow()
    {
        GetWindow<BatchDatasetGenerator>("Batch Dataset Generator");
    }

    private void OnEnable()
    {
        so = new SerializedObject(this);
        envProp = so.FindProperty("environments");
        charProp = so.FindProperty("characters");
        motionProp = so.FindProperty("motions");
    }

    private void OnGUI()
    {
        so.Update();
        EditorGUILayout.LabelField("Environments (outer loop)", EditorStyles.boldLabel);
        EditorGUILayout.PropertyField(envProp, true);
        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Characters (middle loop)", EditorStyles.boldLabel);
        EditorGUILayout.PropertyField(charProp, true);
        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Motion clips (inner loop)", EditorStyles.boldLabel);
        EditorGUILayout.PropertyField(motionProp, true);
        so.ApplyModifiedProperties();

        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Render settings", EditorStyles.boldLabel);
        frameRate = EditorGUILayout.IntField("Frame Rate", frameRate);
        renderWidth = EditorGUILayout.IntField("Width", renderWidth);
        renderHeight = EditorGUILayout.IntField("Height", renderHeight);

        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Camera rig", EditorStyles.boldLabel);
        cameraHeight = EditorGUILayout.FloatField("Height (m)", cameraHeight);
        cameraForwardOffset = EditorGUILayout.FloatField("Distance from subject (m)", cameraForwardOffset);
        cameraTargetHeight = EditorGUILayout.FloatField("Look-at height (m)", cameraTargetHeight);

        EditorGUILayout.Space();
        outputFolderName = EditorGUILayout.TextField("Output folder", outputFolderName);

        int envCount = Mathf.Max(environments.Count, 0);
        int charCount = Mathf.Max(characters.Count, 0);
        int motionCount = Mathf.Max(motions.Count, 0);
        int total = envCount * charCount * motionCount;
        EditorGUILayout.HelpBox(
            $"{envCount} environments x {charCount} characters x {motionCount} motions = {total} clips to render.",
            MessageType.Info);

        using (new EditorGUI.DisabledScope(total == 0))
        {
            if (GUILayout.Button("Run Batch", GUILayout.Height(32)))
            {
                RunBatch();
            }
        }
    }

    private void RunBatch()
    {
        string projectRoot = Directory.GetParent(Application.dataPath).FullName;
        string outputRoot = Path.Combine(projectRoot, outputFolderName);
        Directory.CreateDirectory(outputRoot);

        string manifestPath = Path.Combine(outputRoot, "manifest.csv");
        bool writeHeader = !File.Exists(manifestPath);

        int total = environments.Count * characters.Count * motions.Count;
        int done = 0;

        GameObject camGO = new GameObject("__BatchRenderCamera");
        Camera cam = camGO.AddComponent<Camera>();
        RenderTexture rt = new RenderTexture(renderWidth, renderHeight, 24);
        cam.targetTexture = rt;
        Texture2D tex = new Texture2D(renderWidth, renderHeight, TextureFormat.RGB24, false);

        using (StreamWriter manifest = new StreamWriter(manifestPath, true))
        {
            if (writeHeader)
            {
                manifest.WriteLine("clip_id,environment,character,motion,frame_count,fps,width,height");
            }

            try
            {
                foreach (GameObject envPrefab in environments)
                {
                    GameObject envInstance = null;
                    if (envPrefab != null)
                    {
                        envInstance = (GameObject)PrefabUtility.InstantiatePrefab(envPrefab);
                    }

                    try
                    {
                        foreach (GameObject charPrefab in characters)
                        {
                            if (charPrefab == null) continue;
                            GameObject charInstance = (GameObject)PrefabUtility.InstantiatePrefab(charPrefab);

                            Transform spawn = envInstance != null ? envInstance.transform.Find("SpawnPoint") : null;
                            if (spawn != null)
                            {
                                charInstance.transform.position = spawn.position;
                                charInstance.transform.rotation = spawn.rotation;
                            }
                            else
                            {
                                charInstance.transform.position = Vector3.zero;
                                charInstance.transform.rotation = Quaternion.identity;
                            }

                            Animator animator = charInstance.GetComponent<Animator>();
                            if (animator == null)
                            {
                                Debug.LogError($"[BatchDatasetGenerator] '{charPrefab.name}' has no Animator component - skipped.");
                                UnityEngine.Object.DestroyImmediate(charInstance);
                                continue;
                            }

                            Vector3 targetPos = charInstance.transform.position + Vector3.up * cameraTargetHeight;
                            Vector3 camPos = charInstance.transform.position
                                + charInstance.transform.forward * cameraForwardOffset
                                + Vector3.up * cameraHeight;
                            camGO.transform.position = camPos;
                            camGO.transform.LookAt(targetPos);

                            foreach (AnimationClip clip in motions)
                            {
                                if (clip == null) continue;

                                string envName = envPrefab != null ? envPrefab.name : "NoEnv";
                                string clipId = $"{envName}__{charPrefab.name}__{clip.name}";
                                done++;

                                bool cancelled = EditorUtility.DisplayCancelableProgressBar(
                                    "Batch Dataset Generator",
                                    $"[{done}/{total}] {clipId}",
                                    (float)done / total);
                                if (cancelled)
                                {
                                    Debug.LogWarning("[BatchDatasetGenerator] Cancelled by user.");
                                    UnityEngine.Object.DestroyImmediate(charInstance);
                                    return;
                                }

                                int frameCount = RenderClipFrames(charInstance, clip, cam, rt, tex, outputRoot, clipId);
                                manifest.WriteLine($"{clipId},{envName},{charPrefab.name},{clip.name},{frameCount},{frameRate},{renderWidth},{renderHeight}");
                                manifest.Flush();
                            }

                            UnityEngine.Object.DestroyImmediate(charInstance);
                        }
                    }
                    finally
                    {
                        if (envInstance != null) UnityEngine.Object.DestroyImmediate(envInstance);
                    }
                }
            }
            finally
            {
                EditorUtility.ClearProgressBar();
                UnityEngine.Object.DestroyImmediate(tex);
                UnityEngine.Object.DestroyImmediate(camGO);
                rt.Release();
                Debug.Log($"[BatchDatasetGenerator] Done: rendered {done}/{total} clips into {outputRoot}. Manifest: {manifestPath}");
            }
        }
    }

    /// <summary>
    /// Samples the clip on the character using AnimationMode (the same mechanism
    /// the Editor's own Animation window uses for Humanoid preview -- it retargets
    /// correctly through the character's Avatar, no AnimatorController needed),
    /// rendering one PNG per frame into BatchOutput/<clipId>/frame_00000.png etc.
    /// Returns the number of frames written.
    /// </summary>
    private int RenderClipFrames(GameObject character, AnimationClip clip, Camera cam, RenderTexture rt, Texture2D tex, string outputRoot, string clipId)
    {
        string frameDir = Path.Combine(outputRoot, clipId);
        Directory.CreateDirectory(frameDir);

        int frameCount = Mathf.Max(1, Mathf.RoundToInt(clip.length * frameRate));

        AnimationMode.StartAnimationMode();
        try
        {
            for (int f = 0; f < frameCount; f++)
            {
                float t = frameCount > 1 ? (f / (float)frameRate) : 0f;
                AnimationMode.BeginSampling();
                AnimationMode.SampleAnimationClip(character, clip, t);
                AnimationMode.EndSampling();

                cam.Render();
                RenderTexture prevActive = RenderTexture.active;
                RenderTexture.active = rt;
                tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
                tex.Apply(false);
                RenderTexture.active = prevActive;

                byte[] png = tex.EncodeToPNG();
                File.WriteAllBytes(Path.Combine(frameDir, $"frame_{f:D5}.png"), png);
            }
        }
        finally
        {
            AnimationMode.StopAnimationMode();
        }

        return frameCount;
    }
}
