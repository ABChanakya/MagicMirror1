using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Animations;
using UnityEngine.Playables;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

/// Press Play, then browse motion clips on a Humanoid character in a room.
///  Left/Right (or the buttons): previous / next clip     Space: pause     L: loop     Up/Down: speed
///  C: switch dataset camera <-> wide view                 N: next character
/// Root motion is off, so the character stays on its spawn point.
public class ClipViewer : MonoBehaviour
{
    public List<AnimationClip> clips = new List<AnimationClip>();
    public List<string> labels = new List<string>();
    public List<GameObject> characters = new List<GameObject>();
    public Transform spawnPoint;
    public Camera datasetCamera;
    public Camera wideCamera;

    int clipIndex, charIndex;
    bool paused, loop = true, useDatasetCam = true;
    float speed = 1f;
    GameObject current;
    PlayableGraph graph;
    AnimationClipPlayable playable;

    void Start()
    {
        SpawnCharacter();
        ShowCamera();
    }

    void SpawnCharacter()
    {
        if (graph.IsValid()) graph.Destroy();
        if (current != null) Destroy(current);
        if (characters.Count == 0) return;
        current = Instantiate(characters[charIndex % characters.Count], spawnPoint.position, spawnPoint.rotation);
        var anim = current.GetComponent<Animator>();
        anim.applyRootMotion = false;
        anim.runtimeAnimatorController = null;
        graph = PlayableGraph.Create("ClipViewer");
        graph.SetTimeUpdateMode(DirectorUpdateMode.GameTime);
        var output = AnimationPlayableOutput.Create(graph, "out", anim);
        PlayClip(clipIndex, output);
        graph.Play();
    }

    void PlayClip(int index, AnimationPlayableOutput output)
    {
        if (clips.Count == 0) return;
        clipIndex = (index % clips.Count + clips.Count) % clips.Count;
        if (playable.IsValid()) playable.Destroy();
        playable = AnimationClipPlayable.Create(graph, clips[clipIndex]);
        playable.SetApplyFootIK(false);
        output.SetSourcePlayable(playable);
        playable.SetTime(0);
        playable.SetSpeed(paused ? 0 : speed);
        playable.Play();   // playables created after graph.Play() start out paused
    }

    public int CurrentIndex => clipIndex;
    public Animator CurrentAnimator => current != null ? current.GetComponent<Animator>() : null;
    /// Pause on a specific moment of the current clip (used by automated capture).
    public void Freeze(double t)
    {
        paused = true;
        if (playable.IsValid()) { playable.SetSpeed(0); playable.SetTime(t); }
    }
    public string Diagnostics()
    {
        return "playhead=" + (playable.IsValid() ? playable.GetTime().ToString("F3") : "n/a") +
               " speed=" + (playable.IsValid() ? playable.GetSpeed().ToString("F2") : "n/a") +
               " playState=" + (playable.IsValid() ? playable.GetPlayState().ToString() : "n/a") +
               " graphPlaying=" + (graph.IsValid() && graph.IsPlaying()) +
               " Time.time=" + Time.time.ToString("F2") + " dt=" + Time.deltaTime.ToString("F4") + " timeScale=" + Time.timeScale +
               " focused=" + Application.isFocused + " frame=" + Time.frameCount;
    }
    public void SetWideCamera(bool wide) { useDatasetCam = !wide; ShowCamera(); }

    public void Select(int index)
    {
        if (!graph.IsValid()) return;
        PlayClip(index, (AnimationPlayableOutput)graph.GetOutput(0));
    }

    void ShowCamera()
    {
        if (datasetCamera != null) datasetCamera.enabled = useDatasetCam;
        if (wideCamera != null)
        {
            wideCamera.enabled = !useDatasetCam;
            if (current != null)
            {
                // review camera: 2.5 m up (same height as the dataset camera), close enough that the person fills the frame, looking down at the body
                Vector3 origin = spawnPoint.position + Vector3.up * 2.5f;
                float dist = 2.3f;
                if (Physics.Raycast(spawnPoint.position + Vector3.up * 1.5f, spawnPoint.forward, out RaycastHit hit, 2.3f, ~0, QueryTriggerInteraction.Ignore))
                    dist = Mathf.Max(1.4f, hit.distance - 0.25f);
                wideCamera.fieldOfView = 48f;
                wideCamera.transform.position = origin + spawnPoint.forward * dist;
                wideCamera.transform.LookAt(spawnPoint.position + Vector3.up * 0.95f);
            }
        }
    }

    void Update()
    {
        if (graph.IsValid() && playable.IsValid())
        {
            double len = clips.Count > 0 ? clips[clipIndex].length : 1;
            if (playable.GetTime() >= len)
            {
                if (loop) playable.SetTime(0);
                else { playable.SetTime(len); }
            }
        }
        if (Pressed(Key.Right)) Select(clipIndex + 1);
        if (Pressed(Key.Left)) Select(clipIndex - 1);
        if (Pressed(Key.Space)) TogglePause();
        if (Pressed(Key.L)) loop = !loop;
        if (Pressed(Key.Up)) SetSpeed(speed * 1.5f);
        if (Pressed(Key.Down)) SetSpeed(speed / 1.5f);
        if (Pressed(Key.C)) { useDatasetCam = !useDatasetCam; ShowCamera(); }
        if (Pressed(Key.N)) { charIndex++; SpawnCharacter(); }
    }

    void TogglePause() { paused = !paused; if (playable.IsValid()) playable.SetSpeed(paused ? 0 : speed); }
    void SetSpeed(float s) { speed = Mathf.Clamp(s, 0.1f, 4f); if (playable.IsValid() && !paused) playable.SetSpeed(speed); }

    // Works with the new Input System, the old one, or both.
    enum Key { Left, Right, Up, Down, Space, L, C, N }
    static bool Pressed(Key k)
    {
#if ENABLE_INPUT_SYSTEM
        var kb = Keyboard.current;
        if (kb == null) return false;
        switch (k)
        {
            case Key.Left: return kb.leftArrowKey.wasPressedThisFrame;
            case Key.Right: return kb.rightArrowKey.wasPressedThisFrame;
            case Key.Up: return kb.upArrowKey.wasPressedThisFrame;
            case Key.Down: return kb.downArrowKey.wasPressedThisFrame;
            case Key.Space: return kb.spaceKey.wasPressedThisFrame;
            case Key.L: return kb.lKey.wasPressedThisFrame;
            case Key.C: return kb.cKey.wasPressedThisFrame;
            case Key.N: return kb.nKey.wasPressedThisFrame;
        }
        return false;
#elif ENABLE_LEGACY_INPUT_MANAGER
        switch (k)
        {
            case Key.Left: return Input.GetKeyDown(KeyCode.LeftArrow);
            case Key.Right: return Input.GetKeyDown(KeyCode.RightArrow);
            case Key.Up: return Input.GetKeyDown(KeyCode.UpArrow);
            case Key.Down: return Input.GetKeyDown(KeyCode.DownArrow);
            case Key.Space: return Input.GetKeyDown(KeyCode.Space);
            case Key.L: return Input.GetKeyDown(KeyCode.L);
            case Key.C: return Input.GetKeyDown(KeyCode.C);
            case Key.N: return Input.GetKeyDown(KeyCode.N);
        }
        return false;
#else
        return false;
#endif
    }

    void OnGUI()
    {
        var box = new GUIStyle(GUI.skin.box) { alignment = TextAnchor.UpperLeft, fontSize = 16 };
        string name = clips.Count > 0 ? labels[clipIndex] : "(no clips)";
        double t = playable.IsValid() ? playable.GetTime() : 0;
        float len = clips.Count > 0 ? clips[clipIndex].length : 0;
        GUI.Box(new Rect(10, 10, 620, 118),
            $"Clip {clipIndex + 1}/{clips.Count}:  {name}\n" +
            $"{t:F2} s / {len:F2} s    speed x{speed:F2}    {(paused ? "PAUSED" : "playing")}    loop {(loop ? "on" : "off")}\n" +
            $"Character: {(current != null ? current.name.Replace("(Clone)", "") : "-")}    Camera: {(useDatasetCam ? "dataset" : "wide")}\n" +
            "Left/Right clip   Space pause   L loop   Up/Down speed   C camera   N character", box);
        float y = 138;
        if (GUI.Button(new Rect(10, y, 90, 32), "< Prev")) Select(clipIndex - 1);
        if (GUI.Button(new Rect(105, y, 90, 32), "Next >")) Select(clipIndex + 1);
        if (GUI.Button(new Rect(200, y, 90, 32), paused ? "Play" : "Pause")) TogglePause();
        if (GUI.Button(new Rect(295, y, 90, 32), "Camera")) { useDatasetCam = !useDatasetCam; ShowCamera(); }
        if (GUI.Button(new Rect(390, y, 110, 32), "Character")) { charIndex++; SpawnCharacter(); }
        if (GUI.Button(new Rect(505, y, 90, 32), "Random")) Select(Random.Range(0, Mathf.Max(1, clips.Count)));
    }

    void OnDestroy() { if (graph.IsValid()) graph.Destroy(); }
}
