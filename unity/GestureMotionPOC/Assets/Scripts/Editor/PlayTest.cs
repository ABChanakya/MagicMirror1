using System.IO;
using UnityEditor;
using UnityEngine;

/// Enters Play mode, lets ClipViewer run for real, and saves frames of several clips.
/// Trigger: write "playtest" to Temp/bake_request.txt (handled in BakeOnRequest). State lives in SessionState because entering Play mode reloads scripts.
[InitializeOnLoad]
public static class PlayTest
{
    static readonly string[] Wanted = { "kimodo_swipe_right_92", "kimodo_swipe_left_13", "kimodo_null_1", "real472_pure_ik" };

    static PlayTest() { EditorApplication.update += Tick; }

    public static void Start()
    {
        SessionState.SetInt("pt_stage", 1);
        SessionState.SetInt("pt_i", 0);
        EditorApplication.EnterPlaymode();
    }

    static void Log(string s)
    {
        try { File.AppendAllText(Path.Combine(Directory.GetCurrentDirectory(), "Temp", "bake_status.txt"), System.DateTime.Now.ToString("HH:mm:ss") + " [playtest] " + s + "\n"); } catch { }
    }

    static void Tick()
    {
        int stage = SessionState.GetInt("pt_stage", 0);
        if (stage == 0 || !EditorApplication.isPlaying) return;
        // The editor window is in the background here, so Unity would not tick the player loop by itself: run it manually.
        Application.runInBackground = true;
        EditorApplication.QueuePlayerLoopUpdate();
        var viewer = Object.FindFirstObjectByType<ClipViewer>();
        if (viewer == null) { return; }
        float now = Time.realtimeSinceStartup;
        if (stage == 1) { SessionState.SetFloat("pt_t", now); SessionState.SetInt("pt_stage", 2); Log("play mode running, viewer found, clips=" + viewer.clips.Count); return; }
        float t0 = SessionState.GetFloat("pt_t", now);
        int i = SessionState.GetInt("pt_i", 0);
        // per clip: 3 wide frames (0.30 s, 0.60 s, 0.90 s after the clip starts) + 1 dataset-camera frame (0.60 s)
        if (i >= Wanted.Length * 4) { SessionState.SetInt("pt_stage", 0); Log("PLAYTEST DONE"); EditorApplication.ExitPlaymode(); return; }
        int clipNo = i / 4, k = i % 4; bool wide = k < 3;
        float[] offs = { 0.30f, 0.60f, 0.90f, 0.60f };
        if (stage == 2)
        {
            if (k == 0)
            {
                int idx = viewer.labels.FindIndex(l => l == Wanted[clipNo]);
                viewer.Select(idx);
                SessionState.SetFloat("pt_tsel", now);
            }
            viewer.SetWideCamera(wide);
            if (k == 3) { viewer.Select(viewer.labels.FindIndex(l => l == Wanted[clipNo])); SessionState.SetFloat("pt_tsel", now); }
            SessionState.SetInt("pt_stage", 3); return;
        }
        if (stage == 3 && now - SessionState.GetFloat("pt_tsel", now) > offs[k])
        {
            var cam = wide ? GameObject.Find("WideCamera").GetComponent<Camera>() : GameObject.Find("DatasetCamera").GetComponent<Camera>();
            var rt = new RenderTexture(960, 540, 24) { antiAliasing = 4 };
            cam.targetTexture = rt; cam.enabled = true; cam.Render(); cam.enabled = !wide == false ? cam.enabled : cam.enabled;
            RenderTexture.active = rt;
            var tex = new Texture2D(960, 540, TextureFormat.RGB24, false);
            tex.ReadPixels(new Rect(0, 0, 960, 540), 0, 0); tex.Apply();
            cam.targetTexture = null; RenderTexture.active = null;
            string dir = Path.Combine(Directory.GetCurrentDirectory(), "EnvTests", "playtest"); Directory.CreateDirectory(dir);
            File.WriteAllBytes(Path.Combine(dir, Wanted[clipNo] + (wide ? "_wide" + k : "_dataset") + ".png"), tex.EncodeToPNG());
            Object.Destroy(rt); Object.Destroy(tex);
            Log("captured " + Wanted[clipNo] + (wide ? " wide" + k : " dataset") + " :: " + viewer.Diagnostics());
            SessionState.SetInt("pt_i", i + 1); SessionState.SetInt("pt_stage", 2);
        }
    }
}
