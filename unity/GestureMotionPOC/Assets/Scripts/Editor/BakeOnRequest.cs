using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

/// File-triggered lightmap bake, independent of the MCP link.
/// Write "bake" into <project>/Temp/bake_request.txt; progress is appended to Temp/bake_status.txt.
[InitializeOnLoad]
public static class BakeOnRequest
{
    static readonly string Req = Path.Combine(Directory.GetCurrentDirectory(), "Temp", "bake_request.txt");
    static readonly string Status = Path.Combine(Directory.GetCurrentDirectory(), "Temp", "bake_status.txt");
    static bool baking;
    static double started;

    static BakeOnRequest()
    {
        EditorApplication.update += Tick;
        Log("script loaded; project " + Application.dataPath);
    }

    static void Log(string s)
    {
        try { File.AppendAllText(Status, System.DateTime.Now.ToString("HH:mm:ss") + " " + s + "\n"); } catch { }
    }

    static void Tick()
    {
        if (baking)
        {
            if (!Lightmapping.isRunning)
            {
                baking = false;
                Log("bake finished after " + (EditorApplication.timeSinceStartup - started).ToString("F0") + " s; saving");
                foreach (var z in Object.FindObjectsByType<Zone>()) z.ApplyPreset(2);   // back to Daylight for editing
                EditorSceneManager.SaveOpenScenes();
                AssetDatabase.SaveAssets();
                Log("DONE");
            }
            return;
        }
        if (!File.Exists(Req)) return;
        string cmd = File.ReadAllText(Req).Trim();
        File.Delete(Req);
        if (cmd == "render") { RenderAll(); return; }
        if (cmd == "chars") { RenderChars(); return; }
        if (cmd == "normalize") { Normalize(); return; }
        if (cmd == "quality") { Quality.Apply(Log); return; }
        if (cmd == "captureall") { Log("starting captureall"); CaptureAll.Start(); return; }
        if (cmd == "playtest") { Log("starting playtest"); PlayTest.Start(); return; }
        if (cmd == "clipsheet") { Log(ClipViewerSetup.Sheet().Replace("\n", " || ")); Log("SHEET DONE"); return; }
        if (cmd == "viewer") { Log(ClipViewerSetup.Build().Replace("\n", " | ")); Log("VIEWER DONE"); return; }
        if (cmd != "bake") { Log("unknown request: " + cmd); return; }

        var scene = EditorSceneManager.GetActiveScene();
        if (scene.path != "Assets/Scenes/DatasetWorld.unity")
            scene = EditorSceneManager.OpenScene("Assets/Scenes/DatasetWorld.unity", OpenSceneMode.Single);
        var ls = ZoneBuilder.EnsureLightingSettings();
        Lightmapping.SetLightingSettingsForScene(scene, ls);
        // Bake a NEUTRAL, LOW baseline: the flat ambient colour is baked into every lightmapped surface, so a bright
        // ambient here would make Night/Evening impossible. Presets add their light at runtime on top of this.
        Lightmapping.Clear();
        foreach (var z in Object.FindObjectsByType<Zone>())
        {
            z.SetLightsActive(true);
            z.ApplyPreset(3);   // Overcast as a starting point ...
            foreach (var l in z.GetComponentsInChildren<Light>(true)) l.intensity *= 0.15f;   // ... then very dim: baked bounce must stay subtle
        }
        RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Flat;
        RenderSettings.ambientLight = new Color(0.03f, 0.03f, 0.035f);
        EditorSceneManager.SaveScene(scene);
        Log("starting bake in " + scene.path + " (ambient 0.03, lights at 15% of Overcast)");
        bool ok = Lightmapping.BakeAsync();
        Log("BakeAsync returned " + ok);
        baking = ok;
        started = EditorApplication.timeSinceStartup;
    }

    /// Dataset-camera renders for every zone under a few presets into EnvTests/baked/.
    static void RenderAll()
    {
        var scene = EditorSceneManager.GetActiveScene();
        if (scene.path != "Assets/Scenes/DatasetWorld.unity")
            scene = EditorSceneManager.OpenScene("Assets/Scenes/DatasetWorld.unity", OpenSceneMode.Single);
        string dir = Path.Combine(Directory.GetCurrentDirectory(), "EnvTests", "baked");
        Directory.CreateDirectory(dir);
        var xbot = (GameObject)PrefabUtility.InstantiatePrefab(AssetDatabase.LoadAssetAtPath<GameObject>("Assets/XBot.fbx"));
        var camGO = new GameObject("__TmpCam");
        var cam = camGO.AddComponent<Camera>();
        cam.clearFlags = CameraClearFlags.SolidColor; cam.backgroundColor = new Color(0.1f, 0.1f, 0.1f);
        cam.fieldOfView = RigSettings.FieldOfView; cam.nearClipPlane = RigSettings.NearClip;
        var zones = Object.FindObjectsByType<Zone>();
        foreach (var z in zones)
        {
            xbot.transform.SetPositionAndRotation(z.SpawnPoint.position, z.SpawnPoint.rotation);
            foreach (var o in zones) o.SetLightsActive(o == z);
            camGO.transform.SetPositionAndRotation(z.CameraAnchor.position, z.CameraAnchor.rotation);
            foreach (int i in new[] { 2, 4, 5 })
            {
                z.ApplyPreset(i);
                var rt = new RenderTexture(1920, 1080, 24) { antiAliasing = 4 };
                cam.targetTexture = rt; cam.Render();
                RenderTexture.active = rt;
                var tex = new Texture2D(1920, 1080, TextureFormat.RGB24, false);
                tex.ReadPixels(new Rect(0, 0, 1920, 1080), 0, 0); tex.Apply();
                cam.targetTexture = null; RenderTexture.active = null;
                File.WriteAllBytes(Path.Combine(dir, z.name.Replace("Zone_", "") + "_" + z.presets[i].name + ".png"), tex.EncodeToPNG());
                Object.DestroyImmediate(rt); Object.DestroyImmediate(tex);
            }
            z.ApplyPreset(2);
            Log("rendered " + z.name);
        }
        foreach (var o in zones) o.SetLightsActive(true);
        Object.DestroyImmediate(xbot); Object.DestroyImmediate(camGO);
        EditorSceneManager.SaveScene(scene);
        Log("RENDER DONE");
    }

    /// Every character model in every zone, dataset camera, Daylight -> EnvTests/characters/<Zone>_<Char>.png
    static void RenderChars()
    {
        var scene = EditorSceneManager.GetActiveScene();
        if (scene.path != "Assets/Scenes/DatasetWorld.unity")
            scene = EditorSceneManager.OpenScene("Assets/Scenes/DatasetWorld.unity", OpenSceneMode.Single);
        string dir = Path.Combine(Directory.GetCurrentDirectory(), "EnvTests", "characters2");
        Directory.CreateDirectory(dir);
        var camGO = new GameObject("__TmpCam");
        var cam = camGO.AddComponent<Camera>();
        cam.clearFlags = CameraClearFlags.SolidColor; cam.backgroundColor = new Color(0.1f, 0.1f, 0.1f);
        cam.fieldOfView = RigSettings.FieldOfView; cam.nearClipPlane = RigSettings.NearClip;
        string[] names = { "XBot", "Characters/James", "Characters/Kate", "Characters/Leonard", "Characters/Louise", "Characters/Megan", "Characters/PeasantGirl", "Characters/Remy", "Characters/Shannon", "Characters/TheBoss" };
        var zones = Object.FindObjectsByType<Zone>();
        foreach (string n in names)
        {
            string np = "Assets/Characters/Normalized/" + System.IO.Path.GetFileName(n) + ".prefab";
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(np) ?? AssetDatabase.LoadAssetAtPath<GameObject>("Assets/" + n + ".fbx");
            if (prefab == null) { Log("missing " + n); continue; }
            var g = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
            var rs = g.GetComponentsInChildren<Renderer>();
            var b = rs[0].bounds; foreach (var r in rs) b.Encapsulate(r.bounds);
            var anim = g.GetComponent<Animator>();
            Log(n + " height " + b.size.y.ToString("F2") + " width " + b.size.x.ToString("F2") + " animator " + (anim != null) + " humanoid " + (anim != null && anim.isHuman));
            foreach (var z in zones)
            {
                g.transform.SetPositionAndRotation(z.SpawnPoint.position, z.SpawnPoint.rotation);
                foreach (var o in zones) o.SetLightsActive(o == z);
                z.ApplyPreset(2);
                camGO.transform.SetPositionAndRotation(z.CameraAnchor.position, z.CameraAnchor.rotation);
                var rt = new RenderTexture(1920, 1080, 24) { antiAliasing = 4 };
                cam.targetTexture = rt; cam.Render();
                RenderTexture.active = rt;
                var tex = new Texture2D(1920, 1080, TextureFormat.RGB24, false);
                tex.ReadPixels(new Rect(0, 0, 1920, 1080), 0, 0); tex.Apply();
                cam.targetTexture = null; RenderTexture.active = null;
                File.WriteAllBytes(Path.Combine(dir, z.name.Replace("Zone_", "") + "_" + System.IO.Path.GetFileName(n) + ".png"), tex.EncodeToPNG());
                Object.DestroyImmediate(rt); Object.DestroyImmediate(tex);
            }
            Object.DestroyImmediate(g);
        }
        foreach (var o in zones) { o.SetLightsActive(true); o.ApplyPreset(2); }
        Object.DestroyImmediate(camGO);
        EditorSceneManager.SaveScene(scene);
        Log("CHARS DONE");
    }

    /// Scaled prefab copies (originals untouched) so characters have believable, varied heights.
    static void Normalize()
    {
        var targets = new System.Collections.Generic.Dictionary<string, float> {
            {"James",1.85f},{"Kate",1.68f},{"Leonard",1.80f},{"Louise",1.65f},{"Megan",1.70f},
            {"PeasantGirl",1.65f},{"Remy",1.80f},{"Shannon",1.75f},{"TheBoss",1.85f} };
        Directory.CreateDirectory("Assets/Characters/Normalized");
        foreach (var kv in targets)
        {
            var src = AssetDatabase.LoadAssetAtPath<GameObject>("Assets/Characters/" + kv.Key + ".fbx");
            var g = (GameObject)PrefabUtility.InstantiatePrefab(src);
            var rs = g.GetComponentsInChildren<Renderer>();
            var b = rs[0].bounds; foreach (var r in rs) b.Encapsulate(r.bounds);
            float f = kv.Value / b.size.y;
            g.transform.localScale = Vector3.one * f;
            g.transform.position = Vector3.zero;
            PrefabUtility.SaveAsPrefabAsset(g, "Assets/Characters/Normalized/" + kv.Key + ".prefab");
            Log(kv.Key + ": " + b.size.y.ToString("F2") + " -> " + kv.Value.ToString("F2") + " (scale x" + f.ToString("F3") + ")");
            Object.DestroyImmediate(g);
        }
        AssetDatabase.SaveAssets();
        Log("NORMALIZE DONE");
    }
}
