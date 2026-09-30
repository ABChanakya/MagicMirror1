using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Animations;
using UnityEngine.Playables;

/// Builds Assets/Scenes/ClipViewer.unity: a copy of DatasetWorld with a ClipViewer that plays every clip
/// under Assets/Motion/ on the normalized characters in the living room.  Menu: Tools > Build Clip Viewer Scene.
public static class ClipViewerSetup
{
    [MenuItem("Tools/Build Clip Viewer Scene")]
    public static void BuildMenu() { Debug.Log(Build()); }

    /// Edit-mode check of the viewer's playback path: PlayableGraph on Kate, wide + dataset camera frames for several clip types.
    public static string Sheet()
    {
        var sb = new StringBuilder();
        var viewer = Object.FindFirstObjectByType<ClipViewer>();
        var z = GameObject.Find("Zone_LivingRoom").GetComponent<Zone>();
        string dir = Path.Combine(Directory.GetCurrentDirectory(), "EnvTests", "clips");
        Directory.CreateDirectory(dir);
        var kate = (GameObject)PrefabUtility.InstantiatePrefab(AssetDatabase.LoadAssetAtPath<GameObject>("Assets/Characters/Normalized/Kate.prefab"));
        kate.transform.SetPositionAndRotation(z.SpawnPoint.position, z.SpawnPoint.rotation);
        var anim = kate.GetComponent<Animator>(); anim.applyRootMotion = false;
        var camGO = new GameObject("__TmpCam"); var cam = camGO.AddComponent<Camera>();
        cam.clearFlags = CameraClearFlags.SolidColor; cam.backgroundColor = new Color(0.1f, 0.1f, 0.1f); cam.nearClipPlane = 0.05f;
        string[] want = { "kimodo_swipe_right_9", "kimodo_swipe_right_92", "kimodo_swipe_left_11", "kimodo_swipe_left_13", "kimodo_null_1", "constraint_target_swipe_right_340_gen", "test_swipe_right", "real472_pure_ik" };
        var hr = anim.GetBoneTransform(HumanBodyBones.RightHand); var hl = anim.GetBoneTransform(HumanBodyBones.LeftHand); var hips = anim.GetBoneTransform(HumanBodyBones.Hips);
        foreach (string w in want)
        {
            int idx = viewer.labels.FindIndex(l => l == w || l.StartsWith(w + "."));
            if (idx < 0) idx = viewer.labels.FindIndex(l => l.StartsWith(w));
            if (idx < 0) { sb.AppendLine("no clip " + w); continue; }
            var clip = viewer.clips[idx];
            var graph = UnityEngine.Playables.PlayableGraph.Create("test");
            graph.SetTimeUpdateMode(UnityEngine.Playables.DirectorUpdateMode.Manual);
            var output = UnityEngine.Animations.AnimationPlayableOutput.Create(graph, "o", anim);
            var pl = UnityEngine.Animations.AnimationClipPlayable.Create(graph, clip);
            output.SetSourcePlayable(pl);
            float maxDx = 0, y0 = 0, y1 = -9, y2 = 9; Vector3 hipStart = Vector3.zero, hipEnd = Vector3.zero;
            for (int k = 0; k <= 4; k++)
            {
                double t = clip.length * k / 4.0 * 0.98;
                pl.SetTime(t); graph.Evaluate(0f);
                Vector3 rp = hr.position - z.SpawnPoint.position, lp = hl.position - z.SpawnPoint.position;
                if (k == 0) { hipStart = hips.position; }
                if (k == 4) { hipEnd = hips.position; }
                sb.Append(w.Replace("kimodo_", "").Replace("constraint_target_", "ct_") + " t" + k + " R(" + rp.x.ToString("F2") + "," + rp.y.ToString("F2") + ") L(" + lp.x.ToString("F2") + "," + lp.y.ToString("F2") + ")  ");
                // wide camera
                cam.fieldOfView = 55f;
                camGO.transform.position = z.SpawnPoint.position + z.SpawnPoint.forward * 3.0f + Vector3.up * 1.4f;
                camGO.transform.LookAt(z.SpawnPoint.position + Vector3.up * 1.0f);
                if (k == 0 || k == 2 || k == 4) Snap(cam, Path.Combine(dir, w + "_wide_" + k + ".png"));
            }
            sb.AppendLine("| hips drift " + (hipEnd - hipStart).magnitude.ToString("F2") + " m");
            graph.Destroy();
        }
        Object.DestroyImmediate(kate); Object.DestroyImmediate(camGO);
        return sb.ToString();
    }

    static void Snap(Camera cam, string path)
    {
        var rt = new RenderTexture(960, 540, 24);
        cam.targetTexture = rt; cam.Render();
        RenderTexture.active = rt;
        var tex = new Texture2D(960, 540, TextureFormat.RGB24, false);
        tex.ReadPixels(new Rect(0, 0, 960, 540), 0, 0); tex.Apply();
        cam.targetTexture = null; RenderTexture.active = null;
        File.WriteAllBytes(path, tex.EncodeToPNG());
        Object.DestroyImmediate(rt); Object.DestroyImmediate(tex);
    }

    public static string Build()
    {
        var sb = new StringBuilder();
        AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
        var paths = new List<string>();
        if (File.Exists("Assets/Motion/real472_pure_ik.fbx")) paths.Add("Assets/Motion/real472_pure_ik.fbx");
        paths.AddRange(Directory.GetFiles("Assets/Motion/Clips", "*.fbx").Select(p => p.Replace('\\', '/')).OrderBy(p => p));
        var clips = new List<AnimationClip>(); var labels = new List<string>();
        int notHuman = 0, noClip = 0;
        foreach (string p in paths)
        {
            var imp = AssetImporter.GetAtPath(p) as ModelImporter;
            if (imp == null || imp.animationType != ModelImporterAnimationType.Human)
            {
                notHuman++;
                if (imp != null) { imp.animationType = ModelImporterAnimationType.Human; imp.avatarSetup = ModelImporterAvatarSetup.CreateFromThisModel; imp.SaveAndReimport(); }
            }
            AnimationClip clip = null;
            foreach (var o in AssetDatabase.LoadAllAssetsAtPath(p)) if (o is AnimationClip c && !c.name.StartsWith("__preview")) clip = c;
            if (clip == null) { noClip++; continue; }
            clips.Add(clip); labels.Add(Path.GetFileNameWithoutExtension(p));
        }
        sb.AppendLine("clips: " + clips.Count + " of " + paths.Count + " (re-imported as Human: " + notHuman + ", no clip: " + noClip + ")");
        var dur = clips.Select(c => c.length).ToList();
        sb.AppendLine("clip length min/max: " + dur.Min().ToString("F2") + " / " + dur.Max().ToString("F2") + " s; humanMotion all: " + clips.All(c => c.humanMotion));

        var ds = EditorSceneManager.GetActiveScene();
        if (ds.path != "Assets/Scenes/DatasetWorld.unity") ds = EditorSceneManager.OpenScene("Assets/Scenes/DatasetWorld.unity", OpenSceneMode.Single);
        EditorSceneManager.SaveScene(ds);
        EditorSceneManager.SaveScene(ds, "Assets/Scenes/ClipViewer.unity", true);
        var scene = EditorSceneManager.OpenScene("Assets/Scenes/ClipViewer.unity", OpenSceneMode.Single);
        var old = GameObject.Find("ClipViewer"); if (old != null) Object.DestroyImmediate(old);
        var z = GameObject.Find("Zone_LivingRoom").GetComponent<Zone>();
        foreach (var o in Object.FindObjectsByType<Zone>()) o.SetLightsActive(true);
        z.ApplyPreset(2);

        foreach (var n in new[] { "DatasetCamera", "WideCamera" }) { var g = GameObject.Find(n); if (g != null) Object.DestroyImmediate(g); }
        var cam1 = new GameObject("DatasetCamera"); var c1 = cam1.AddComponent<Camera>();
        c1.fieldOfView = RigSettings.FieldOfView; c1.nearClipPlane = RigSettings.NearClip; c1.clearFlags = CameraClearFlags.SolidColor; c1.backgroundColor = new Color(0.1f, 0.1f, 0.1f);
        cam1.transform.SetPositionAndRotation(z.CameraAnchor.position, z.CameraAnchor.rotation); cam1.tag = "MainCamera";
        var cam2 = new GameObject("WideCamera"); var c2 = cam2.AddComponent<Camera>();
        c2.fieldOfView = 55f; c2.nearClipPlane = 0.05f; c2.clearFlags = CameraClearFlags.SolidColor; c2.backgroundColor = new Color(0.1f, 0.1f, 0.1f); c2.enabled = false;

        var go = new GameObject("ClipViewer");
        var v = go.AddComponent<ClipViewer>();
        v.clips = clips; v.labels = labels; v.spawnPoint = z.SpawnPoint; v.datasetCamera = c1; v.wideCamera = c2;
        foreach (string n in new[] { "Kate", "James", "Leonard", "Louise", "Megan", "Remy", "Shannon", "PeasantGirl", "TheBoss" })
        {
            var p = AssetDatabase.LoadAssetAtPath<GameObject>("Assets/Characters/Normalized/" + n + ".prefab");
            if (p != null) v.characters.Add(p);
        }
        var xb = AssetDatabase.LoadAssetAtPath<GameObject>("Assets/XBot.fbx"); if (xb != null) v.characters.Add(xb);
        sb.AppendLine("characters: " + v.characters.Count);
        EditorUtility.SetDirty(go);
        EditorSceneManager.SaveScene(scene);
        sb.AppendLine("scene saved: Assets/Scenes/ClipViewer.unity (active). Zones in scene: " + Object.FindObjectsByType<Zone>().Length);
        return sb.ToString();
    }
}
