using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;

/// Play-mode capture of every clip: 8 evenly spaced frames per clip from the wide (full-body) camera,
/// plus hand/foot/hip positions and dataset-camera viewport coordinates -> EnvTests/capture/{frames,metrics.csv}.
/// Trigger: "captureall" in Temp/bake_request.txt. Progress in Temp/bake_status.txt. State in SessionState (scripts reload on Play).
[InitializeOnLoad]
public static class CaptureAll
{
    const int Frames = 8;
    const int W = 900, H = 1200;
    static CaptureAll() { EditorApplication.update += Tick; }

    static string Dir(string sub) { var d = Path.Combine(Directory.GetCurrentDirectory(), "EnvTests", "capture", sub); Directory.CreateDirectory(d); return d; }
    static void Log(string s) { try { File.AppendAllText(Path.Combine(Directory.GetCurrentDirectory(), "Temp", "bake_status.txt"), System.DateTime.Now.ToString("HH:mm:ss") + " [capture] " + s + "\n"); } catch { } }

    public static void Start()
    {
        string csv = Path.Combine(Dir(""), "metrics.csv");
        File.WriteAllText(csv, "clip,k,t,rh_x,rh_y,rh_z,lh_x,lh_y,lh_z,hips_x,hips_y,hips_z,head_y,lf_x,lf_z,rf_x,rf_z,rh_vx,rh_vy,lh_vx,lh_vy\n");
        SessionState.SetInt("ca_stage", 1); SessionState.SetInt("ca_clip", 0); SessionState.SetInt("ca_k", 0);
        EditorApplication.EnterPlaymode();
    }

    static void Tick()
    {
        int stage = SessionState.GetInt("ca_stage", 0);
        if (stage == 0 || !EditorApplication.isPlaying) return;
        Application.runInBackground = true;
        EditorApplication.QueuePlayerLoopUpdate();
        var viewer = Object.FindFirstObjectByType<ClipViewer>();
        if (viewer == null) return;
        int clip = SessionState.GetInt("ca_clip", 0), k = SessionState.GetInt("ca_k", 0);

        if (stage == 1) { viewer.SetWideCamera(true); SessionState.SetInt("ca_stage", 2); Log("running, clips=" + viewer.clips.Count); return; }

        if (stage == 3)   // a frame was set up on the previous tick: capture it now
        {
            Capture(viewer, clip, k);
            k++;
            if (k >= Frames) { k = 0; clip++; if (clip % 25 == 0) Log("clip " + clip + "/" + viewer.clips.Count); }
            SessionState.SetInt("ca_clip", clip); SessionState.SetInt("ca_k", k);
            if (clip >= viewer.clips.Count) { SessionState.SetInt("ca_stage", 0); Log("CAPTURE DONE"); EditorApplication.ExitPlaymode(); return; }
        }

        // set up (clip, k)
        if (k == 0) viewer.Select(clip);
        float len = viewer.clips[clip].length;
        viewer.Freeze(len * 0.985f * k / (Frames - 1));
        SessionState.SetInt("ca_stage", 3);
    }

    static void Capture(ClipViewer v, int clip, int k)
    {
        var wide = v.wideCamera; var data = v.datasetCamera;
        string name = v.labels[clip];
        var rt = new RenderTexture(W, H, 24) { antiAliasing = 4 };
        wide.targetTexture = rt; wide.enabled = true; wide.Render(); wide.enabled = false;
        RenderTexture.active = rt;
        var tex = new Texture2D(W, H, TextureFormat.RGB24, false);
        tex.ReadPixels(new Rect(0, 0, W, H), 0, 0); tex.Apply();
        wide.targetTexture = null; RenderTexture.active = null;
        File.WriteAllBytes(Path.Combine(Dir("frames"), name + "_" + k + ".png"), tex.EncodeToPNG());
        Object.Destroy(rt); Object.Destroy(tex);

        var a = v.CurrentAnimator; var sp = v.spawnPoint.position;
        Vector3 rh = a.GetBoneTransform(HumanBodyBones.RightHand).position - sp, lh = a.GetBoneTransform(HumanBodyBones.LeftHand).position - sp;
        Vector3 hp = a.GetBoneTransform(HumanBodyBones.Hips).position - sp, hd = a.GetBoneTransform(HumanBodyBones.Head).position - sp;
        Vector3 lf = a.GetBoneTransform(HumanBodyBones.LeftFoot).position - sp, rf = a.GetBoneTransform(HumanBodyBones.RightFoot).position - sp;
        Vector3 rv = data.WorldToViewportPoint(a.GetBoneTransform(HumanBodyBones.RightHand).position), lv = data.WorldToViewportPoint(a.GetBoneTransform(HumanBodyBones.LeftHand).position);
        var ci = System.Globalization.CultureInfo.InvariantCulture;
        string F(float f) { return f.ToString("F3", ci); }
        var sb = new StringBuilder();
        sb.Append(name).Append(',').Append(k).Append(',').Append(F(v.clips[clip].length * 0.985f * k / (Frames - 1)));
        foreach (var f in new[] { rh.x, rh.y, rh.z, lh.x, lh.y, lh.z, hp.x, hp.y, hp.z, hd.y, lf.x, lf.z, rf.x, rf.z, rv.x, rv.y, lv.x, lv.y }) sb.Append(',').Append(F(f));
        sb.Append('\n');
        File.AppendAllText(Path.Combine(Dir(""), "metrics.csv"), sb.ToString());
    }
}
