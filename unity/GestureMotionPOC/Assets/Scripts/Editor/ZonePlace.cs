using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

/// Placement + audit helpers for building a Zone's Props. All coordinates are ZONE-LOCAL metres
/// (origin = floor centre, +Z = camera side, +X = image-left, -X = image-right).
/// Lessons baked in: prefab root rotation/scale is preserved (Poly Haven FBX carry 270deg/100x),
/// seating faces are derived from the mesh (never hard-coded), items rest on whatever is below them.
public static class ZonePlace
{
    public const string PH = "Assets/Environments/PolyHaven/";
    public const string MP = "Assets/Furniture Mega Pack/Prefabs/";
    public const string OF = "Assets/Office Room Furniture/Office/Prefabs/";
    public const string NP = "Assets/nappin/HouseInteriorPack/Prefabs/";
    public static Zone K;
    public static StringBuilder Log = new StringBuilder();

    public static string Poly(string id) { return PH + id + "/" + id + "_1k.fbx"; }

    public static Bounds B(GameObject g)
    {
        var rs = g.GetComponentsInChildren<Renderer>();
        Bounds b = rs[0].bounds;
        foreach (var r in rs) b.Encapsulate(r.bounds);
        return b;
    }

    static void Colliders(GameObject g)
    {
        foreach (var c in g.GetComponentsInChildren<BoxCollider>()) Object.DestroyImmediate(c);
        foreach (var mf in g.GetComponentsInChildren<MeshFilter>()) mf.gameObject.AddComponent<BoxCollider>();
    }

    public static float SurfaceY(float x, float z, GameObject ignore = null)
    {
        Physics.SyncTransforms();
        // start just BELOW the ceiling slab, otherwise the ray lands on the roof of a low room
        Vector3 o = K.transform.TransformPoint(new Vector3(x, K.ceilingHeight - 0.05f, z));
        RaycastHit best = default; bool any = false;
        foreach (var h in Physics.RaycastAll(o, Vector3.down, 10f, ~0, QueryTriggerInteraction.Ignore))
        {
            if (ignore != null && h.collider.transform.IsChildOf(ignore.transform)) continue;
            if (!any || h.distance < best.distance) { best = h; any = true; }
        }
        return any ? best.point.y - K.transform.position.y : 0f;
    }

    /// Instantiate keeping the prefab's root rotation. yaw = extra rotation about zone-up. height>0 scales uniformly to that height
    /// (measured after rotation); width>0 scales so the longest horizontal side equals it; scale>0 sets a fixed factor.
    public static GameObject Make(string path, float yaw, float height = 0f, float longest = 0f, float scale = 0f)
    {
        var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(path);
        if (prefab == null) { Log.AppendLine("MISSING " + path); return null; }
        var g = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
        g.transform.SetParent(K.transform.Find("Props"), false);
        g.transform.rotation = K.transform.rotation * Quaternion.Euler(0, yaw, 0) * prefab.transform.localRotation;
        if (scale > 0) g.transform.localScale = prefab.transform.localScale * scale;
        else if (height > 0) g.transform.localScale = g.transform.localScale * (height / B(g).size.y);
        else if (longest > 0) { var b = B(g); g.transform.localScale = g.transform.localScale * (longest / Mathf.Max(b.size.x, b.size.z)); }
        return g;
    }

    static void MoveTo(GameObject g, float x, float z, float bottomY)
    {
        Bounds b = B(g);
        Vector3 t = K.transform.TransformPoint(new Vector3(x, 0, z));
        g.transform.position += new Vector3(t.x - b.center.x, K.transform.position.y + bottomY - b.min.y, t.z - b.center.z);
    }

    /// Place so the bounds centre is at (x,z) and the bottom rests on the surface below (or on y if given).
    public static GameObject Put(string path, float x, float z, float yaw, float height = 0f, float longest = 0f, float scale = 0f, float y = float.NaN)
    {
        var g = Make(path, yaw, height, longest, scale);
        if (g == null) return null;
        MoveTo(g, x, z, float.IsNaN(y) ? SurfaceY(x, z, g) : y);
        Colliders(g);
        Physics.SyncTransforms();
        return g;
    }

    /// Wall-hung item centred at height yc.
    public static GameObject PutWall(string path, float x, float z, float yaw, float yc, float height = 0f, float longest = 0f, float scale = 0f)
    {
        var g = Make(path, yaw, height, longest, scale);
        if (g == null) return null;
        Bounds b = B(g);
        g.transform.position += K.transform.TransformPoint(new Vector3(x, yc, z)) - b.center;
        Colliders(g);
        return g;
    }

    /// Item on a board of `shelf` at absolute height `level`, `offset` metres along the shelf's long axis.
    public static GameObject OnShelf(GameObject shelf, string path, float level, float offset, float extraYaw = 0f, float height = 0f, float scale = 0f)
    {
        Bounds sb = B(shelf);
        Vector3 axis = sb.size.x >= sb.size.z ? Vector3.right : Vector3.forward;
        Vector3 c = new Vector3(sb.center.x, 0, sb.center.z) + axis * offset;
        var g = Make(path, extraYaw, height, 0f, scale);
        if (g == null) return null;
        Bounds b = B(g);
        g.transform.position += new Vector3(c.x - b.center.x, K.transform.position.y + level - b.min.y, c.z - b.center.z);
        Colliders(g);
        return g;
    }

    /// World direction the piece faces, from the mesh: the backrest/tall side is the "back"; front is opposite.
    public static Vector3 Front(GameObject g)
    {
        Bounds b = B(g);
        float yCut = b.min.y + 0.65f * b.size.y;
        Vector3 sum = Vector3.zero; int n = 0;
        foreach (var mf in g.GetComponentsInChildren<MeshFilter>())
        {
            var v = mf.sharedMesh.vertices; var m = mf.transform.localToWorldMatrix;
            for (int i = 0; i < v.Length; i += 2)
            {
                Vector3 w = m.MultiplyPoint3x4(v[i]);
                if (w.y > yCut) { sum += new Vector3(w.x, 0, w.z); n++; }
            }
        }
        if (n == 0) return Vector3.forward;
        Vector3 back = sum / n - new Vector3(b.center.x, 0, b.center.z); back.y = 0;
        return (-back).normalized;
    }

    /// Rotate a seat about its centre so it faces a zone-local point.
    public static void FaceTo(GameObject g, float tx, float tz)
    {
        Bounds b = B(g);
        Vector3 want = K.transform.TransformPoint(new Vector3(tx, 0, tz)) - b.center; want.y = 0;
        float d = Vector3.SignedAngle(Front(g), want.normalized, Vector3.up);
        g.transform.RotateAround(b.center, Vector3.up, d);
        Colliders(g);
        Physics.SyncTransforms();
    }

    public static void LampLight(GameObject lamp, string name, float intensity, float range, Color? col = null)
    {
        if (lamp == null) return;
        Bounds lb = B(lamp);
        var lg = new GameObject(name + "@" + intensity.ToString(System.Globalization.CultureInfo.InvariantCulture));
        lg.transform.SetParent(lamp.transform, true);
        lg.transform.position = new Vector3(lb.center.x, lb.max.y - 0.12f, lb.center.z);
        var li = lg.AddComponent<Light>(); li.type = LightType.Point; li.range = range; li.intensity = intensity;
        li.color = col ?? new Color(1f, 0.78f, 0.5f); li.shadows = LightShadows.None; li.lightmapBakeType = LightmapBakeType.Mixed;
    }

    public static Material Plain(string name, Color c, float smooth)
    {
        string p = "Assets/Environments/Materials/" + name + ".mat";
        var m = AssetDatabase.LoadAssetAtPath<Material>(p);
        if (m == null) { m = new Material(Shader.Find("Universal Render Pipeline/Lit")); AssetDatabase.CreateAsset(m, p); }
        m.SetColor("_BaseColor", c); m.SetFloat("_Smoothness", smooth);
        EditorUtility.SetDirty(m);
        return m;
    }

    /// Textured box under Props (wall, fence, curb, path, deck ...). Centre (cx,cz), size (sx,sz), from y0 up by h. Has a collider.
    public static GameObject Slab(string name, float cx, float cz, float sx, float sz, float y0, float h, Material mat, bool collider = true)
    {
        var g = GameObject.CreatePrimitive(PrimitiveType.Cube);
        g.name = name;
        g.transform.SetParent(K.transform.Find("Props"), false);
        g.transform.localPosition = new Vector3(cx, y0 + h / 2f, cz);
        g.transform.localScale = new Vector3(sx, h, sz);
        g.GetComponent<MeshRenderer>().sharedMaterial = mat;
        if (!collider) Object.DestroyImmediate(g.GetComponent<Collider>());
        return g;
    }

    public static Transform Find(string prefix, float xmin = -99, float xmax = 99, float zmin = -99, float zmax = 99)
    {
        foreach (Transform t in K.transform.Find("Props"))
        {
            if (!t.name.StartsWith(prefix)) continue;
            Vector3 c = K.transform.InverseTransformPoint(B(t.gameObject).center);
            if (c.x >= xmin && c.x <= xmax && c.z >= zmin && c.z <= zmax) return t;
        }
        return null;
    }

    static bool Container(string n)
    {
        return n.Contains("Shelf") || n.Contains("shelves") || n.Contains("bookshelf") || n.StartsWith("Bed") || n.StartsWith("Sofa") || n.StartsWith("Drawer")
            || n.StartsWith("Cabinet") || n.StartsWith("Closet") || n.StartsWith("Table") || n.Contains("table") || n.Contains("desk") || n.Contains("Desk")
            || n.Contains("Refrigerator") || n.Contains("cabinet") || n.Contains("dresser") || n.Contains("Dresser");
    }

    /// Prop-prop overlaps (containers vs their contents are skipped) and props poking through the shell walls.
    public static string Audit(Zone z)
    {
        var sb = new StringBuilder();
        var props = new List<Transform>();
        foreach (Transform t in z.transform.Find("Props")) { if (t.name.StartsWith("Rug") || t.GetComponentInChildren<Renderer>() == null) continue; props.Add(t); }
        for (int i = 0; i < props.Count; i++)
            for (int j = i + 1; j < props.Count; j++)
            {
                string a = props[i].name, b = props[j].name;
                bool wa = a.StartsWith("Window"), wb = b.StartsWith("Window");
                if (wa && wb) continue;
                if (!wa && !wb && (Container(a) || Container(b))) continue;
                bool hit = false;
                foreach (var ra in props[i].GetComponentsInChildren<Renderer>())
                {
                    if (ra.name.StartsWith("WindowGlass") || ra.name.StartsWith("Frame") || ra.name.StartsWith("Mullion")) continue;
                    foreach (var rb in props[j].GetComponentsInChildren<Renderer>())
                    {
                        if (rb.name.StartsWith("WindowGlass") || rb.name.StartsWith("Frame") || rb.name.StartsWith("Mullion")) continue;
                        Bounds A = ra.bounds, Bb = rb.bounds;
                        float ox = Mathf.Min(A.max.x, Bb.max.x) - Mathf.Max(A.min.x, Bb.min.x);
                        float oy = Mathf.Min(A.max.y, Bb.max.y) - Mathf.Max(A.min.y, Bb.min.y);
                        float oz = Mathf.Min(A.max.z, Bb.max.z) - Mathf.Max(A.min.z, Bb.min.z);
                        if (ox > 0.04f && oy > 0.04f && oz > 0.04f) { hit = true; break; }
                    }
                    if (hit) break;
                }
                if (hit) sb.AppendLine("  OVERLAP " + a + " x " + b);
            }
        foreach (var t in props)
        {
            Bounds b = B(t.gameObject);
            if (b.size.x < 0.12f || b.size.z < 0.12f || t.name.StartsWith("(Prb)Door") || t.name.StartsWith("Window")) continue;
            Vector3 mn = z.transform.InverseTransformPoint(b.min), mx = z.transform.InverseTransformPoint(b.max);
            float x0 = Mathf.Min(mn.x, mx.x), x1 = Mathf.Max(mn.x, mx.x), z0 = Mathf.Min(mn.z, mx.z), z1 = Mathf.Max(mn.z, mx.z);
            if (x0 < -z.width / 2 - 0.06f || x1 > z.width / 2 + 0.06f || z0 < -z.depth / 2 - 0.06f || z1 > z.depth / 2 + 0.06f)
                sb.AppendLine("  THROUGH-WALL " + t.name);
        }
        return sb.ToString();
    }

    public static string Check(Zone z)
    {
        var issues = new List<string>();
        Physics.SyncTransforms();
        z.Validate(issues);
        return z.name + " validate: " + (issues.Count == 0 ? "OK" : string.Join(" | ", issues)) + " || " + z.Coverage() + "\n" + Audit(z);
    }

    /// Render the dataset camera for one zone into EnvTests/<folder>/<name>.png (with a temporary XBot, or `character` prefab path).
    public static void Shot(Zone z, string folder, string name, int preset = 2, string character = "Assets/XBot.fbx")
    {
        var zones = Object.FindObjectsByType<Zone>();
        var xbot = (GameObject)PrefabUtility.InstantiatePrefab(AssetDatabase.LoadAssetAtPath<GameObject>(character));
        xbot.transform.SetPositionAndRotation(z.SpawnPoint.position, z.SpawnPoint.rotation);
        var camGO = new GameObject("__TmpCam");
        var cam = camGO.AddComponent<Camera>();
        cam.clearFlags = CameraClearFlags.SolidColor; cam.backgroundColor = new Color(0.1f, 0.1f, 0.1f);
        cam.fieldOfView = RigSettings.FieldOfView; cam.nearClipPlane = RigSettings.NearClip;
        foreach (var o in zones) o.SetLightsActive(o == z);
        z.ApplyPreset(preset);
        camGO.transform.SetPositionAndRotation(z.CameraAnchor.position, z.CameraAnchor.rotation);
        var rt = new RenderTexture(1920, 1080, 24) { antiAliasing = 4 };
        cam.targetTexture = rt; cam.Render();
        RenderTexture.active = rt;
        var tex = new Texture2D(1920, 1080, TextureFormat.RGB24, false);
        tex.ReadPixels(new Rect(0, 0, 1920, 1080), 0, 0); tex.Apply();
        cam.targetTexture = null; RenderTexture.active = null;
        string dir = Path.Combine(Path.GetDirectoryName(Application.dataPath), "EnvTests", folder);
        Directory.CreateDirectory(dir);
        File.WriteAllBytes(Path.Combine(dir, name + ".png"), tex.EncodeToPNG());
        Object.DestroyImmediate(rt); Object.DestroyImmediate(tex);
        foreach (var o in zones) { o.SetLightsActive(true); o.ApplyPreset(2); }
        Object.DestroyImmediate(xbot); Object.DestroyImmediate(camGO);
    }

    public static void Save()
    {
        Physics.SyncTransforms();
        EditorSceneManager.SaveScene(EditorSceneManager.GetActiveScene());
        AssetDatabase.SaveAssets();
    }
}
