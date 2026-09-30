using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

/// Generates a Zone's shell, lights, light probes, SpawnPoint and CameraAnchor from its parameters.
/// Furniture lives under the zone's "Props" child, which this never touches.
/// Lighting model: shell is static and baked (indirect only); lights are Mixed so presets can still
/// change direct colour/intensity; props and the character are lit by the zone's light probes.
public static class ZoneBuilder
{
    const string MatDir = "Assets/Environments/Materials/";
    const float Wall = 0.1f;

    [MenuItem("GameObject/Zone/Create Zone", false, 10)]
    static void CreateZone()
    {
        var go = new GameObject("Zone");
        go.AddComponent<Zone>();
        Rebuild(go.GetComponent<Zone>());
        Selection.activeGameObject = go;
        Undo.RegisterCreatedObjectUndo(go, "Create Zone");
    }

    public static void Rebuild(Zone z)
    {
        if (z.ceilingHeight < RigSettings.MinCeiling)
        {
            Debug.LogWarning($"[ZoneBuilder] '{z.name}': ceiling raised from {z.ceilingHeight} to {RigSettings.MinCeiling} m (minimum).");
            z.ceilingHeight = RigSettings.MinCeiling;
        }
        if (z.floorMaterial == null) z.floorMaterial = AssetDatabase.LoadAssetAtPath<Material>(MatDir + "Room_Floor.mat");
        if (z.wallMaterial == null) z.wallMaterial = AssetDatabase.LoadAssetAtPath<Material>(MatDir + "Room_Wall.mat");
        if (z.ceilingMaterial == null) z.ceilingMaterial = z.wallMaterial;

        foreach (string n in new[] { "Shell", "Lights", "Probes", "SpawnPoint", "CameraAnchor" })
        {
            Transform old = z.transform.Find(n);
            if (old != null) Object.DestroyImmediate(old.gameObject);
        }
        if (z.transform.Find("Props") == null)
            new GameObject("Props").transform.SetParent(z.transform, false);

        float W = z.width, D = z.depth, H = z.ceilingHeight;
        Transform shell = new GameObject("Shell").transform;
        shell.SetParent(z.transform, false);

        if (z.outdoor)
        {
            // Open ground far beyond the yard so no void shows at the frame edge (floor is visible ~11 m out).
            Box(shell, "Floor", new Vector3(0, -Wall / 2, 0), new Vector3(W + 60f, Wall, D + 60f), z.floorMaterial);
        }
        else
        {
            Box(shell, "Floor", new Vector3(0, -Wall / 2, 0), new Vector3(W, Wall, D), z.floorMaterial);
            Box(shell, "Ceiling", new Vector3(0, H + Wall / 2, 0), new Vector3(W + 2 * Wall, Wall, D + 2 * Wall), z.ceilingMaterial);
        }

        float dw = Mathf.Min(z.doorWidth, Mathf.Min(W, D) - 0.4f);
        float dh = Mathf.Min(z.doorHeight, H - 0.3f);
        Vector3 doorLightPos = Vector3.zero;
        foreach (DoorWall side in z.outdoor ? new DoorWall[0] : new[] { DoorWall.Left, DoorWall.Right, DoorWall.Back, DoorWall.Front })
        {
            Vector3 n, t; float len, dist;
            switch (side)
            {
                case DoorWall.Left:  n = Vector3.left;    t = Vector3.forward; len = D; dist = W / 2; break;
                case DoorWall.Right: n = Vector3.right;   t = Vector3.forward; len = D; dist = W / 2; break;
                case DoorWall.Back:  n = Vector3.back;    t = Vector3.right;   len = W; dist = D / 2; break;
                default:             n = Vector3.forward; t = Vector3.right;   len = W; dist = D / 2; break;
            }
            float ext = (side == DoorWall.Back || side == DoorWall.Front) ? Wall : 0f; // close the corners
            float lo = -len / 2 - ext, hi = len / 2 + ext;
            string nm = "Wall_" + side;
            if (side != z.doorWall)
            {
                Seg(shell, nm, n, t, dist, lo, hi, 0, H, z.wallMaterial);
                continue;
            }
            float c = Mathf.Clamp(z.doorOffset, -len / 2 + dw / 2 + 0.2f, len / 2 - dw / 2 - 0.2f);
            float a0 = c - dw / 2, a1 = c + dw / 2;
            Seg(shell, nm + "_A", n, t, dist, lo, a0, 0, H, z.wallMaterial);
            Seg(shell, nm + "_B", n, t, dist, a1, hi, 0, H, z.wallMaterial);
            Seg(shell, nm + "_Lintel", n, t, dist, a0, a1, dh, H, z.wallMaterial);

            // Enclosed stub behind the doorway so it never shows a black void.
            float sd = 1.5f, sw = dw + 0.6f, sh = dh + 0.4f;
            float d0 = dist + Wall;                       // outer face of the wall
            Seg(shell, "Stub_Floor", n, t, d0 + sd / 2, c - sw / 2, c + sw / 2, -Wall, 0, z.floorMaterial, sd);
            Seg(shell, "Stub_Ceiling", n, t, d0 + sd / 2, c - sw / 2, c + sw / 2, sh, sh + Wall, z.ceilingMaterial, sd);
            Seg(shell, "Stub_End", n, t, d0 + sd, c - sw / 2, c + sw / 2, 0, sh, z.wallMaterial);
            Seg(shell, "Stub_SideA", n, t, d0 + sd / 2, c - sw / 2 - Wall, c - sw / 2, 0, sh, z.wallMaterial, sd);
            Seg(shell, "Stub_SideB", n, t, d0 + sd / 2, c + sw / 2, c + sw / 2 + Wall, 0, sh, z.wallMaterial, sd);
            doorLightPos = n * (dist + 0.8f) + t * c + Vector3.up * dh;
        }

        // Lights: Mixed (indirect baked, direct realtime, no realtime shadows). Names carry the base
        // intensity ("label@intensity") so presets scale from it.
        Transform lights = new GameObject("Lights").transform;
        lights.SetParent(z.transform, false);
        if (z.outdoor)
        {
            var sun = new GameObject("Sun@2");
            sun.transform.SetParent(lights, false);
            sun.transform.position = z.transform.position + Vector3.up * 8f;
            var sl = sun.AddComponent<Light>();
            sl.type = LightType.Directional; sl.intensity = 2f; sl.shadows = LightShadows.Soft;
            sl.lightmapBakeType = LightmapBakeType.Realtime;
            sun.transform.rotation = z.transform.rotation * Quaternion.Euler(52f, -30f, 0f);
        }
        else
        {
            AddLight(lights, "Main@6", new Vector3(0, H - 0.4f, 0), 10f, 6f);
            AddLight(lights, "Back@3", new Vector3(0, H - 0.4f, -D / 2 + 1f), 7f, 3f);
            AddLight(lights, "Front@3", new Vector3(0, H - 0.4f, D / 2 - 1f), 7f, 3f);
            AddLight(lights, "Door@1.5", doorLightPos, 3f, 1.5f);
        }

        // Light probes for props and the character (props are not lightmapped).
        Transform probes = new GameObject("Probes").transform;
        probes.SetParent(z.transform, false);
        var pg = probes.gameObject.AddComponent<LightProbeGroup>();
        var pos = new List<Vector3>();
        foreach (float y in new[] { 0.4f, 1.3f, 2.3f })
            for (int ix = 0; ix < 4; ix++)
                for (int iz = 0; iz < 4; iz++)
                    pos.Add(new Vector3(Mathf.Lerp(-W / 2 + 0.3f, W / 2 - 0.3f, ix / 3f), y, Mathf.Lerp(-D / 2 + 0.3f, D / 2 - 0.3f, iz / 3f)));
        pg.probePositions = pos.ToArray();

        Transform sp = new GameObject("SpawnPoint").transform;
        sp.SetParent(z.transform, false);
        sp.localPosition = new Vector3(0, 0, -D / 2 + z.spawnBackOffset);
        sp.localRotation = Quaternion.identity;

        Vector3 camPos; Quaternion camRot;
        RigSettings.CameraPose(sp.position, sp.rotation, out camPos, out camRot);
        Transform anchor = new GameObject("CameraAnchor").transform;
        anchor.SetParent(z.transform, false);
        anchor.SetPositionAndRotation(camPos, camRot);

        EditorUtility.SetDirty(z);
    }

    // Adds colliders to every prop mesh that has none, so the validator can see it.
    public static void AddPropColliders(Zone z)
    {
        Transform props = z.transform.Find("Props");
        if (props == null) return;
        foreach (MeshFilter mf in props.GetComponentsInChildren<MeshFilter>())
            if (mf.GetComponent<Collider>() == null) mf.gameObject.AddComponent<BoxCollider>();
    }

    /// Low-cost bake settings: indirect-only mixed lighting, small lightmaps, few samples.
    public static LightingSettings EnsureLightingSettings()
    {
        const string path = "Assets/Environments/DatasetWorld_LightingSettings.lighting";
        var ls = AssetDatabase.LoadAssetAtPath<LightingSettings>(path);
        if (ls == null)
        {
            ls = new LightingSettings();
            AssetDatabase.CreateAsset(ls, path);
        }
        ls.bakedGI = true;
        ls.realtimeGI = false;
        ls.lightmapper = LightingSettings.Lightmapper.ProgressiveCPU;
        ls.mixedBakeMode = MixedLightingMode.IndirectOnly;
        ls.directSampleCount = 16;
        ls.indirectSampleCount = 64;
        ls.environmentSampleCount = 64;
        ls.maxBounces = 2;
        ls.lightmapResolution = 8;
        ls.lightmapMaxSize = 1024;
        ls.lightmapCompression = LightmapCompression.NormalQuality;
        ls.filteringMode = LightingSettings.FilterMode.Auto;
        EditorUtility.SetDirty(ls);
        AssetDatabase.SaveAssets();
        return ls;
    }

    // Wall/stub segment in a frame where n = outward normal, t = tangent. Thickness along n is
    // `Wall` unless `thick` is given (then it is centred at `dist`).
    static void Seg(Transform parent, string name, Vector3 n, Vector3 t, float dist, float u0, float u1, float y0, float y1, Material m, float thick = -1f)
    {
        float th = thick > 0 ? thick : Wall;
        float nd = thick > 0 ? dist : dist + Wall / 2;
        Vector3 centre = n * nd + t * ((u0 + u1) / 2f) + Vector3.up * ((y0 + y1) / 2f);
        Vector3 size = new Vector3(Mathf.Abs(n.x), 0, Mathf.Abs(n.z)) * th
                     + new Vector3(Mathf.Abs(t.x), 0, Mathf.Abs(t.z)) * (u1 - u0)
                     + Vector3.up * (y1 - y0);
        Box(parent, name, centre, size, m);
    }

    static GameObject Box(Transform parent, string name, Vector3 pos, Vector3 size, Material mat)
    {
        GameObject g = GameObject.CreatePrimitive(PrimitiveType.Cube);
        g.name = name;
        g.transform.SetParent(parent, false);
        g.transform.localPosition = pos;
        g.transform.localScale = size;
        g.GetComponent<MeshRenderer>().sharedMaterial = mat;
        g.isStatic = true;
        return g;
    }

    static void AddLight(Transform parent, string name, Vector3 pos, float range, float intensity)
    {
        var g = new GameObject(name);
        g.transform.SetParent(parent, false);
        g.transform.localPosition = pos;
        var l = g.AddComponent<Light>();
        l.type = LightType.Point;
        l.range = range;
        l.intensity = intensity;
        l.shadows = LightShadows.None;
        l.lightmapBakeType = LightmapBakeType.Mixed;
    }
}

[CustomEditor(typeof(Zone))]
public class ZoneEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();
        var z = (Zone)target;
        EditorGUILayout.Space();
        if (GUILayout.Button("Rebuild shell, lights, spawn & camera anchor")) ZoneBuilder.Rebuild(z);
        if (GUILayout.Button("Add colliders to props")) ZoneBuilder.AddPropColliders(z);
        if (GUILayout.Button("Validate"))
        {
            var issues = new List<string>();
            z.Validate(issues);
            if (issues.Count == 0) Debug.Log($"[Zone] '{z.name}' OK");
            else foreach (string s in issues) Debug.LogWarning($"[Zone] '{z.name}': {s}");
        }
        for (int i = 0; i < z.presets.Count; i++)
            if (GUILayout.Button("Apply lighting: " + z.presets[i].name)) z.ApplyPreset(i);
    }
}
