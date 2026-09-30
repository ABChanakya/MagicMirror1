using System.IO;
using UnityEditor;
using UnityEngine;

/// Helpers that add rugs, windows and curtains to a Zone's "Props" (regenerated shells never touch these).
/// Fabric materials come from CC0 Poly Haven textures in Assets/Environments/PolyHavenTex/<id>/.
public static class ZoneDecor
{
    const string TexDir = "Assets/Environments/PolyHavenTex/";
    const string MatDir = "Assets/Environments/Materials/";

    public static Material FabricMaterial(string texId, Color tint, Vector2 tiling, float smoothness = 0.15f)
    {
        string path = MatDir + "Fabric_" + texId + "_" + ColorUtility.ToHtmlStringRGB(tint) + "_" + tiling.x.ToString("F1") + "x" + tiling.y.ToString("F1") + ".mat";
        var m = AssetDatabase.LoadAssetAtPath<Material>(path);
        if (m != null) return m;
        m = new Material(Shader.Find("Universal Render Pipeline/Lit"));
        var diff = AssetDatabase.LoadAssetAtPath<Texture2D>(TexDir + texId + "/" + texId + "_diff_1k.jpg");
        string norPath = TexDir + texId + "/" + texId + "_nor_gl_1k.jpg";
        var imp = AssetImporter.GetAtPath(norPath) as TextureImporter;
        if (imp != null && imp.textureType != TextureImporterType.NormalMap) { imp.textureType = TextureImporterType.NormalMap; imp.SaveAndReimport(); }
        var nor = AssetDatabase.LoadAssetAtPath<Texture2D>(norPath);
        m.SetTexture("_BaseMap", diff);
        m.SetTextureScale("_BaseMap", tiling);
        m.SetColor("_BaseColor", tint);
        if (nor != null)
        {
            m.SetTexture("_BumpMap", nor);
            m.SetTextureScale("_BumpMap", tiling);
            m.EnableKeyword("_NORMALMAP");
        }
        m.SetFloat("_Smoothness", smoothness);
        AssetDatabase.CreateAsset(m, path);
        return m;
    }

    static Material Plain(string name, Color c, float smooth, bool emissive = false)
    {
        string path = MatDir + name + ".mat";
        var m = AssetDatabase.LoadAssetAtPath<Material>(path);
        if (m != null) return m;
        m = new Material(Shader.Find("Universal Render Pipeline/Lit"));
        m.SetColor("_BaseColor", c);
        m.SetFloat("_Smoothness", smooth);
        AssetDatabase.CreateAsset(m, path);
        if (emissive)
        {
            // Enable after the asset exists, otherwise the keyword is lost on save.
            m.EnableKeyword("_EMISSION");
            m.SetColor("_EmissionColor", Color.white);
            m.globalIlluminationFlags = MaterialGlobalIlluminationFlags.RealtimeEmissive;
            EditorUtility.SetDirty(m);
            AssetDatabase.SaveAssetIfDirty(m);
        }
        return m;
    }

    static GameObject Cube(Transform parent, string name, Vector3 localPos, Vector3 size, Material mat, bool collider)
    {
        var g = GameObject.CreatePrimitive(PrimitiveType.Cube);
        g.name = name;
        g.transform.SetParent(parent, false);
        g.transform.localPosition = localPos;
        g.transform.localScale = size;
        g.GetComponent<MeshRenderer>().sharedMaterial = mat;
        if (!collider) Object.DestroyImmediate(g.GetComponent<Collider>());
        return g;
    }

    /// Flat rug at zone-local (x, z). No collider: it is 2 cm tall and must not count as an obstacle.
    public static GameObject AddRug(Zone z, string name, float x, float zz, float width, float depth, float yRot, string texId, Color tint, Color borderTint)
    {
        var root = new GameObject(name);
        root.transform.SetParent(z.transform.Find("Props"), false);
        root.transform.localPosition = new Vector3(x, 0, zz);
        root.transform.localRotation = Quaternion.Euler(0, yRot, 0);
        var border = Cube(root.transform, "Border", new Vector3(0, 0.006f, 0), new Vector3(width + 0.14f, 0.012f, depth + 0.14f), FabricMaterial(texId, borderTint, new Vector2((width + 0.14f) / 1.2f, (depth + 0.14f) / 1.2f)), false);
        var body = Cube(root.transform, "Body", new Vector3(0, 0.016f, 0), new Vector3(width, 0.008f, depth), FabricMaterial(texId, tint, new Vector2(width / 1.2f, depth / 1.2f)), false);
        border.GetComponent<MeshRenderer>().shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        body.GetComponent<MeshRenderer>().shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        return root;
    }

    /// Window panel flush on a wall with emissive glass (follows lighting presets), frame, mullions,
    /// a light that spills into the room, and optional curtains. `offset` is along the wall (+Z for Left/Right, +X for Back/Front).
    public static GameObject AddWindow(Zone z, string name, DoorWall wall, float offset, float width, float height, float sill, string curtainTex, Color curtainTint)
    {
        Vector3 n; Vector3 t; float dist;
        switch (wall)
        {
            case DoorWall.Left:  n = Vector3.left;    t = Vector3.forward; dist = z.width / 2; break;
            case DoorWall.Right: n = Vector3.right;   t = Vector3.forward; dist = z.width / 2; break;
            case DoorWall.Back:  n = Vector3.back;    t = Vector3.right;   dist = z.depth / 2; break;
            default:             n = Vector3.forward; t = Vector3.right;   dist = z.depth / 2; break;
        }
        var root = new GameObject(name);
        root.transform.SetParent(z.transform.Find("Props"), false);
        root.transform.localPosition = n * dist + t * offset;
        root.transform.localRotation = Quaternion.LookRotation(-n, Vector3.up); // local +Z points into the room

        var frameM = Plain("Window_Frame", new Color(0.93f, 0.93f, 0.9f), 0.3f);
        var glassM = Plain("Window_Glass", new Color(0.05f, 0.05f, 0.05f), 0.9f, true);
        float fw = 0.06f, fd = 0.07f, cy = sill + height / 2;

        var glass = Cube(root.transform, "WindowGlass", new Vector3(0, cy, 0.006f), new Vector3(width, height, 0.01f), glassM, false);
        glass.GetComponent<MeshRenderer>().shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        Cube(root.transform, "FrameL", new Vector3(-width / 2 - fw / 2, cy, fd / 2), new Vector3(fw, height + 2 * fw, fd), frameM, false);
        Cube(root.transform, "FrameR", new Vector3(width / 2 + fw / 2, cy, fd / 2), new Vector3(fw, height + 2 * fw, fd), frameM, false);
        Cube(root.transform, "FrameT", new Vector3(0, sill + height + fw / 2, fd / 2), new Vector3(width + 2 * fw, fw, fd), frameM, false);
        Cube(root.transform, "FrameB", new Vector3(0, sill - fw / 2, fd / 2 + 0.03f), new Vector3(width + 2 * fw + 0.1f, fw, fd + 0.06f), frameM, false);
        Cube(root.transform, "MullionV", new Vector3(0, cy, fd / 2), new Vector3(0.03f, height, 0.03f), frameM, false);
        Cube(root.transform, "MullionH", new Vector3(0, cy, fd / 2), new Vector3(width, 0.03f, 0.03f), frameM, false);

        if (!string.IsNullOrEmpty(curtainTex))
        {
            float cw = Mathf.Max(0.35f, width * 0.32f), top = sill + height + 0.25f, bottom = 0.12f, ch = top - bottom;
            var cm = FabricMaterial(curtainTex, curtainTint, new Vector2(cw / 0.8f, ch / 0.8f), 0.1f);
            Cube(root.transform, "CurtainL", new Vector3(-width / 2 - cw / 2 + 0.05f, bottom + ch / 2, 0.11f), new Vector3(cw, ch, 0.09f), cm, false);
            Cube(root.transform, "CurtainR", new Vector3(width / 2 + cw / 2 - 0.05f, bottom + ch / 2, 0.11f), new Vector3(cw, ch, 0.09f), cm, false);
            Cube(root.transform, "CurtainRod", new Vector3(0, top + 0.03f, 0.11f), new Vector3(width + 2 * cw + 0.1f, 0.03f, 0.03f), Plain("Curtain_Rod", new Color(0.2f, 0.2f, 0.2f), 0.5f), false);
        }

        // Light spilling into the room; colour and intensity follow the preset.
        var lg = new GameObject("Win@2");
        lg.transform.SetParent(root.transform, false);
        lg.transform.localPosition = new Vector3(0, cy, 0.7f);
        var l = lg.AddComponent<Light>();
        l.type = LightType.Point; l.range = 4.5f; l.intensity = 2f; l.shadows = LightShadows.None;
        l.lightmapBakeType = LightmapBakeType.Mixed;
        return root;
    }

    // ---- Poly Haven cut-out leaf fix ------------------------------------------------------------
    // Poly Haven ships leaf colour and alpha as separate files and Unity imports the leaf material
    // untextured (white). This builds an RGBA texture + URP alpha-clip material per plant (new files
    // only) and assigns it on every prop renderer in the zone that still uses the untextured import.
    public static int FixLeafMaterials(Zone z)
    {
        int n = 0;
        Transform props = z.transform.Find("Props");
        if (props == null) return 0;
        foreach (var r in props.GetComponentsInChildren<Renderer>(true))
        {
            var mats = r.sharedMaterials; bool ch = false;
            for (int i = 0; i < mats.Length; i++)
            {
                var m = mats[i];
                if (m == null || !m.name.EndsWith("_leaves") || m.name.EndsWith("_urp")) continue;
                Transform root = r.transform;
                while (root.parent != null && root.parent.name != "Props") root = root.parent;
                string id = root.name.Replace("_1k", "").Replace("(Clone)", "");
                var lm = LeafMaterial(id, m.name);
                if (lm != null) { mats[i] = lm; ch = true; n++; }
            }
            if (ch) r.sharedMaterials = mats;
        }
        return n;
    }

    static Texture2D LoadPng(string p)
    {
        var t = new Texture2D(2, 2, TextureFormat.RGBA32, false);
        t.LoadImage(File.ReadAllBytes(p));
        return t;
    }

    static Material LeafMaterial(string plantId, string matName)
    {
        string dir = "Assets/Environments/PolyHaven/" + plantId + "/textures";
        string matPath = "Assets/Environments/PolyHaven/" + plantId + "/" + matName + "_urp.mat";
        var m = AssetDatabase.LoadAssetAtPath<Material>(matPath);
        if (m != null) return m;
        string diffP = dir + "/" + matName + "_diff_1k.png", alphaP = dir + "/" + matName + "_alpha_1k.png", outP = dir + "/" + matName + "_rgba_1k.png";
        if (!File.Exists(diffP)) return null;
        if (!File.Exists(outP))
        {
            var d = LoadPng(diffP);
            Color32[] px = d.GetPixels32();
            if (File.Exists(alphaP))
            {
                var a = LoadPng(alphaP);
                if (a.width == d.width && a.height == d.height)
                {
                    Color32[] ap = a.GetPixels32();
                    for (int i = 0; i < px.Length; i++) px[i].a = ap[i].r;
                }
            }
            var o = new Texture2D(d.width, d.height, TextureFormat.RGBA32, false);
            o.SetPixels32(px); o.Apply();
            File.WriteAllBytes(outP, o.EncodeToPNG());
            AssetDatabase.ImportAsset(outP, ImportAssetOptions.ForceSynchronousImport);
        }
        m = new Material(Shader.Find("Universal Render Pipeline/Lit"));
        m.SetTexture("_BaseMap", AssetDatabase.LoadAssetAtPath<Texture2D>(outP));
        string norP = dir + "/" + matName + "_nor_gl_1k.png";
        if (File.Exists(norP))
        {
            var imp = (TextureImporter)AssetImporter.GetAtPath(norP);
            if (imp.textureType != TextureImporterType.NormalMap) { imp.textureType = TextureImporterType.NormalMap; imp.SaveAndReimport(); }
            m.SetTexture("_BumpMap", AssetDatabase.LoadAssetAtPath<Texture2D>(norP));
            m.EnableKeyword("_NORMALMAP");
        }
        m.SetFloat("_AlphaClip", 1f); m.SetFloat("_Cutoff", 0.5f); m.EnableKeyword("_ALPHATEST_ON");
        m.SetFloat("_Cull", 0f); m.SetFloat("_Smoothness", 0.3f); m.renderQueue = 2450;
        AssetDatabase.CreateAsset(m, matPath);
        return m;
    }
}
