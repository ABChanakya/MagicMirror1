using System;
using System.IO;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

/// One-shot render-quality upgrade for the dataset: full-resolution character textures, MSAA 4x, forced anisotropic filtering.
public static class Quality
{
    public static void Apply(Action<string> log)
    {
        int tex = 0, up = 0;
        foreach (string g in AssetDatabase.FindAssets("t:Texture2D", new[] { "Assets/Characters" }))
        {
            string p = AssetDatabase.GUIDToAssetPath(g);
            var ti = AssetImporter.GetAtPath(p) as TextureImporter;
            if (ti == null) continue;
            ti.GetSourceTextureWidthAndHeight(out int w, out int h);
            int want = Mathf.Min(4096, Mathf.NextPowerOfTwo(Mathf.Max(w, h)));
            bool changed = false;
            if (ti.maxTextureSize < want) { ti.maxTextureSize = want; changed = true; up++; }
            if (ti.anisoLevel < 8) { ti.anisoLevel = 8; changed = true; }
            if (ti.filterMode != FilterMode.Trilinear) { ti.filterMode = FilterMode.Trilinear; changed = true; }
            if (ti.textureCompression == TextureImporterCompression.Compressed) { ti.textureCompression = TextureImporterCompression.CompressedHQ; changed = true; }
            if (changed) ti.SaveAndReimport();
            tex++;
        }
        log("character textures checked " + tex + ", raised max size on " + up);
        var rp = GraphicsSettings.currentRenderPipeline as UniversalRenderPipelineAsset;
        if (rp != null) { rp.msaaSampleCount = 4; EditorUtility.SetDirty(rp); log("URP asset '" + rp.name + "' MSAA -> " + rp.msaaSampleCount + ", renderScale " + rp.renderScale); }
        else log("no URP asset active");
        for (int i = 0; i < QualitySettings.names.Length; i++)
        {
            QualitySettings.SetQualityLevel(i, false);
            QualitySettings.anisotropicFiltering = AnisotropicFiltering.ForceEnable;
            var lvl = QualitySettings.renderPipeline as UniversalRenderPipelineAsset;
            if (lvl != null && lvl.msaaSampleCount < 4) { lvl.msaaSampleCount = 4; EditorUtility.SetDirty(lvl); }
        }
        AssetDatabase.SaveAssets();
        log("QUALITY DONE");
    }
}
