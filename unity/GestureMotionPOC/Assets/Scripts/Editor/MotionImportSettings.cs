using UnityEditor;
using UnityEngine;

/// Every FBX under Assets/Motion/ is a Kimodo/BVH-derived motion clip: import it as Humanoid
/// with its own avatar and animation, so any Humanoid character can play it.
public class MotionImportSettings : AssetPostprocessor
{
    void OnPreprocessModel()
    {
        if (!assetPath.StartsWith("Assets/Motion/")) return;
        var mi = (ModelImporter)assetImporter;
        mi.animationType = ModelImporterAnimationType.Human;
        mi.avatarSetup = ModelImporterAvatarSetup.CreateFromThisModel;
        mi.importAnimation = true;
        mi.importBlendShapes = false;
        mi.importCameras = false;
        mi.importLights = false;
        mi.materialImportMode = ModelImporterMaterialImportMode.None;
    }
}
