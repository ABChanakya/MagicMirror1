using UnityEngine;

/// Single source of truth for the dataset camera rig and per-zone clearance rules.
/// Used by BatchDatasetGenerator, Zone and ZoneBuilder so they cannot drift apart.
public static class RigSettings
{
    public const float CameraHeight = 2.5f;
    public const float CameraForwardOffset = 1.0f;
    public const float TargetHeight = 1.3f;
    public const float FieldOfView = 75f;   // vertical
    public const float NearClip = 0.05f;

    // Free space around the SpawnPoint (metres). "Front" is the camera side.
    public const float ClearFront = 2.0f;
    public const float ClearSide = 1.5f;
    public const float ClearBack = 1.5f;
    public const float PersonHeight = 2.2f;   // person plus raised arms
    public const float CameraRadius = 0.15f;  // camera must have this much free space around it

    public const float MinCeiling = 3.2f;

    public static void CameraPose(Vector3 spawnPos, Quaternion spawnRot, out Vector3 pos, out Quaternion rot)
    {
        Vector3 forward = spawnRot * Vector3.forward;
        pos = spawnPos + forward * CameraForwardOffset + Vector3.up * CameraHeight;
        rot = Quaternion.LookRotation((spawnPos + Vector3.up * TargetHeight) - pos, Vector3.up);
    }
}
