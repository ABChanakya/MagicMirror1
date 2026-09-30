import bpy, os, glob, json

src_dirs = [os.path.expanduser("~/Downloads/motion_work/bvh_export")]
extra = [os.path.expanduser("~/Downloads/motion_work/real472_pure_ik.bvh"),
         os.path.expanduser("~/Downloads/motion_work/test_swipe_right.bvh")]
out_dir = os.path.expanduser("~/Downloads/motion_work/fbx")
os.makedirs(out_dir, exist_ok=True)

files = extra + sorted(glob.glob(os.path.join(src_dirs[0], "*.bvh")))
report = []
for path in files:
    name = os.path.splitext(os.path.basename(path))[0]
    try:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_anim.bvh(filepath=path, global_scale=0.01, rotate_mode='NATIVE',
                                axis_forward='-Z', axis_up='Y', update_scene_fps=True)
        arm = [o for o in bpy.data.objects if o.type == 'ARMATURE'][0]
        sc = bpy.context.scene
        act = arm.animation_data.action
        f0, f1 = int(act.frame_range[0]), int(act.frame_range[1])
        sc.frame_start, sc.frame_end = f0, f1
        bpy.ops.object.select_all(action='DESELECT')
        arm.select_set(True)
        bpy.context.view_layer.objects.active = arm
        out = os.path.join(out_dir, name + ".fbx")
        bpy.ops.export_scene.fbx(filepath=out, use_selection=True, object_types={'ARMATURE'},
                                 bake_anim=True, bake_anim_use_all_bones=True, bake_anim_use_nla_strips=False,
                                 bake_anim_use_all_actions=False, bake_anim_force_startend_keying=True,
                                 add_leaf_bones=False, apply_scale_options='FBX_SCALE_ALL',
                                 axis_forward='-Z', axis_up='Y', primary_bone_axis='Y', secondary_bone_axis='X')
        report.append((name, f1 - f0 + 1, "ok"))
    except Exception as e:
        report.append((name, 0, "FAIL " + str(e)[:80]))
print("REPORT_JSON=" + json.dumps(report))
