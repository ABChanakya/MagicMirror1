# Environment build log

Project: unity/GestureMotionPOC — Unity 6000.6.2f1, URP 17.6.0 (Universal Render Pipeline).
Only URP-compatible (or built-in assets that can be upgraded via Render Pipeline Converter) packs should be used.

## 2026-09-29
- Step 1 (Unity Asset Store): started. Pipeline check done: URP.
- BatchDatasetGenerator.cs: cameraForwardOffset 1.6 -> 1.0.
- Step 1: candidates verified via store pages (price/URP/publisher). Waiting on user to add to My Assets + import.
  - Rejected: Furniture FREE Pack (DEXSOFT) — built-in only, not URP; Office Pack: Furniture (Supercyan) — paid EUR16.47.
- Step 1: user imported 3 packs into Assets/. Inspection:
  - nappin/HouseInteriorPack: 57 prefabs (sofa, beds, kitchen island/sink/fridge, door, lights...), built-in Standard materials -> pink in URP until Render Pipeline Converter is run.
  - Furniture Mega Pack: 511 prefabs (beds, sofas, tables, chairs, kitchen, closet, drawers, bathroom); materials already URP/Lit.
  - Office Room Furniture: 10 prefabs + demo scene; built-in Standard materials -> needs converter.
  - Note: Unity_* capture tools are not exposed in this Claude session, so screenshots need another route (see chat).
- Unity MCP fixed (server --project-path pointed at wrong folder). Tools available.
- Converted 49 nappin/Office Standard materials to URP/Lit via editor script. Left alone: 2 Office Legacy/Transparent Diffuse plant mats (FloorPlant_3, Objects_2), nappin Skybox mat.
- Built Assets/Environments/LivingRoom.prefab: 5x5 m, 2.8 m walls, doorway in left wall, sofa/coffee table/media console/lamp (nappin, rescaled 0.5-0.7 since pack is oversized and long axis is Z), 2 point lights, SpawnPoint, box colliders.
- XBot test (fov 75, cam 1.0 m fwd, 2.5 m up): image at unity/GestureMotionPOC/EnvTests/living_room_xbot.png. XBot measures 1.90 m. Hands in frame, no pink.
- Unity_Camera_Capture failed ("Failed to render scene preview"); rendered test image with an editor script instead.
- Known issue: nappin MediaConsole TV mesh renders as a large tilted white panel; needs fixing or replacing.
- WAITING for user OK before building bedroom/kitchen/office.

## Plan change: one big scene, zones (report sent, awaiting OK)
- Pipeline URP 17.6; all 10 characters Humanoid (XBot + Shannon, Kate, Leonard, Louise, Remy, TheBoss, James, PeasantGirl, Megan).
- Note: the 49 nappin/Office materials were converted in place (pack assets modified). No further pack edits.
- Camera geometry: 2.5 m up, 1.0 m fwd, aim 1.3 m => ~50 deg down; at 75 deg vFOV top edge is ~13 deg below horizontal, so sky is never in frame.
- Open decisions for user: hallway/bathroom vs 3.0x3.5 m clearance rule; baked vs switchable lighting.

## Work item 2: parametric setup + first indoor zone (2026-09-29)
- User decisions: no bathroom; foyer gets relaxed clearance. Defaults I chose (not yet confirmed): realtime switchable lights (no baking), vFOV 75.
- New scripts: Assets/Scripts/Environments/{RigSettings,Zone}.cs, Assets/Scripts/Editor/ZoneBuilder.cs. BatchDatasetGenerator now takes camera constants + FOV from RigSettings.
- New scene: Assets/Scenes/DatasetWorld.unity with Zone_LivingRoom (5.5x6x3.2 m). SampleScene and LivingRoom.prefab untouched.
- Correction: Mega Pack is NOT consistently scaled (~2x oversized: sofa 1.5-1.9 m tall, closet 5.6 m). Props are sized by target height at placement.
- Validator tested: flags an object in the person volume and an object at the camera; passes clean zone.
- Unity_Camera_Capture works with the real instanceID (earlier failure was a wrong ID format). Unity_SceneView_Capture2DScene is XY-plane only; useless for a 3D floor plan (returned blank/grey).
- Doorway showed a black void; ZoneBuilder now builds an enclosed stub behind the door.
- Screenshots: unity/GestureMotionPOC/EnvTests/zone_livingroom_xbot.png (script render, pre-stub) + tool capture.

## Work item 2 continued: bedroom, door variety, baking (2026-09-29)
- ZoneBuilder v2: door wall is a parameter (Left/Right/Back/Front) with offset along that wall; enclosed stub behind any door; Mixed lights; light probe grid; shell static.
- Zone_Bedroom (5 x 6.5 x 3.2 m, door on RIGHT wall) added at x=20 in DatasetWorld.unity. Living room door stays on the left. Later zones: vary the door wall.
- Bed needs ~2.4 m depth, so bedroom depth 6.5 m and spawn 4.0 m from the back wall to keep the 1.5 m back clearance.
- Fixed: bed and closets were facing the wrong way (rotated 180 deg).
- BAKING BLOCKED: Lightmapping.BakeAsync failed: "required version of Apple Rosetta could not be found". Needs `softwareupdate --install-rosetta --agree-to-license` by the user. Lighting settings asset created (Assets/Environments/DatasetWorld_LightingSettings.lighting; CPU lightmapper, indirect-only, 8 texels/unit, 64 indirect samples). Until baked, scene is realtime with 4 unshadowed point lights per zone; only the active zone's lights need to be on.
- Baking tradeoff: baked indirect stays at the bake-time preset; presets change direct light only.
- Review images (dataset cam x3 presets, 3 corner views, top-down) per zone: unity/GestureMotionPOC/EnvTests/review/<zone>/
- Removed temp XBot_Test and __ZoneTestCam from the scene.

## Lived-in props (2026-09-29)
- Rosetta is now installed (arch -x86_64 works), so baking can be retried.
- Step 1 (Asset Store) candidates offered for lamps/books/decor: Free PBR Lamps (New Solution Studio, free, URP), Low Poly Simple Furniture FREE (Gobormu, free, URP), Books Essentials (Daniel Riches, free, URP, medieval-style), Free Kitchen - Cabinets and Equipment (Boxx-Games, free, URP). Paid ones skipped. Free store packs lack modern household clutter -> Step 2.
- Step 2 (Poly Haven, CC0): downloaded 39 models (books, plants, vases, frames, clocks, lamps, tea set, laptop, TV, etc.) to Assets/Environments/PolyHaven/ (256 MB incl. textures; consider Git LFS or .gitignore if the repo gets heavy). camera_01 404. Credits in Assets/Environments/CREDITS.md.
- Lived-in props placed in Zone_LivingRoom and Zone_Bedroom (lamps with Mixed lights that follow presets via Zone.lampFactor, pillows, plants, frames, clocks, vases, basket). Leaf materials for Poly Haven plants rebuilt as URP cut-out materials (RGBA texture made from diff+alpha; new files only). hanging_picture_frame_02 has a degenerate mesh -> replaced by fancy_picture_frame_01.
- Camera framing finding: dataset cam sees the back wall only up to ~1.65 m (living room) / ~1.4 m (bedroom, deeper room). Hang wall decor at <=1.5 m, or on side walls. Keep in mind for all later zones.
- BAKE STILL BLOCKED: Rosetta is installed (15:10) but the running Unity Editor (started 14:14) cached "not found". Restart Unity, then re-run: ZoneBuilder.EnsureLightingSettings + Lightmapping.BakeAsync (done by the "Start low-cost lightmap bake" command).

## Fuller rooms + more times of day (2026-09-29)
- Research (typical room contents): rugs, curtains/windows, wall art, plants, bookshelves with books/objects, side tables + lamps, throw pillows, baskets, clocks, mirrors (sources: living-room and bedroom essentials checklists; see chat).
- New tooling: Zone.DefaultPresets() = Morning, Midday, Daylight, Overcast, Evening, Night, Artificial (7). Lamp lights ("Lamp*@x") scale with preset.lampFactor; window lights ("Win*@x") follow preset colour+intensity; WindowGlass renderers get preset.window as emission via MaterialPropertyBlock. ZoneDecor.AddRug / AddWindow (frame, mullions, emissive glass, curtains from CC0 fabric textures, spill light). Window_Glass.mat needed EnableKeyword AFTER CreateAsset.
- Added: rugs (living: curly_teddy_natural, calm on purpose - busy plaid sat right behind the hands; bedroom: teddy under bed + floral runner), windows+curtains (living right wall, bedroom left wall), stocked bookshelves (books, vases, plants, bowls, tea set, baskets, frames, clock), more plants/baskets/pillows.
- BUG (mine, fixed): single-mesh Poly Haven FBX carry root rotation (270 X) and scale (100). My Place() reset rotation/scale -> shelves lying flat, vases/baskets/frames at 1/100 size (this was also the "degenerate frame_02"). New helpers preserve prefab root rot/scale (Spawn in the decorate script). Validator caught the lying shelves.
- Tool note: Unity_RunCommand refuses scripts that delete files ("User interactions are not supported"); delete stale images from the shell instead.
- BAKE STILL PENDING: Unity Editor must be restarted (it cached "Rosetta not found"), then run the bake command. Baking only affects static shell geometry; props are non-static, so decorating first does not invalidate the bake.

## Kitchen (started 2026-09-29)
- USER DECISION: do NOT bake now; baking deferred to later. (Needs Unity restart first because of the Rosetta cache; lighting settings asset + Mixed lights are already set up.)
- Mega Pack kitchen pieces are authored at 2x: uniform 0.5 gives real sizes (fridge 0.9x1.9 m); cabinets use 0.45 so counters are 0.91 m. Family C (grey/white) chosen for a clean, consistent look.
- Extra CC0 downloads: floor/wall textures (interior_tiles, long_white_tiles, laminate_floor_02, plank_flooring_02, ...) and kitchen props (fruit, brass pots, bin, ...). Credits in CREDITS.md.
- Zone_Kitchen built (5.5 x 6 x 3.2 m, at x=40): interior_tiles floor, white-tile backsplash, fridge + 6-cabinet run on the back wall (sink cabinet, hob, hood), 4-cabinet run on the left wall with microwave, windows (left: above counter, right: linen curtains), breakfast table + 2 chairs + lamp + fruit, stocked pantry shelf, hessian runner, plants, wall clock. Door on the BACK wall (x=+1.9) with hallway stub. Validates OK.
- Removed metal_trash_can (its open-lid mesh is 1.8 m wide). Nudged breakfast set 0.2 m to clear the person volume.
- Living room and bedroom floors now textured (laminate_floor_02 / rectangular_parquet); shells rebuilt (props untouched).
- Baking still deferred by the user.

## Layout rules (advisor, from user feedback "kitchen is better: stuff left AND right, not just back")
Camera geometry (2.5 m up, 1.0 m offset, 75 vFOV): image sides are MIRRORED (image left = zone +X/Right wall; image right = -X/Left wall). Top of frame = back wall up to ~1.4-1.65 m.
- Prime band per side: |x| >= clearSide (1.5) to the wall, z from spawn-1.5 to spawn+0.8. Visible wall height falls from ~1.9 m (spawn-1.5) to ~0.7 m (spawn line) to ~0.2 m (camera line). Best pieces 0.7-1.2 m tall; low items (plants, baskets) at spawn+0.3..+0.8 fill lower corners.
- Back corners show only as small upper corners: no hero piece there. Anything beyond spawn+1.0 is behind the camera and never seen.
- Cover each prime band with a CONTINUOUS run (counter, dresser+chair, sideboard+plant, media unit+shelf). Isolated pieces read as sparse; flat rugs and doorways read as empty.
- Doors: keep varying the wall, but never inside a prime band (back wall off-centre, side wall behind spawn-1.5, or front wall).
- Hand background: hands land ~28%/72% of width at mid-height: keep |x| 1.5-1.9 at 0.9-1.5 m plain (cabinet fronts, sofa arm, bed foot); push books/fruit/plants/patterns toward walls; avoid skin-toned props at hand height.
- Alternate which side gets the tall run; do not mirror one layout into every room.
- Foyer plan: ~4.0 m wide, clearSide/clearBack 1.0, front door (nappin Door) off-centre on back wall, shoe rack+bench+coat hanger one side, console+mirror+lamp+key bowl+boots+plant other side, runner down the centre.
- Tooling: Zone.Coverage() reports per-side % occupied and largest gap over the prime band (calibrated on the kitchen).

## Layout retrofit (advisor plan applied)
- Coverage baseline (image-right/-X | image-left/+X): Kitchen 73%/82% (reference), Living 55%/91%, Dining 36%/91%, Bedroom 64%/18%. Targets: >=70% covered, largest gap <=0.8 m.
- After retrofit: Bedroom 91%/100%, Dining 100%/91%, Living 100%/100%, Kitchen unchanged. All validate OK.
- Changes: Bedroom door -> Right wall back corner (z -2.6); dresser + mirror-art + vase/plant/photo moved into the +X band; two armchairs (Mega Chair06 @0.47) + basket; wardrobe stays -X back; floor lamp tried and removed (white shade blew out at the frame edge). Dining door -> Back wall (x -2.05); sideboard (Drawer03 @0.65) with lamp/bowl/plant + picture on the -X band. Living door -> Left wall back corner (z -2.4); closet moved into the -X band; TV stand (Drawer03 @0.42) + TV in the +X band; sideboard that sat behind the camera removed.
- Mistake caught by the validator: Mega Sofa05 "armchair" is NOT 2x-scaled like the others (1.5 m wide at scale 1); do not use it as an armchair.
- New: ZoneDecor.FixLeafMaterials(zone) (permanent helper) and Zone.Coverage().
- Before/after images: EnvTests/before/*.png vs EnvTests/after/*.png.

## Chair facing + clipping pass (user: "chairs facing the wall, lots of clipping, chairs in places that make no sense")
- Cause of wrong facing: I guessed chair front from two renders (wrong). Now the front is derived from the mesh: the backrest side = centroid of the upper vertices; front = opposite. Every chair is rotated to face its target (dining/kitchen chairs: square to the nearest table edge; armchairs: the room). Bug found: the living-room armchair had been aimed at a small SIDE table (my "nearest table" rule matched Table02); now faces the room.
- Clipping audit (per-renderer AABB overlaps between every pair of props + windows/curtains, and props vs shell walls; shelf/table/bed/sofa containers skipped as false positives since items rest on them). Found and fixed: living plant inside closet+bookshelf (removed), closet/bookshelf 5 cm overlap (moved), kitchen plant blocking the back door and through the wall (removed), kitchen wine bottles overlapping plant (removed), bedroom plant through back wall (moved), bedroom + dining plants in front of curtains (removed), pillow sets 2x wider than the armchairs (scaled), bedroom shelf plant vs clock (moved up).
- Remaining audit hits are legitimate contact (bowl+fruit, chair tucked under table edge, items on beds/sofas).
- Coverage after pass: Bedroom 64%/100% (image-right slightly under the 70% target after removing the curtain-side plant), Dining 100%/91%, Living 100%/82%, Kitchen 73%/82%.
- Rule for later zones: derive chair fronts from the mesh (never hard-code a yaw), and run the overlap audit before rendering.

## Sofa fix (user: "the sofa is a bit odd")
- Root cause: the sofa FACED THE WALL (rolled back toward the room, seat against the wall) - same class of mistake as the chairs. Also it was too big (Sofa03 at 0.61x = 2.9 x 1.15 m; the pack is authored at 2x, correct scale is 0.5) and had a plain slab base.
- Replaced with Mega Pack Sofa16 at 0.5x (2.45 x 0.90 x 0.98 m, legs, two seat cushions, rolled arms). Facing set from the mesh (front derived from backrest side): now +Z (into the room). Bed02 verified facing +Z. Pillows sit on the cushions (cushion top 0.58 m; base frame top 0.40 m). Side table and its lamp moved 0.5 m closer.
- Rule: run the mesh-derived facing check on every seating piece and bed in every new zone.

## Zone_Foyer (2026-09-29) - relaxed clearance on purpose
- 4.0 x 5.5 x 3.2 m at x=80. clearSide 1.0, clearBack 1.0 (front 2.0 unchanged). Door: Back wall off-centre (x +0.75), closed with nappin (Prb)Door at 0.62x (2.1 x 1.27 m).
- nappin pieces are ~1.6x oversized (use 0.62x) and their long axis is Z with the front on +/-X: for a piece against a side wall use yaw 0/180, NOT 90 (that made pieces stick through walls). The door is the exception (thin in X, needs yaw 90 for a back wall).
- Facing had to be fixed visually for non-seating pieces: console drawers and mirror faced the wall (flipped 180).
- Props placed while a piece was still mis-oriented ended up on the floor inside the console; lamp + bowl re-seated on the 0.74 m top. Lesson: re-run surface placement after any re-orientation.
- Removed: vintage_suitcase (a 1.6 m stack), hanging fisherman's hat (floated at 0.5 m). Frame swapped (frame_02 artwork missing).
- Result: validate OK; coverage 82%/91% (image-right/-X, image-left/+X). Images: EnvTests/after/Foyer_daylight.png, Foyer_night.png.
- Indoor zones done: living room, bedroom, kitchen, dining room, foyer (bathroom dropped by the user). Remaining: office zones (3), outdoor zones (4), generator loop over zones/presets, bake (deferred by the user).

## Bake (2026-09-29, after Unity restart)
- Connection mess: the Hub had opened the OLD standalone project (~/Coding/GestureMotionPOC) and a stray NEW project (~/Coding/GestureMotionPOC/My project, created via -createproject). Two editors fought over the relay ports (9001/9002); MCP answered from the wrong one. Stray editor closed, its empty "My project" folder deleted (fresh template only; the old GestureMotionPOC project is untouched). Correct project launched by hand: Unity -projectPath <MagicMirror1>/unity/GestureMotionPOC.
- MCP link went stale afterwards (only /mcp refreshes it), so a file-triggered runner was added: Assets/Scripts/Editor/BakeOnRequest.cs. Write "bake" or "render" into <project>/Temp/bake_request.txt; progress goes to Temp/bake_status.txt (needs the Unity window brought to front once so it imports script changes: `open -a .../Unity.app`).
- Rosetta OK after restart: LightBaker runs (CPU lightmapper, IndirectOnly, 8 texels/unit). First bake took 12 s; outputs in Assets/Scenes/DatasetWorld/ (3 lightmaps, LightingData.asset, ReflectionProbe-0.exr).
- PROBLEM found by rendering Night/Evening after the first bake: I baked with flat ambient 0.6 + Daylight lights, and the ambient is baked into every lightmapped surface -> Night/Evening looked like day. Fix: bake a low neutral baseline (ambient 0.08, Overcast preset) so presets can still darken/colour the room. Images of the first (bad) bake: EnvTests/baked/ (overwritten by the re-render).
- Note: baked lighting affects only the static shell; props/character are lit by realtime lights + light probes.
- FINAL bake settings (bake #3): Lightmapping.Clear(), all zone lights at Overcast x0.15, flat ambient 0.03. Bake time ~8-12 s. Verified with dataset-camera renders (EnvTests/baked/<Zone>_<Daylight|Evening|Night>.png): Daylight bright, Evening warm orange, Night dark blue with lamps as the light sources. Bake #1 (ambient 0.6) made Night look like day; bake #2 (ambient 0.08, Overcast full) was still too bright at Night. Rule: bake dark, let presets add light.
- To re-bake after moving furniture: write "bake" into Temp/bake_request.txt (Unity window must be in front once), then "render" to regenerate the check images. Props are not static, so moving them does not need a re-bake; changing walls/floors/ceilings or zone sizes does.

## Characters in the rooms + email (2026-09-29)
- All 9 character FBX are Humanoid. Sizes were inconsistent: Remy 4.15 m (!), PeasantGirl 2.14, TheBoss 2.13 (stylised); others 1.86-1.97. Originals untouched: scaled PREFAB copies in Assets/Characters/Normalized/ (James 1.85, Kate 1.68, Leonard 1.80, Louise 1.65, Megan 1.70, PeasantGirl 1.65, Remy 1.80, Shannon 1.75, TheBoss 1.85). The generator's character list should use these prefabs. XBot stays 1.90.
- Renders: EnvTests/characters2/<Zone>_<Char>.png (50) and contact sheets EnvTests/characters_sheet_<Zone>_v2.png. Bake re-run with characters in the scene (10 s).
- PeasantGirl/TheBoss are stylised/cartoonish (hat, wide skirt): decide whether to keep them.
- Email to Prof. Kromer: user chose draft-only. Gmail draft action failed twice (auto-mode classifier error), so nothing was created; text saved in docs/email-draft-kromer.txt, images in EnvTests/email/*.jpg (attach by hand; the Gmail tool cannot take files this size).

## Office zones (2026-09-29)
- New reusable helper: Assets/Scripts/Editor/ZonePlace.cs (Put/PutWall/OnShelf/FaceTo/Slab/Audit/Check/Shot). Lessons in it: preserve prefab root rot/scale; mesh-derived facing; raycasts must start BELOW the ceiling slab (bug: a 3.4 m ray origin sat above a 3.2 m ceiling, so furniture landed on the roof and the rooms rendered empty).
- Zone_OpenOffice (7x7x3.5, door Front wall, at x=100): metal_office_desk workstations on the +X wall, stocked steel shelving on the -X wall (boards at 0.14/0.64/1.14/1.66/2.14 m), two-desk island + whiteboard + projector screen at the back, windows both sides. Floor: fleece carpet (busy plaid and rust-stained slate were rejected: they sit right behind the hands).
- Zone_MeetingRoom (5.5x6.5x3.2, door Right wall back corner, x=120): table + 6 chairs facing it, whiteboard, projector screen, china cabinet with TV, armchair + nesting table + floor lamp, window on the armchair wall.
- Zone_PrivateOffice (5.4x6.6x3.2, door Left wall back corner, x=140): desk + chair facing the visitor, three stocked bookcases, armchair + side table + lamp, curtained window, rug. The first 4.6 m width was too narrow for the 1.5 m side clearance, so it was widened to 5.4.
- Office pack assets are authored at odd scales (desk 5 m long, chair 3 m): always size by target height.

## Outdoor mode (Zone.outdoor)
- Zone.outdoor: open ground slab (W+60 x D+60, so no void at the frame edge), no walls/ceiling, directional "Sun@2" light (realtime soft shadows) whose elevation/yaw follow each preset (LightingPreset.sunElevation/sunYaw; Zone.SetSunAngles fills defaults). Fences/hedges/walls are placed as props (ZonePlace.Slab).
- Camera note: sky is never in frame (top edge ~13 deg below horizontal), so outdoor variety = ground materials + nearby props.
- Poly Haven outdoor batch downloaded (trees, shrubs, benches, lamps, fences, ground textures): Assets/Environments is now 2.1 GB (git/LFS decision pending).

## Remote pull + real472_pure_ik (2026-09-30)
- Remote: bhaskara@10.215.17.122 (k017-ws22, Linux, disk 96% full). Key auth works non-interactively (ssh -o BatchMode=yes). Read-only access only; nothing on the remote was changed.
- Pulled to ~/Downloads: real472_pure_ik.bvh (the file named in the request), ~/Downloads/motion_work/{test_swipe_right.bvh, bvh_export/ (289 BVH clips), convert.py, real472_pure_ik.fbx}, ~/Downloads/remote_10.215.17.122/Magicmirror3/ (notes + gesture_system results/logs/config; prof_email_draft.md, GESTURE_IMPROVEMENTS.md).
- MISTAKE: my first rsync used --include='*/' before --exclude, so rsync's first-match-wins let it walk into .git/.venv (8 GB) before I stopped it; the junk was deleted, rsync killed. Use explicit file lists (scp) or put excludes FIRST.
- Deliberately NOT pulled: 365-384 MB video-model checkpoints (best_model.pt, twostream_video.pt, twostream_fusion.pt), refs/ (81 MB), gesture_system/data (100 MB), kimodo *.npz, credential files, draxil/Sport/slur folders (unrelated).
- Motion set on the remote (bvh_export): 289 clips = 110 swipe_right, 99 swipe_left, 73 null, 7 constraint_target. Almost all 60 frames (2 s). Kimodo skeleton, 78 joints, cm units, hips ~0.99 m, head ~1.59 m.
- real472_pure_ik.bvh: only 30 frames (1 s). Right hand travels 0.53 m sideways but rises only 0.06 m above hip height (other swipe_right clips: +0.20 m). On Kate in the living room the swipe is barely visible (hand stays at 0.89-1.00 m, ~11% of frame width). Conversion path: Blender headless (blender_execute MCP addon was not running) -> FBX (Assets/Motion/real472_pure_ik.fbx), Humanoid import: avatar valid, 55 bones mapped, no missing required bones; Unity reports animation import warnings (not yet inspected). Blender timeline default was 250 frames: set frame range from the action before export, otherwise the FBX is padded with 7 s of standing still.
- Model results found (gesture_system): cross-validated balanced accuracy ~0.86-0.88 (scaling_results_v2: 0.872 at full data, plateau); landmark_best val balanced 0.883 (train acc 0.99 -> overfit gap); held-out session twostream landmark: acc 0.640, balanced 0.653, per class swipe_left 0.86, swipe_right 0.40, null 0.69. prof_email_draft.md: KiMoDo synthetic data raised held-out balanced accuracy 61.5% -> 66.2%. The gap between CV (~87%) and held-out session (~65%) points to domain shift, and swipe_right is the weak class; that is the case for varied rendered environments.
- The outdoor zones are parked at: assets downloaded (Assets/Environments now ~2.1 GB), Zone.outdoor mode + ZonePlace.Slab added and NOT yet compiled/tested, no outdoor zone built yet.

## Clip viewer (2026-09-30) - "just show me the clips in Unity"
- 291 clips converted BVH -> FBX headless with Blender (~27 s; ~/Downloads/motion_work/batch_convert.py, output ~/Downloads/motion_work/fbx). Copied to Assets/Motion/Clips (290) + Assets/Motion/real472_pure_ik.fbx. Assets/Scripts/Editor/MotionImportSettings.cs makes everything under Assets/Motion Humanoid with its own avatar. All 291: valid avatars, 0.97-1.97 s, humanMotion true, no missing required bones.
- Scene: Assets/Scenes/ClipViewer.unity = copy of DatasetWorld (baked lighting kept) + ClipViewer (Assets/Scripts/ClipViewer.cs) in the living room. Keys: Left/Right clip, Space pause, L loop, Up/Down speed, C dataset<->wide camera, N next character; on-screen buttons too. Rebuild with menu Tools > Build Clip Viewer Scene.
- Verified by actually entering Play mode from a script (Assets/Scripts/Editor/PlayTest.cs, trigger "playtest" in Temp/bake_request.txt) and saving frames (EnvTests/playtest/). Bugs found this way: (1) wide camera was outside the living room (front wall 2.2 m away) -> now raycasts to the wall; (2) clips created after graph.Play() are PAUSED -> the first clip played but every clip after an arrow-key press was frozen; fixed with playable.Play() + SetTime(0); (3) NOT a viewer bug: with Unity in the background the player loop does not tick (Time.frameCount stuck at 2), so the harness forces QueuePlayerLoopUpdate + runInBackground.
- Edit-mode sampling (PlayableGraph.Evaluate then Camera.Render) shows a stale T-pose; do not trust it, test in Play mode.
- Observed motion: swipe_right_92 raises the right hand to head height; swipe_left_13 sweeps the left hand across the face; null_1 idle; real472_pure_ik barely moves (hand stays near the hip).

## Full clip capture + review (2026-09-30)
- CaptureAll.cs (trigger "captureall"): Play mode, 8 frames per clip x 291 = 2328 frames + metrics.csv (hand/hip/foot positions, dataset-camera viewport of both hands) in EnvTests/capture/. ~2.5 min. Analysis: ~/Downloads/motion_work/analyze.py -> summary.json. Contact sheets: sheet_swipe_right/left/null/flagged.png; GIFs: EnvTests/capture/for_you/.
- Numbers: 115 swipe_right / 102 swipe_left / 73 null / 1 other. Right-swipes always move the RIGHT hand (115/115), left-swipes the LEFT hand (102/102). Hand stays inside the dataset-camera frame in 100% of frames for all clips. Median swipe: hand rises ~0.5 m above the hips, 1.5 m of hand travel, ~1 s. Hips never drift/bob. Direction (in the dataset image, while the hand is raised): 91/115 swipe_right clips travel image-left, 89/102 swipe_left travel image-right; 6 and 0 go the other way, rest ambiguous.
- Flagged: constraint_target_swipe_left_171_gen and _99_gen (hand never rises 15 cm above the hips; 99 also tiny motion), kimodo_null_4 (walks/steps, 1.09 m hand path) and kimodo_null_94 (0.9 m): null clips with big motion. real472_pure_ik: tiny motion (hand 0.15 above hips).
- Visual check (my own look at 24 clips): natural human arm motion, clean rise/peak/return, no popping, no foot sliding, no limb through body. Limitations: idle null clips are near-static; kimodo_null_8 is a leg-kick/step, kimodo_null_4 a step, i.e. not all nulls are "hands down".

## Quality pass (2026-09-30): "camera is low, models are low-res"
- Causes: (1) character textures capped at 2048 although 24 of 65 are 4096 -> raised to source size (Quality.cs, also trilinear + aniso 8 + HQ compression); (2) MSAA was OFF (URP asset PC_RPAsset m_MSAA=1) -> 4x, and all offscreen RenderTextures now request antiAliasing=4; forced anisotropic filtering in QualitySettings; (3) review captures were 400x520 crops from a 1.5 m-high wide camera -> now 900x1200 frames from 2.5 m up (same height as the dataset camera), FOV 48, closer, looking down at the body.
- The DATASET camera (2.5 m up, 1.0 m out, 75 deg) is unchanged: that geometry was specified earlier. If "low" meant the dataset camera, tell me the height you want.
- Everything re-rendered: capture (EnvTests/capture, old low-res kept in capture_v1_lowres), baked room images (EnvTests/baked, 8 zones x Daylight/Evening/Night; old in baked_v1), characters in rooms (EnvTests/characters2, 80 images), new sheets + GIFs. Metrics unchanged (same clips/poses).
- Generator note: BatchDatasetGenerator's RenderTexture still has no MSAA and is 1280x720; set antiAliasing=4 and consider 1920x1080 before batch rendering.
