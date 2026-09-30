#!/usr/bin/env python3
"""Download the CC0 Poly Haven models/textures used by the dataset scene (about 2 GB) into
unity/GestureMotionPOC/Assets/Environments/PolyHaven{,Tex}/.  Safe to re-run: existing files are skipped.
The .meta files are already in git, so Unity re-links every reference to the same GUIDs.
Licence: CC0 (https://polyhaven.com/license). Author credits: unity/GestureMotionPOC/Assets/Environments/CREDITS.md
"""
import json, os, sys, urllib.request

MODELS=['Chandelier_01', 'Shelf_01', 'Television_01', 'WoodenTable_03', 'alarm_clock_01', 'antique_ceramic_vase_01', 'bananas', 'binder_notebook', 'book_encyclopedia_set_01', 'boombox', 'brass_candleholders', 'brass_goblets', 'brass_pan_01', 'brass_pot_01', 'brass_pot_02', 'brass_vase_01', 'carved_wooden_plate', 'ceramic_pot', 'ceramic_vase_01', 'ceramic_vase_02', 'ceramic_vase_03', 'ceramic_vase_04', 'chess_set', 'classic_laptop', 'clipboard', 'coffee_table_round_01', 'decorative_book_set_01', 'desk_lamp_arm_01', 'dining_table', 'drawer_cabinet', 'dustpan', 'fancy_picture_frame_01', 'fir_tree_01', 'fire_alarm', 'fire_hydrant', 'fishermans_hat', 'food_apple_01', 'food_avocado_01', 'food_kiwi_01', 'food_lime_01', 'garden_gnome', 'garden_hose_wall_mounted_01', 'garden_sprinkler_01', 'grass_medium_01', 'grass_medium_02', 'hamburger_buns', 'hanging_picture_frame_01', 'hanging_picture_frame_02', 'hanging_picture_frame_03', 'industrial_pastic_container', 'island_tree_01', 'island_tree_02', 'island_tree_03', 'jacaranda_tree', 'large_iron_gate', 'long_life_food', 'mantel_clock_01', 'metal_office_desk', 'metal_stool_01', 'metal_stool_02', 'metal_trash_can', 'modern_ceiling_lamp_01', 'modern_coffee_table_01', 'modular_chainlink_fence', 'modular_street_seating', 'multi_cleaner_bottle', 'office_notepads', 'outdoor_table_chair_set_01', 'painted_wooden_bench', 'pine_tree_01', 'planter_box_01', 'planter_box_02', 'planter_box_03', 'planter_pot_clay', 'plastic_bottle_gallon', 'plastic_thermos', 'postcard_set_01', 'pot_enamel_01', 'potted_plant_01', 'potted_plant_02', 'potted_plant_04', 'projector_screen', 'rubber_boots', 'russian_food_cans_01', 'shrub_01', 'shrub_02', 'shrub_03', 'shrub_04', 'shrub_sorrel_01', 'side_table_01', 'standing_chalkboard_01', 'standing_picture_frame_01', 'standing_picture_frame_02', 'stationery_supplies', 'steel_frame_shelves_01', 'steel_frame_shelves_02', 'street_lamp_01', 'street_lamp_02', 'sweet_potato', 'tea_set_01', 'television_02', 'throw_pillows_01', 'trashbag', 'tree_small_02', 'tree_stump_01', 'vintage_cabinet_01', 'vintage_electric_kettle', 'vintage_oil_lamp', 'vintage_suitcase', 'wall_clock', 'wicker_basket_01', 'wicker_basket_02', 'wine_bottles_01', 'wooden_bookshelf_worn', 'wooden_bowl_01', 'wooden_bowl_02', 'wooden_broom', 'wooden_bucket_01', 'wooden_candlestick', 'wooden_display_shelves_01', 'wooden_ladder_02', 'wooden_lantern_01']
TEXTURES=['asphalt_04', 'brick_wall_003', 'brown_leather', 'clay_roof_tiles', 'concrete_floor_painted', 'concrete_pavement_02', 'curly_teddy_natural', 'dark_wood', 'dirty_carpet', 'fabric_pattern_05', 'fabric_pattern_07', 'floral_jacquard', 'forest_ground_04', 'gingham_check', 'grass_path_2', 'gravel_concrete_02', 'grey_cartago_01', 'herringbone_parquet', 'hessian_380', 'interior_tiles', 'knitted_fleece', 'laminate_floor_02', 'leafy_grass', 'long_white_tiles', 'marble_01', 'painted_plaster_wall', 'patterned_terracotta_tiling', 'plank_flooring_02', 'poly_wool_herringbone', 'quatrefoil_jacquard_fabric', 'raked_dirt', 'rectangular_parquet', 'rectangular_paving', 'red_brick_03', 'rough_linen', 'scuba_suede', 'sparse_grass', 'square_brick_paving', 'velour_velvet', 'white_stucco', 'wood_floor_deck', 'wool_boucle']
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "unity", "GestureMotionPOC", "Assets", "Environments")

def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=120).read()

def model(i):
    f = json.loads(get(f"https://api.polyhaven.com/files/{i}"))["fbx"]["1k"]["fbx"]
    d = os.path.join(ROOT, "PolyHaven", i); os.makedirs(os.path.join(d, "textures"), exist_ok=True)
    for rel, url in [(i + "_1k.fbx", f["url"])] + [(r, m["url"]) for r, m in f.get("include", {}).items()]:
        p = os.path.join(d, rel); os.makedirs(os.path.dirname(p), exist_ok=True)
        if not os.path.exists(p): open(p, "wb").write(get(url))

def texture(i):
    files = json.loads(get(f"https://api.polyhaven.com/files/{i}"))
    d = os.path.join(ROOT, "PolyHavenTex", i); os.makedirs(d, exist_ok=True)
    for key in ("Diffuse", "nor_gl", "Rough"):
        if key in files and "1k" in files[key]:
            fmt = "jpg" if "jpg" in files[key]["1k"] else next(iter(files[key]["1k"]))
            url = files[key]["1k"][fmt]["url"]; p = os.path.join(d, os.path.basename(url))
            if not os.path.exists(p): open(p, "wb").write(get(url))

if __name__ == "__main__":
    bad = []
    for kind, ids, fn in (("model", MODELS, model), ("texture", TEXTURES, texture)):
        for n, i in enumerate(ids, 1):
            try: fn(i); print(f"[{kind} {n}/{len(ids)}] {i}")
            except Exception as e: bad.append((i, str(e))); print("FAILED", i, e, file=sys.stderr)
    print("done;", len(bad), "failures", bad[:5])
