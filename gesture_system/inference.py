"""
Real-time gesture inference with sliding window over webcam feed.

Usage
-----
python inference.py --model-path checkpoints/best_model.pt
python inference.py --model-path checkpoints/best_model.pt --landmark-only
python inference.py --model-path checkpoints/best_model.pt --camera 1

Controls
--------
q  — quit
s  — toggle stats display
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import yaml

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False
    print("[inference] Warning: mediapipe not installed — landmarks will be zeros.")


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Colours (BGR for OpenCV)
_COLOUR_GREEN  = (0,   220,  0)
_COLOUR_YELLOW = (0,   220, 220)
_COLOUR_RED    = (0,   0,   220)
_COLOUR_WHITE  = (255, 255, 255)
_COLOUR_BLACK  = (0,   0,   0)
_COLOUR_GRAY   = (128, 128, 128)

_CLASS_COLOURS = [
    (255, 80,  80),   # swipe_left  — blue-ish
    (80,  255, 80),   # swipe_right — green
    (80,  80,  255),  # swipe_up    — red
    (255, 255, 80),   # swipe_down  — yellow
    (180, 180, 180),  # void        — gray
]


# ──────────────────────────────────────────────────────────────────────────────
# MediaPipe helpers
# ──────────────────────────────────────────────────────────────────────────────

def _init_mediapipe():
    if not _MP_AVAILABLE:
        return None
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def _extract_landmarks(hands_model, frame_rgb: np.ndarray) -> Tuple[np.ndarray, Optional[object]]:
    """
    Run MediaPipe on one frame.

    Returns
    -------
    landmarks_63 : np.ndarray (63,) — zeros if no hand detected
    raw_result   : mediapipe result object or None (for drawing)
    """
    if hands_model is None:
        return np.zeros(63, dtype=np.float32), None

    try:
        results = hands_model.process(frame_rgb)
    except Exception:
        return np.zeros(63, dtype=np.float32), None

    if not results.multi_hand_landmarks:
        return np.zeros(63, dtype=np.float32), None

    lm = results.multi_hand_landmarks[0].landmark
    coords = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)  # (21, 3)
    return coords.flatten(), results


def _draw_landmarks(frame_bgr: np.ndarray, results) -> np.ndarray:
    """Draw MediaPipe hand landmarks on the frame."""
    if results is None or not results.multi_hand_landmarks:
        return frame_bgr
    mp_drawing = mp.solutions.drawing_utils
    mp_hands   = mp.solutions.hands
    for hand_lm in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame_bgr,
            hand_lm,
            mp_hands.HAND_CONNECTIONS,
        )
    return frame_bgr


# ──────────────────────────────────────────────────────────────────────────────
# Frame pre-processing
# ──────────────────────────────────────────────────────────────────────────────

def _preprocess_frame(frame_rgb: np.ndarray, image_size: int = 224) -> torch.Tensor:
    """
    Convert HxWx3 uint8 RGB → (3, image_size, image_size) float32 tensor,
    ImageNet-normalised.
    """
    t = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0  # (3, H, W)
    t = TF.resize(t, [image_size, image_size], antialias=True)
    t = (t - _IMAGENET_MEAN) / _IMAGENET_STD
    return t


# ──────────────────────────────────────────────────────────────────────────────
# Landmark normalisation (same as preprocessing, applied live)
# ──────────────────────────────────────────────────────────────────────────────

def _normalise_landmark_window(lm_window: np.ndarray) -> np.ndarray:
    """
    lm_window: (30, 63)
    Returns normalised (30, 63): wrist-centred + scale-normalised.
    Zero frames (no hand) are left as zero.
    """
    lm_3d = lm_window.reshape(30, 21, 3).copy()

    # Skip normalisation for all-zero frames (no hand)
    non_zero_mask = (np.abs(lm_3d).sum(axis=(1, 2)) > 1e-8)  # (30,)

    if non_zero_mask.any():
        # Subtract wrist from non-zero frames
        wrist = lm_3d[:, 0:1, :]
        lm_3d[non_zero_mask] -= wrist[non_zero_mask]

        # Scale by overall hand extent
        xy = lm_3d[non_zero_mask, :, :2].reshape(-1, 2)
        span_x = xy[:, 0].max() - xy[:, 0].min()
        span_y = xy[:, 1].max() - xy[:, 1].min()
        scale  = max(float(max(span_x, span_y)), 1e-6)
        lm_3d[non_zero_mask] /= scale

    return lm_3d.reshape(30, 63).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Overlay drawing
# ──────────────────────────────────────────────────────────────────────────────

def _draw_overlay(
    frame_bgr: np.ndarray,
    pred_class: int,
    confidence: float,
    probs: np.ndarray,
    classes: list,
    cooldown_active: bool,
    last_fired: Optional[str],
) -> np.ndarray:
    H, W = frame_bgr.shape[:2]

    # ── Status indicator circle (top-right) ───────────────────────────────
    circle_colour = _COLOUR_YELLOW if cooldown_active else _COLOUR_GREEN
    cv2.circle(frame_bgr, (W - 30, 30), 18, circle_colour, -1)
    status_text = "COOLDOWN" if cooldown_active else "READY"
    cv2.putText(frame_bgr, status_text, (W - 120, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, circle_colour, 1, cv2.LINE_AA)

    # ── Top-left: current prediction + confidence ─────────────────────────
    pred_name = classes[pred_class] if pred_class < len(classes) else "unknown"
    pred_text = f"{pred_name}: {confidence*100:.1f}%"
    # Shadow
    cv2.putText(frame_bgr, pred_text, (11, 41),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, _COLOUR_BLACK, 3, cv2.LINE_AA)
    cv2.putText(frame_bgr, pred_text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, _COLOUR_WHITE, 2, cv2.LINE_AA)

    # Last fired gesture
    if last_fired:
        fired_text = f"Fired: {last_fired}"
        cv2.putText(frame_bgr, fired_text, (11, 71),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, _COLOUR_BLACK, 3, cv2.LINE_AA)
        cv2.putText(frame_bgr, fired_text, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, _COLOUR_GREEN, 2, cv2.LINE_AA)

    # ── Bottom: confidence bars for all classes ───────────────────────────
    bar_height = 20
    bar_max_w  = 200
    margin     = 10
    bar_y_base = H - (len(classes) * (bar_height + 4)) - margin

    for i, (cls_name, prob) in enumerate(zip(classes, probs)):
        y = bar_y_base + i * (bar_height + 4)
        bar_w = int(prob * bar_max_w)
        colour = _CLASS_COLOURS[i] if i < len(_CLASS_COLOURS) else _COLOUR_GRAY

        # Background
        cv2.rectangle(frame_bgr, (margin, y), (margin + bar_max_w, y + bar_height),
                      _COLOUR_BLACK, -1)
        # Filled bar
        if bar_w > 0:
            cv2.rectangle(frame_bgr, (margin, y), (margin + bar_w, y + bar_height),
                          colour, -1)
        # Border
        cv2.rectangle(frame_bgr, (margin, y), (margin + bar_max_w, y + bar_height),
                      _COLOUR_GRAY, 1)
        # Label
        label = f"{cls_name[:12]:12s} {prob*100:5.1f}%"
        cv2.putText(frame_bgr, label, (margin + bar_max_w + 5, y + bar_height - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _COLOUR_WHITE, 1, cv2.LINE_AA)

    return frame_bgr


# ──────────────────────────────────────────────────────────────────────────────
# Gesture callback
# ──────────────────────────────────────────────────────────────────────────────

def fire_gesture(class_name: str):
    """Called when a confirmed gesture is detected."""
    print(f"\n>>> GESTURE FIRED: {class_name.upper()} <<<\n")
    # In a full MagicMirror integration, emit an HTTP request or socket event here.


# ──────────────────────────────────────────────────────────────────────────────
# Main inference loop
# ──────────────────────────────────────────────────────────────────────────────

def run_inference(args):
    # ── Config ────────────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    inf_cfg   = cfg['inference']
    data_cfg  = cfg['data']
    model_cfg = cfg['model']

    classes             = data_cfg['classes']
    num_classes         = model_cfg['num_classes']
    image_size          = data_cfg['image_size']
    num_frames          = data_cfg['num_frames']
    conf_threshold      = inf_cfg['confidence_threshold']
    consec_required     = inf_cfg['consecutive_predictions_required']
    infer_every_n       = inf_cfg['inference_every_n_frames']
    COOLDOWN            = inf_cfg['cooldown_frames']

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[inference] Device: {device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    from models.fusion_head import GestureRecognitionModel

    model = GestureRecognitionModel(
        landmark_d_model=model_cfg['landmark_d_model'],
        landmark_nhead=model_cfg['landmark_nhead'],
        landmark_num_layers=model_cfg['landmark_num_layers'],
        landmark_dim_feedforward=model_cfg['landmark_dim_feedforward'],
        landmark_dropout=model_cfg['landmark_dropout'],
        fusion_hidden=model_cfg['fusion_hidden'],
        fusion_dropout=model_cfg['fusion_dropout'],
        num_classes=num_classes,
        pretrained_swin=False,  # weights loaded from checkpoint
    ).to(device)

    if args.model_path:
        model_path = Path(args.model_path)
        if model_path.exists():
            ckpt = torch.load(str(model_path), map_location=device)
            state = ckpt.get('model_state_dict', ckpt)
            model.load_state_dict(state)
            print(f"[inference] Loaded model from {model_path}")
        else:
            print(f"[inference] Warning: {model_path} not found — using random weights")
    else:
        print("[inference] No model path provided — using random weights")

    model.eval()

    # ── MediaPipe ─────────────────────────────────────────────────────────────
    hands = _init_mediapipe()

    # ── Webcam ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[inference] Error: cannot open camera {args.camera}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    print(f"[inference] Camera {args.camera} opened. Press 'q' to quit.")

    # ── Sliding window buffers ────────────────────────────────────────────────
    # Each entry: (landmarks_63: np.ndarray, frame_tensor: torch.Tensor)
    buffer: deque = deque(maxlen=num_frames)

    prediction_history: deque = deque(maxlen=consec_required)
    cooldown_frames = 0
    frame_count     = 0
    last_fired      = None

    # Display state
    current_pred       = 4  # default: void
    current_confidence = 0.0
    current_probs      = np.zeros(num_classes)
    show_stats         = True

    while True:
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            print("[inference] Camera read failed — retrying...")
            time.sleep(0.05)
            continue

        frame_count += 1
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # ── MediaPipe landmark extraction ─────────────────────────────────
        try:
            lm_63, mp_results = _extract_landmarks(hands, frame_rgb)
        except Exception as e:
            print(f"[inference] MediaPipe error: {e}")
            lm_63       = np.zeros(63, dtype=np.float32)
            mp_results  = None

        # ── Pre-process frame ─────────────────────────────────────────────
        frame_tensor = _preprocess_frame(frame_rgb, image_size=image_size)  # (3, H, W)

        # ── Append to sliding window ──────────────────────────────────────
        buffer.append((lm_63, frame_tensor))

        # ── Run inference every N frames once buffer is full ──────────────
        if len(buffer) == num_frames and frame_count % infer_every_n == 0:

            # Build tensors
            lm_array   = np.stack([b[0] for b in buffer], axis=0)   # (30, 63)
            lm_norm    = _normalise_landmark_window(lm_array)        # (30, 63)
            frames_list = [b[1] for b in buffer]                    # list of (3,H,W)

            lm_t = torch.from_numpy(lm_norm).unsqueeze(0).to(device)      # (1,30,63)
            fr_t = torch.stack(frames_list, dim=1).unsqueeze(0).to(device) # (1,3,30,H,W)

            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                    enabled=(device.type == 'cuda')):
                    if args.landmark_only:
                        logits = model.forward_landmark_only(lm_t)
                    else:
                        logits = model(lm_t, fr_t)

            probs_t = F.softmax(logits.float(), dim=-1).squeeze(0)  # (num_classes,)
            pred_class  = int(probs_t.argmax().item())
            confidence  = float(probs_t[pred_class].item())
            probs_np    = probs_t.cpu().numpy()

            current_pred       = pred_class
            current_confidence = confidence
            current_probs      = probs_np

            prediction_history.append((pred_class, confidence))

            # ── Gesture firing logic ──────────────────────────────────────
            if cooldown_frames == 0 and len(prediction_history) == consec_required:
                preds       = [p for p, _ in prediction_history]
                confs       = [c for _, c in prediction_history]
                all_agree   = len(set(preds)) == 1
                not_void    = preds[0] != classes.index('null') if 'null' in classes else preds[0] != 4
                high_conf   = min(confs) >= conf_threshold

                if all_agree and not_void and high_conf:
                    class_name = classes[preds[0]]
                    fire_gesture(class_name)
                    last_fired      = class_name
                    cooldown_frames = COOLDOWN
                    prediction_history.clear()

        # ── Decrement cooldown ────────────────────────────────────────────
        if cooldown_frames > 0:
            cooldown_frames -= 1

        # ── Draw landmarks ────────────────────────────────────────────────
        frame_bgr = _draw_landmarks(frame_bgr, mp_results)

        # ── Draw HUD overlay ──────────────────────────────────────────────
        if show_stats:
            frame_bgr = _draw_overlay(
                frame_bgr,
                pred_class=current_pred,
                confidence=current_confidence,
                probs=current_probs,
                classes=classes,
                cooldown_active=(cooldown_frames > 0),
                last_fired=last_fired,
            )

        # ── Frame counter ─────────────────────────────────────────────────
        cv2.putText(
            frame_bgr, f"frame {frame_count}",
            (frame_bgr.shape[1] - 120, frame_bgr.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, _COLOUR_GRAY, 1, cv2.LINE_AA,
        )

        cv2.imshow("Gesture Recognition", frame_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            show_stats = not show_stats

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    if hands is not None:
        hands.close()
    print("[inference] Exited.")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Real-time gesture inference")
    parser.add_argument(
        '--model-path', type=str, default='checkpoints/best_model.pt',
        help='Path to model checkpoint (.pt)',
    )
    parser.add_argument(
        '--config', type=str, default='config.yaml',
        help='Path to config.yaml',
    )
    parser.add_argument(
        '--camera', type=int, default=0,
        help='Camera index (default: 0)',
    )
    parser.add_argument(
        '--landmark-only', action='store_true',
        help='Use forward_landmark_only() — no Video Swin branch',
    )
    args = parser.parse_args()

    run_inference(args)


if __name__ == '__main__':
    main()
