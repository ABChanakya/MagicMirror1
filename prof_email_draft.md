Subject: Update — KiMoDo synthetic data is measurably helping

Hi Mr. Kromer,

Quick update: I tested the models you suggested (LTX-2.5 IC-LoRA and KiMoDo).
The video/frame-based approaches — video-to-video, image-to-video, and
frame-by-frame image-to-image — all plateaued around 25-30% on my automated
pose validator, and the visual quality wasn't usable even on the ones that
passed.

KiMoDo is different: it generates 3D motion directly rather than video, so I
can skip rendering entirely and train straight on the motion data. After I
found and fixed a bug in my own validator's left/right scoring, generated
clips hit a 65% pass rate. I merged a first batch of validated synthetic
clips into training and re-tested on a held-out recording session (never
seen during training): balanced accuracy went from 61.5% to 66.2%, with the
weakest class (swipe-right) improving the most. I'm scaling up generation
now to see how far this trend goes.

MediaPipe's real-time landmark extraction already works well, so the open
question is purely whether KiMoDo gives me enough synthetic variety. If it
plateaus below what I need, building a Unity environment to render the
motions in varied settings is my fallback — but I'd rather not add that
complexity unless the numbers say I need to.

Best,
Chanakya
