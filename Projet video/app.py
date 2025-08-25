from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import io, time, base64, re
from pathlib import Path
from PIL import Image
import torch
import numpy as np
from torchvision import models, transforms

# =======================
# Paths & Constantes
# =======================
BASE_DIR = Path(__file__).resolve().parent
PLACES_CHECKPOINT = (BASE_DIR / "resnet50_places365.pth")
if not PLACES_CHECKPOINT.exists():
    alt = BASE_DIR / "resnet50_places365.pth.tar"
    if alt.exists():
        PLACES_CHECKPOINT = alt
PLACES_LABELS_TXT = BASE_DIR / "categories_places365.txt"

# =======================
# Modèles (chargés sur CPU si nécessaire)
# =======================

# YOLO (détection objets)
try:
    from ultralytics import YOLO
    yolo_model = YOLO("yolov8n.pt")
    # Force CPU si pas de GPU
    if not torch.cuda.is_available():
        yolo_model.to("cpu")
except Exception:
    yolo_model = None

# YOLOv8 Pose (détection humaine + points clés)
try:
    from ultralytics import YOLO as YOLOPose
    yolo_pose = YOLOPose("yolov8n-pose.pt")
    if not torch.cuda.is_available():
        yolo_pose.to("cpu")
except Exception:
    yolo_pose = None

# Faster R-CNN ResNet50 (COCO) — on filtrera la classe "person"
try:
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
    FASTER_WEIGHTS = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    faster_person = fasterrcnn_resnet50_fpn(weights=FASTER_WEIGHTS)
    faster_person.eval()
except Exception:
    faster_person = None

# ResNet (ImageNet) pour top-k "détection" simple + labels via meta (pas besoin de fichier .json)
IMAGENET_WEIGHTS = models.ResNet50_Weights.IMAGENET1K_V2
resnet_imagenet = models.resnet50(weights=IMAGENET_WEIGHTS)
resnet_imagenet.eval()
IMAGENET_LABELS = IMAGENET_WEIGHTS.meta.get("categories", [])

preprocess_224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ResNet Places365 (classification de scène)
resnet_places = None
places_classes = []

def _load_places_labels(path: Path):
    """Lit categories_places365.txt en gérant plusieurs formats possibles."""
    if not path.exists():
        print(f"[Places365] Labels introuvables: {path}")
        return []
    cats = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # Formats possibles: "0 abbey" | "0  abbey" | "0 'abbey'"
            m = re.match(r"^\s*\d+\s+['\"]?([^'\"]+)['\"]?\s*$", s)
            if m:
                cats.append(m.group(1))
            else:
                # fallback: dernier token
                cats.append(s.split()[-1].strip("'\""))
    print(f"[Places365] {len(cats)} labels chargés depuis {path.name}")
    return cats

def _load_places_model(path: Path):
    """Charge un checkpoint entraîné sur CUDA en le **mappant sur CPU** si besoin."""
    if not path.exists():
        print(f"[Places365] Checkpoint introuvable: {path}")
        return None
    print(f"[Places365] Chargement checkpoint (CPU): {path.name}")
    m = models.resnet50(num_classes=365)
    ckpt = torch.load(str(path), map_location="cpu")  # ⬅️ important pour éviter l’erreur CUDA
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}  # DataParallel cleanup
    missing, unexpected = m.load_state_dict(state, strict=False)
    print(f"[Places365] Missing keys: {len(missing)} ; Unexpected keys: {len(unexpected)}")
    m.eval()
    return m

# Charge au démarrage
places_classes = _load_places_labels(PLACES_LABELS_TXT)
resnet_places = _load_places_model(PLACES_CHECKPOINT)

print(f"[Places365] Résumé -> ckpt_existe={PLACES_CHECKPOINT.exists()} ; "
      f"labels_existent={PLACES_LABELS_TXT.exists()} ; nb_labels={len(places_classes)} ; "
      f"model_loaded={'oui' if resnet_places is not None else 'non'}")

# =======================
# Flask
# =======================
app = Flask(__name__)
app.secret_key = "change_me_in_production"

# Utils
def b64_to_pil(b64img: str) -> Image.Image:
    # b64img = "data:image/png;base64,...."
    try:
        _, b64data = b64img.split(",", 1)
    except ValueError:
        b64data = b64img  # au cas où on enverrait du base64 "pur"
    img_bytes = base64.b64decode(b64data)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")

# Pages
@app.route("/")
def index():
    return redirect(url_for("consent"))

@app.route("/consent", methods=["GET", "POST"])
def consent():
    if request.method == "POST":
        accepted = request.form.get("accept_terms") == "on"
        if accepted:
            session["accepted_terms"] = True
            return redirect(url_for("benchmark"))
    return render_template("consent.html")

@app.route("/benchmark")
def benchmark():
    if not session.get("accepted_terms"):
        return redirect(url_for("consent"))
    # Injecte les 365 labels dans le template
    return render_template("benchmark.html", places_labels=places_classes)

# (Optionnel) exposer les labels via API
@app.get("/api/places_labels")
def api_places_labels():
    return jsonify({"ok": True, "labels": places_classes, "count": len(places_classes)})

# =======================
# APIs temps réel
# =======================
@app.post("/api/detect")
def api_detect():
    """
    JSON: { "image": "data:image/png;base64,...", "model": "yolo" | "resnet" }
    """
    data = request.get_json(silent=True) or {}
    if "image" not in data:
        return jsonify({"ok": False, "error": "No image provided"}), 400
    model_choice = (data.get("model") or "yolo").lower()

    pil_img = b64_to_pil(data["image"])
    np_bgr = np.array(pil_img)[:, :, ::-1].copy()  # RGB->BGR (YOLO accepte aussi RGB, mais BGR OK)

    if model_choice == "yolo":
        if yolo_model is None:
            return jsonify({"ok": False, "error": "YOLO model not available"}), 500
        t0 = time.time()
        results = yolo_model(np_bgr, imgsz=640, conf=0.25, iou=0.5, verbose=False)
        latency_ms = (time.time() - t0) * 1000.0

        dets = []
        names = yolo_model.model.names if hasattr(yolo_model, "model") else {}
        b = results[0].boxes
        if b is not None and b.xyxy is not None:
            for i in range(len(b.xyxy)):
                xyxy = b.xyxy[i].tolist()
                cls_id = int(b.cls[i].item()) if b.cls is not None else -1
                conf = float(b.conf[i].item()) if b.conf is not None else None
                label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                dets.append({"bbox": xyxy, "label": label, "cls_id": cls_id, "conf": conf})

        return jsonify({
            "ok": True,
            "model": "yolo",
            "latency_ms": round(latency_ms, 2),
            "objects": [d["label"] for d in dets],
            "detections": dets
        })

    elif model_choice == "resnet":
        tensor = preprocess_224(pil_img).unsqueeze(0)
        t0 = time.time()
        with torch.no_grad():
            preds = resnet_imagenet(tensor)
        latency_ms = (time.time() - t0) * 1000.0

        probs = torch.softmax(preds, dim=1)
        topk = torch.topk(probs, k=5)
        indices = topk.indices[0].tolist()
        values = [float(v) for v in topk.values[0].tolist()]
        names = [IMAGENET_LABELS[i] if 0 <= i < len(IMAGENET_LABELS) else f"cls_{i}" for i in indices]

        return jsonify({
            "ok": True,
            "model": "resnet",
            "latency_ms": round(latency_ms, 2),
            "objects": names,
            "topk": [{"cls_id": int(i), "label": n, "prob": v} for i, n, v in zip(indices, names, values)]
        })

    else:
        return jsonify({"ok": False, "error": "Unknown model"}), 400

@app.post("/api/scene")
def api_scene():
    """
    JSON: { "image": "data:image/png;base64,..."}
    """
    data = request.get_json(silent=True) or {}
    if "image" not in data:
        return jsonify({"ok": False, "error": "No image provided"}), 400

    # Diagnostics clairs si modèles/labels manquants
    reasons = []
    if resnet_places is None:
        reasons.append(f"checkpoint non chargé ({PLACES_CHECKPOINT.name})")
    if not places_classes:
        reasons.append(f"labels non chargés ({PLACES_LABELS_TXT.name})")
    if reasons:
        return jsonify({"ok": False, "error": "Places365 indisponible : " + ' ; '.join(reasons)}), 500

    pil_img = b64_to_pil(data["image"])
    tensor = preprocess_224(pil_img).unsqueeze(0)

    t0 = time.time()
    with torch.no_grad():
        preds = resnet_places(tensor)
    latency_ms = (time.time() - t0) * 1000.0

    cls = int(preds.argmax(1).item())
    scene_label = places_classes[cls] if 0 <= cls < len(places_classes) else None

    return jsonify({
        "ok": True,
        "latency_ms": round(latency_ms, 2),
        "cls_id": cls,        # index numérique
        "scene": scene_label  # libellé texte si dispo
    })

@app.post("/api/human")
def api_human():
    """
    JSON: { "image": "data:image/png;base64,...", "model": "yolov8-pose" | "fasterrcnn" }
    Retour:
      - persons: [{bbox:[x1,y1,x2,y2], score:float, keypoints:[[x,y,conf],...]}]
      - latency_ms, model
    """
    data = request.get_json(silent=True) or {}
    if "image" not in data:
        return jsonify({"ok": False, "error": "No image provided"}), 400
    model_choice = (data.get("model") or "yolov8-pose").lower()

    pil_img = b64_to_pil(data["image"])
    np_bgr = np.array(pil_img)[:, :, ::-1].copy()

    if model_choice == "yolov8-pose":
        if yolo_pose is None:
            return jsonify({"ok": False, "error": "YOLOv8 Pose not available"}), 500
        t0 = time.time()
        results = yolo_pose(np_bgr, imgsz=640, conf=0.35, iou=0.5, verbose=False)
        latency_ms = (time.time() - t0) * 1000.0

        persons = []
        r = results[0]
        boxes = getattr(r, "boxes", None)
        kps = getattr(r, "keypoints", None)

        # keypoints: xy [N,17,2] ; conf [N,17]
        if kps is not None and kps.xy is not None:
            xy = kps.xy.tolist()
            conf = (kps.conf.tolist() if getattr(kps, "conf", None) is not None else None)
        else:
            xy, conf = [], []

        N = len(xy)
        for i in range(N):
            bbox = boxes.xyxy[i].tolist() if boxes is not None else None
            score = float(boxes.conf[i].item()) if (boxes is not None and boxes.conf is not None) else None
            kplist = []
            for j in range(len(xy[i])):
                kp_conf = conf[i][j] if conf else None
                kplist.append([
                    float(xy[i][j][0]),
                    float(xy[i][j][1]),
                    float(kp_conf) if kp_conf is not None else None
                ])
            persons.append({"bbox": bbox, "score": score, "keypoints": kplist})

        return jsonify({
            "ok": True,
            "model": "yolov8-pose",
            "latency_ms": round(latency_ms, 2),
            "persons": persons
        })

    elif model_choice == "fasterrcnn":
        if faster_person is None:
            return jsonify({"ok": False, "error": "Faster R-CNN not available"}), 500
        t0 = time.time()
        with torch.no_grad():
            # Faster R-CNN attend des tensors [C,H,W] en [0..1]
            img_t = transforms.functional.to_tensor(pil_img)
            out = faster_person([img_t])[0]
        latency_ms = (time.time() - t0) * 1000.0

        boxes = out["boxes"].cpu().numpy().tolist()
        labels = out["labels"].cpu().numpy().tolist()
        scores = out["scores"].cpu().numpy().tolist()

        persons = []
        for b, l, s in zip(boxes, labels, scores):
            # COCO: classe 1 = person
            if int(l) == 1 and float(s) >= 0.5:
                persons.append({"bbox": b, "score": float(s), "keypoints": []})

        return jsonify({
            "ok": True,
            "model": "fasterrcnn",
            "latency_ms": round(latency_ms, 2),
            "persons": persons
        })

    else:
        return jsonify({"ok": False, "error": "Unknown model"}), 400

# =======================
# Main
# =======================
if __name__ == "__main__":
    # app.run(host="0.0.0.0", port=5000, debug=True)  # décommente si tu veux exposer sur ton LAN
    app.run(debug=True)
