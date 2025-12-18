import os, json
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import os

import joblib
import torch
import torch.nn as nn
from skimage.feature import hog

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

load_dotenv()

TOKEN = os.getenv("TELEGRAM_TOKEN")

if TOKEN is None:
    raise RuntimeError("TELEGRAM_TOKEN not set")

class SimpleCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), 
            nn.Flatten(),
            nn.Linear(32*7*7, 128), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes)
        )
    def forward(self, x):
        return self.net(x)


with open("label_map.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)

n_classes = len(label_map)

svm = joblib.load("svm_hog.pkl")
knn_best = joblib.load("knn_best.pkl")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleCNN(n_classes).to(device)
model.load_state_dict(torch.load("cnn.pt", map_location=device))
model.eval()


def preprocess_pil_to_28x28_gray(pil_img: Image.Image) -> np.ndarray:
    #  28x28
    img = pil_img.convert("L").resize((28, 28), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0  # [0..1]
    return arr  # (28,28)


def hog_features(img_28x28: np.ndarray) -> np.ndarray:
    f = hog(
        img_28x28,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        orientations=9,
        feature_vector=True
    )
    return f[None, :]  # (1, D)


def cnn_tensor(img_28x28: np.ndarray) -> torch.Tensor:
    x = torch.tensor(img_28x28[None, None, :, :], dtype=torch.float32)  # (1,1,28,28)
    x = x.repeat(1, 3, 1, 1)  # (1,3,28,28)
    return x.to(device)


def predict_all(img_28x28: np.ndarray):
    # SVM (HOG)
    p1 = int(svm.predict(hog_features(img_28x28))[0])

    # kNN (flatten)
    flat = img_28x28.reshape(1, -1)
    p2 = int(knn_best.predict(flat)[0])

    # CNN
    with torch.no_grad():
        logits = model(cnn_tensor(img_28x28))
        p3 = int(torch.argmax(logits, dim=1).cpu().item())

    return p1, p2, p3


# ---------- 4) handlers ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Пришли мне картинку клетки (или любое изображение), "
        "а я выдам предсказания 3 моделей: HOG+SVM, kNN, CNN.\n"
        "Команды: /start"
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]
    file = await photo.get_file()
    data = await file.download_as_bytearray()

    pil = Image.open(io.BytesIO(data))
    img = preprocess_pil_to_28x28_gray(pil)

    p1, p2, p3 = predict_all(img)

    msg = (
        f"HOG + SVM: {p1} | {label_map[str(p1)]}\n"
        f"kNN: {p2} | {label_map[str(p2)]}\n"
        f"CNN: {p3} | {label_map[str(p3)]}"
    )
    await update.message.reply_text(msg)

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    file = await doc.get_file()
    data = await file.download_as_bytearray()

    pil = Image.open(io.BytesIO(data))
    img = preprocess_pil_to_28x28_gray(pil)

    p1, p2, p3 = predict_all(img)

    msg = (
        f"HOG + SVM: {p1} | {label_map[str(p1)]}\n"
        f"kNN: {p2} | {label_map[str(p2)]}\n"
        f"CNN: {p3} | {label_map[str(p3)]}"
    )
    await update.message.reply_text(msg)


def main():
    token = "8309616126:AAGLseyUu3Ld39Q-a5dH8GkPmEpnpHrWD0Q"

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.IMAGE, handle_document))

    print("Bot running... Device:", device)
    app.run_polling()


if __name__ == "__main__":
    import io
    main()
