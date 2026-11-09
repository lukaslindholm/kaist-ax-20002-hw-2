"""HW2 starter: hand-crafted image feature + softmax regression으로 제출 파일 만들기 (numpy/pandas만 사용)

이 baseline은 "stage 1~3의 이미지와 클래스 비율이 stage 4, 5에서도 그대로"라고 가정한다:
    phi(x) = 밝기 histogram, 밝기 통계, gradient 통계 (35차원)
    p(y | x) = softmax(W^T phi(x) + b),   stage 1~3을 하나로 합쳐 cross-entropy 최소화
여러분의 과제는 stage에 따라 무엇이 변하는지 진단하고, stage 4, 5에서도 버티는 classifier를 만드는 것이다.

폴더 구조:  data/train.csv, data/test.csv, data/images.npz, data/sample_submission.csv
실행:       python starter.py   ->  submission.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")
N_CLASSES = 6
SEED = 0


def load():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    z = np.load(DATA / "images.npz", allow_pickle=True)
    assert (z["id_train"] == train.image_id).all() and (z["id_test"] == test.image_id).all()
    Xtr = z["X_train"].astype(np.float64) / 255  # (n, 64, 64), [0, 1]
    Xte = z["X_test"].astype(np.float64) / 255
    return train, test, Xtr, Xte


def features(X):
    """X: (n, 64, 64) in [0, 1] -> (n, 35) feature matrix"""
    n = len(X)
    flat = X.reshape(n, -1)
    hist = np.stack([np.histogram(f, bins=16, range=(0, 1))[0] for f in flat]) / flat.shape[1]
    stats = np.column_stack([flat.mean(1), flat.std(1), np.percentile(flat, [5, 25, 50, 75, 95], axis=1).T])
    gx = np.abs(np.diff(X, axis=2)).reshape(n, -1)   # 가로 방향 밝기 변화
    gy = np.abs(np.diff(X, axis=1)).reshape(n, -1)   # 세로 방향 밝기 변화
    grad = np.column_stack([gx.mean(1), gy.mean(1), gx.std(1), gy.std(1), gx.mean(1) / (gy.mean(1) + 1e-6)])
    prof = np.column_stack([X.mean(axis=2).std(1), X.mean(axis=1).std(1)])  # 행/열 평균 profile의 변동
    lap = (X[:, 1:-1, 1:-1] * 4 - X[:, :-2, 1:-1] - X[:, 2:, 1:-1] - X[:, 1:-1, :-2] - X[:, 1:-1, 2:])
    sharp = np.column_stack([np.abs(lap).reshape(n, -1).mean(1), lap.reshape(n, -1).std(1)])
    return np.hstack([hist, stats, grad, prof, sharp])


def softmax(a):
    a = a - a.max(axis=1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(axis=1, keepdims=True)


def fit_softmax(X, y, lam=1e-3, lr=0.5, epochs=2000):
    """full-batch gradient descent on mean cross-entropy + (lam/2)||W||^2"""
    n, d = X.shape
    Y = np.eye(N_CLASSES)[y]
    W, b = np.zeros((d, N_CLASSES)), np.zeros(N_CLASSES)
    for _ in range(epochs):
        G = (softmax(X @ W + b) - Y) / n          # d loss / d logits
        W -= lr * (X.T @ G + lam * W)
        b -= lr * G.sum(axis=0)
    return W, b


def main():
    train, test, Xtr, Xte = load()
    Xtr, Xte = features(Xtr), features(Xte)
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-8   # feature별 표준화 (train 통계)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

    # 참고용 검증: stage 1, 2로 학습 -> stage 3에서 정확도
    m12 = train.stage.isin([1, 2]).to_numpy()
    W, b = fit_softmax(Xtr[m12], train.label.to_numpy()[m12])
    acc3 = (softmax(Xtr[~m12] @ W + b).argmax(1) == train.label.to_numpy()[~m12]).mean()
    print(f"train on stage 1-2, accuracy on stage 3: {acc3:.3f}")

    # 제출: stage 1~3 전체로 학습
    W, b = fit_softmax(Xtr, train.label.to_numpy())
    pred = softmax(Xte @ W + b).argmax(axis=1)
    sub = pd.DataFrame({"id": test.image_id, "label": pred})
    sub.to_csv("submission.csv", index=False)
    print("saved submission.csv", sub.shape, "predicted class counts:", np.bincount(pred, minlength=N_CLASSES))


if __name__ == "__main__":
    main()
