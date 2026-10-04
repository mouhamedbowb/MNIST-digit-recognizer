# MNIST Digit Recognizer

Draw a digit with your mouse or finger and a small convolutional neural network guesses it, with a probability for each of the ten digits. The model is trained with PyTorch and then runs **entirely in the browser**, so the app deploys as a static-style Next.js site with no Python server.

Based on the lecture "CNN to classify MNIST digits" from Mike X Cohen's course *A deep understanding of deep learning*.

## How it works

| Part | What it does |
| --- | --- |
| `backend/model.py` | CNN architecture: 2 conv layers (10 and 20 filters, 5x5), max pooling, ReLU, then 500 -> 50 -> 10 dense layers (30,840 parameters) |
| `backend/train.py` | Trains on MNIST with Adam and random rotation, scale and shift augmentation |
| `backend/export_weights.py` | Writes the trained weights to `web/public/model.json` |
| `web/lib/cnn.ts` | The same network written in TypeScript, so predictions need no server |
| `web/components/DrawCanvas.tsx` | Drawing canvas that crops, scales and centers the digit the way MNIST images are prepared |

## Run it locally

### 1. Train the model (Python 3.10+)

Download `mnist_train.csv` (60,000 rows: label, then 784 pixel values).

    cd backend
    pip install -r requirements.txt
    python train.py path/to/mnist_train.csv
    python export_weights.py

This creates `mnist_cnn.pt` and `web/public/model.json`. `web/public/model.json` is committed to the repository, which is what lets Vercel serve the model. Re-run both scripts and commit the new file whenever you retrain.

### 2. Start the web app (Node 18+)

    cd web
    npm install
    npm run dev

Open http://localhost:3000.

## Deploy to Vercel (free Hobby plan)

1. Push this repository to GitHub.
2. On vercel.com choose **Add New > Project** and import the repository.
3. Set **Root Directory** to `web`. Vercel detects Next.js on its own.
4. Click **Deploy**.

## Tips and limits

- Draw the digit upright, large, and in one or two strokes. CNNs are not rotation invariant, so a digit turned 90 degrees will be misread.
- The model always spreads 100% across the ten digits, so it can answer confidently on shapes that are not digits.
- Test accuracy on MNIST is around 98-99%. Mouse-drawn digits look different from MNIST digits, so real-world accuracy is lower.
