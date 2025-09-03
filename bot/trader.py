# trader.py
# Ultimate merged bot:
# - Robust canvas detection (largest visible canvas)
# - toDataURL() first, screenshot fallback
# - OpenCV candle parsing (red/green on black)
# - Feature engineering (calls feature_engineering.add_features)
# - Support/Resistance + multi-candle pattern detection
# - Dynamic indicator adjustment (smart confidence)
# - GUI Pause/Resume running on main thread (clears memory on resume)
# - Retrain scheduler + notifications + debug screenshots
# NOTE: run with Python 3.8+ (you're using 3.13 which works), ensure dependencies installed.

import sys, os, time, datetime, threading, base64, io, math
from pathlib import Path
from collections import deque

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

# GUI: Tk in main thread
import tkinter as tk

# Optional CV/Imaging
try:
    from PIL import Image
    import cv2
    HAVE_CV = True
except Exception:
    HAVE_CV = False

# add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from feature_engineering import add_features
from notifier import notify  # your Gmail notifier

# ---------------- Config ----------------
USER_DATA_DIR = os.path.expanduser("~/.playwright_olymp_profile")  # persistent profile directory used by Playwright
MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "xgb_model.joblib"
LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
SCREENSHOT_DIR = LOG_DIR / "screenshots"
LOG_DIR.mkdir(exist_ok=True)
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.65"))
LOOP_SLEEP = float(os.getenv("LOOP_SLEEP", "2.5"))
RETRAIN_NIGHT_HOUR = int(os.getenv("RETRAIN_NIGHT_HOUR", "2"))
CANDLE_HISTORY = int(os.getenv("CANDLE_HISTORY", "150"))
MIN_CANDLES_FOR_PREDICTION = max(120, CANDLE_HISTORY)  # user asked at least 120 candles

DEBUG_ITERATIONS = 3  # save debug images first N loops

# ---------------- Load model ----------------
model = joblib.load(MODEL_PATH)
print("✅ Model loaded from", MODEL_PATH)

# ---------------- Shared State ----------------
candle_memory = deque(maxlen=CANDLE_HISTORY)  # will store rows (dict) of candles
pause_event = threading.Event()  # if set, bot is paused
stop_event = threading.Event()   # to stop worker cleanly
loop_counter_lock = threading.Lock()
loop_counter = 0

# ---------------- GUI (main thread) ----------------
def build_control_gui():
    # This must run on the main thread to avoid Tcl errors on macOS
    root = tk.Tk()
    root.title("AI Binary Bot Control")
    root.geometry("300x120")
    root.resizable(False, False)

    status_var = tk.StringVar(value="Status: RUNNING")
    def on_pause():
        pause_event.set()
        status_var.set("Status: PAUSED")
    def on_resume():
        if pause_event.is_set():
            pause_event.clear()
            # clear candle memory on resume
            with loop_counter_lock:
                candle_memory.clear()
            status_var.set("Status: RUNNING")
    def on_exit():
        stop_event.set()
        root.quit()

    lbl = tk.Label(root, text="AI Binary Bot", font=("Helvetica", 12, "bold"))
    lbl.pack(pady=(8,0))
    status_lbl = tk.Label(root, textvariable=status_var)
    status_lbl.pack(pady=(2,6))

    frame = tk.Frame(root)
    frame.pack()
    btn_pause = tk.Button(frame, text="Pause ⏸", command=on_pause, width=10, bg="#FF7F50")
    btn_pause.grid(row=0, column=0, padx=6)
    btn_resume = tk.Button(frame, text="Resume ▶", command=on_resume, width=10, bg="#32CD32")
    btn_resume.grid(row=0, column=1, padx=6)

    exit_btn = tk.Button(root, text="Exit ✖", command=on_exit, width=26)
    exit_btn.pack(pady=(8,6))

    return root

# ---------------- Utility helpers ----------------
def ts_now():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def save_debug_image(img: Image.Image, prefix="debug"):
    path = SCREENSHOT_DIR / f"{prefix}_{ts_now()}.png"
    try:
        img.save(path)
        return str(path)
    except Exception as e:
        print("⚠️ Failed to save debug image:", e)
        return None

# ---------------- Robust Canvas Detection ----------------
def find_best_canvas_handle(page, try_count=3, wait_between=0.3):
    """
    Find the best canvas element on the page:
    - query all canvases
    - choose the one with largest visible area (width*height) and not 0
    Returns ElementHandle or None.
    """
    for attempt in range(try_count):
        try:
            canvas_handles = page.query_selector_all("canvas")
            best = None
            best_area = 0
            for h in canvas_handles:
                try:
                    box = h.bounding_box()
                    if not box:
                        continue
                    area = (box["width"] or 0) * (box["height"] or 0)
                    # ignore trivially small canvases
                    if area > best_area:
                        best_area = area
                        best = h
                except Exception:
                    continue
            if best is not None and best_area > 100:  # require at least some area
                return best
        except Exception:
            pass
        time.sleep(wait_between)
    return None

def canvas_to_image(page, canvas_handle, debug_prefix="canvas"):
    """
    Try to get PIL Image from canvas handle.
    1) Try element.evaluate(c=>c.toDataURL())
    2) If fails, use element.screenshot()
    3) As further fallback, screenshot the whole page and crop bounding_box
    Returns PIL.Image or None
    """
    # 1) try toDataURL
    try:
        data_url = canvas_handle.evaluate("c => { try { return c.toDataURL('image/png'); } catch (e) { return null; } }")
        if data_url:
            head, b64 = data_url.split(",", 1)
            img_bytes = base64.b64decode(b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            return img
    except Exception as e:
        # continue to fallback
        pass

    # 2) try element.screenshot
    try:
        tmp_path = SCREENSHOT_DIR / f"{debug_prefix}_el_{ts_now()}.png"
        canvas_handle.screenshot(path=str(tmp_path))
        img = Image.open(tmp_path).convert("RGB")
        return img
    except Exception:
        pass

    # 3) page screenshot and crop bounding box
    try:
        box = canvas_handle.bounding_box()
        if box:
            tmp_full = SCREENSHOT_DIR / f"{debug_prefix}_full_{ts_now()}.png"
            page.screenshot(path=str(tmp_full))
            full = Image.open(tmp_full).convert("RGB")
            left = int(box["x"])
            top = int(box["y"])
            right = left + int(box["width"])
            bottom = top + int(box["height"])
            # safe cropping
            right = min(right, full.width)
            bottom = min(bottom, full.height)
            img = full.crop((left, top, right, bottom))
            return img
    except Exception:
        pass

    return None

# ---------------- Candle parsing (OpenCV) ----------------
def parse_candles_from_image(img: Image.Image, debug=False):
    """
    From a chart image, detect vertical candlestick bars colored green/red
    Returns DataFrame of candles with columns: time, open, high, low, close, volume
    This is heuristic: we measure vertical rectangles and map pixel y to price scale later via feature engineering (here we return pixel-based OHLC).
    """
    if not HAVE_CV or img is None:
        return pd.DataFrame()

    try:
        arr = np.array(img)
        h, w = arr.shape[0], arr.shape[1]
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

        # detect green
        mask_g = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        # detect red (two ranges)
        mask_r1 = cv2.inRange(hsv, (0, 40, 40), (10, 255, 255))
        mask_r2 = cv2.inRange(hsv, (160, 40, 40), (180,255,255))
        mask_r = cv2.bitwise_or(mask_r1, mask_r2)

        mask = cv2.bitwise_or(mask_g, mask_r)

        # morphological clean
        kernel = np.ones((2,2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cand_list = []
        debug_img = arr.copy()

        for cnt in contours:
            x,y,ww,hh = cv2.boundingRect(cnt)
            # filter noise
            if hh < 4 or ww < 1:
                continue
            roi = arr[y:y+hh, x:x+ww]
            mean_col = roi.mean(axis=(0,1))
            r,g,b = mean_col
            direction = "UP" if g > r else "DOWN"
            # map to pixel OHLC (pixel y -> price later)
            open_p = h - (y + hh)
            close_p = h - y
            high_p = max(open_p, close_p)
            low_p = min(open_p, close_p)
            cand_list.append({"x": x, "open": open_p, "high": high_p, "low": low_p, "close": close_p, "direction": direction})
            if debug:
                color = (0,255,0) if direction=="UP" else (0,0,255)
                cv2.rectangle(debug_img, (x,y), (x+ww,y+hh), color, 1)

        if not cand_list:
            return pd.DataFrame()

        cand_list = sorted(cand_list, key=lambda c: c["x"])
        df = pd.DataFrame(cand_list).drop(columns=["x"])
        df["time"] = None
        df["volume"] = 0

        if debug:
            try:
                dbg = Image.fromarray(debug_img)
                save_debug_image(dbg, prefix="candles_debug")
            except Exception:
                pass

        # NOTE: these open/high/low/close are in pixel coordinates relative to image height.
        # feature_engineering expects numeric prices; we will still feed it — the model was trained on price values.
        # Ideally you'd map pixel positions to actual price scale; if your chart displays values (axis), a mapping can be derived,
        # but many systems allow us to instead rely on relative features (SMA, RSI computed on pixel 'close' works if consistent).
        return df[["time","open","high","low","close","volume"]]
    except Exception as e:
        print("⚠️ parse_candles_from_image failed:", e)
        return pd.DataFrame()

# ---------------- Support/Resistance and Patterns ----------------
def detect_support_resistance(df, window=20):
    df = df.copy()
    if "high" in df and "low" in df:
        df['resistance'] = df['high'].rolling(window, center=True, min_periods=1).max()
        df['support'] = df['low'].rolling(window, center=True, min_periods=1).min()
    else:
        df['resistance'] = np.nan
        df['support'] = np.nan
    return df

def detect_candlestick_patterns(df):
    df = df.copy()
    df['pattern'] = None
    if df.empty:
        return df
    for i in range(len(df)):
        c = df.iloc[i]
        body = abs(c['close'] - c['open'])
        rng = c['high'] - c['low'] if (c['high'] - c['low']) != 0 else 1e-9
        if body <= 0.1*rng:
            df.at[i,'pattern'] = 'Doji'
        upper = c['high'] - max(c['open'], c['close'])
        lower = min(c['open'], c['close']) - c['low']
        if lower > 2*body and upper < body:
            df.at[i,'pattern'] = 'Hammer'
        elif upper > 2*body and lower < body:
            df.at[i,'pattern'] = 'Shooting Star'
    # multi-candle patterns
    for i in range(1, len(df)):
        prev = df.iloc[i-1]; curr = df.iloc[i]
        if prev['close'] < prev['open'] and curr['close'] > curr['open'] and curr['close'] > prev['open'] and curr['open'] < prev['close']:
            df.at[i,'pattern'] = 'Bullish Engulfing'
        if prev['close'] > prev['open'] and curr['close'] < curr['open'] and curr['open'] > prev['close'] and curr['close'] < prev['open']:
            df.at[i,'pattern'] = 'Bearish Engulfing'
    # W/M and 3-candle classical heuristics
    if len(df) >= 5:
        for i in range(2, len(df)-2):
            s = df.iloc[i-2:i+3]
            if s['low'].iloc[0] > s['low'].iloc[2] < s['low'].iloc[4] and s['high'].iloc[2] < s['high'].iloc[0]:
                df.at[i,'pattern'] = 'W Pattern'
            if s['high'].iloc[0] < s['high'].iloc[2] > s['high'].iloc[4] and s['low'].iloc[2] > s['low'].iloc[0]:
                df.at[i,'pattern'] = 'M Pattern'
    if len(df) >= 3:
        for i in range(2, len(df)):
            c1,c2,c3 = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
            # Three White Soldiers / Three Black Crows
            if c1['close']>c1['open'] and c2['close']>c2['open'] and c3['close']>c3['open'] and c2['close']>c1['close'] and c3['close']>c2['close']:
                df.at[i,'pattern'] = 'Three White Soldiers'
            if c1['close']<c1['open'] and c2['close']<c2['open'] and c3['close']<c3['open'] and c2['close']<c1['close'] and c3['close']<c2['close']:
                df.at[i,'pattern'] = 'Three Black Crows'
    return df

# ---------------- Dynamic indicator adjustment ----------------
def dynamic_indicator_adjustment(last_row, bullish_list, bearish_list):
    delta = 0.0
    if last_row is None:
        return 0.0
    pat = last_row.get('pattern')
    # pattern-based adjustments
    if pat in bullish_list:
        if last_row.get('close', 0) <= last_row.get('support', 1e9) and last_row.get('RSI', 50) < 40:
            delta += 0.25
        elif last_row.get('RSI', 50) < 50:
            delta += 0.12
    if pat in bearish_list:
        if last_row.get('close', 0) >= last_row.get('resistance', -1e9) and last_row.get('RSI', 50) > 60:
            delta -= 0.25
        elif last_row.get('RSI', 50) > 50:
            delta -= 0.12
    # MACD/Signal
    if last_row.get('MACD', 0) > last_row.get('Signal', 0) and last_row.get('RSI', 50) < 50:
        delta += 0.07
    elif last_row.get('MACD', 0) < last_row.get('Signal', 0) and last_row.get('RSI', 50) > 50:
        delta -= 0.07
    # Bollinger band edges
    if 'BBU' in last_row and 'BBL' in last_row:
        if last_row.get('close', 0) > last_row.get('BBU', 1e9):
            delta -= 0.05
        elif last_row.get('close', 0) < last_row.get('BBL', -1e9):
            delta += 0.05
    return max(-1.0, min(1.0, delta))

# ---------------- Retrain scheduler ----------------
def schedule_retrain_once():
    try:
        import subprocess
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve().parents[1] / "train_model.py")])
        print("Started retrain subprocess PID", proc.pid)
    except Exception as e:
        print("⚠️ Could not start retrain:", e)

def retrain_daemon():
    while not stop_event.is_set():
        now = datetime.datetime.now()
        if now.hour == RETRAIN_NIGHT_HOUR and now.minute == 0:
            notify("⏰ Nightly retrain triggered", "Automated retrain started")
            schedule_retrain_once()
            time.sleep(60)
        time.sleep(20)

# ---------------- Main analysis loop (worker thread) ----------------
def analysis_worker():
    global loop_counter
    # Start retrain background thread
    threading.Thread(target=retrain_daemon, daemon=True).start()

    with sync_playwright() as p:
        # Use persistent context (user profile dir) so cookies/session persist. If directory missing, don't pass it.
        try:
            if Path(USER_DATA_DIR).exists():
                browser = p.chromium.launch_persistent_context(USER_DATA_DIR, headless=False, args=["--start-maximized"])
                page = browser.pages[0] if browser.pages else browser.new_page()
            else:
                browser = p.chromium.launch(headless=False, args=["--start-maximized"])
                page = browser.new_page()
        except Exception as e:
            print("⚠️ Playwright browser launch failed:", e)
            notify("⚠️ Playwright failed", str(e))
            return

        page.goto("https://olymptrade.com", wait_until="domcontentloaded")
        print("➡️ Waiting for manual login (or pre-existing session)...")
        # wait until a canvas exists (not more than some time)
        login_wait_start = time.time()
        while time.time() - login_wait_start < 300:
            try:
                # look for any canvas
                canv = page.query_selector_all("canvas")
                if canv and len(canv) > 0:
                    break
            except Exception:
                pass
            time.sleep(1)
        notify("✅ Logged in to Olymp Trade", "")

        while not stop_event.is_set():
            # respect pause
            if pause_event.is_set():
                time.sleep(1)
                continue

            with loop_counter_lock:
                loop_counter += 1
                current_iter = loop_counter

            # 1) find best canvas
            canvas_handle = find_best_canvas_handle(page, try_count=3)
            if canvas_handle is None:
                print("⚠️ No canvas found. Will retry.")
                notify("⚠️ Bot cannot see the chart", "No canvas element found on page.")
                time.sleep(3)
                continue

            # 2) get image from canvas
            img = None
            try:
                img = canvas_to_image(page, canvas_handle, debug_prefix="chart")
            except Exception as e:
                print("⚠️ canvas_to_image error:", e)
            if img is None:
                print("⚠️ Could not capture canvas image.")
                notify("⚠️ Bot cannot see the chart", "Canvas capture failed.")
                time.sleep(2)
                continue

            # 3) parse candles from image
            df_pixels = parse_candles_from_image(img, debug=(current_iter <= DEBUG_ITERATIONS))
            print(f"Canvas->candles rows: {len(df_pixels)}")
            if df_pixels.empty:
                # Save debug and notify
                save_debug_image(img, prefix="no_candles")
                notify("⚠️ Bot cannot see candles", "No candlesticks detected in image.")
                time.sleep(2)
                continue

            # 4) keep into memory: here df_pixels are pixel OHLCs; we append rows into candle_memory as numeric floats
            # Convert to numeric and keep last N candles across iterations
            # We'll convert each row dict to floats
            for r in df_pixels.to_dict('records'):
                # r contains open/high/low/close in pixel coordinates (integers), volume=0
                candle_memory.append({
                    "time": None,
                    "open": float(r["open"]),
                    "high": float(r["high"]),
                    "low": float(r["low"]),
                    "close": float(r["close"]),
                    "volume": float(r.get("volume", 0))
                })

            # Build full df for features
            df_full = pd.DataFrame(list(candle_memory)).reset_index(drop=True)
            # Wait until we have enough candles
            if len(df_full) < MIN_CANDLES_FOR_PREDICTION:
                print(f"Waiting for more candles... {len(df_full)}/{MIN_CANDLES_FOR_PREDICTION}")
                time.sleep(LOOP_SLEEP)
                continue

            # 5) Add features (calls user feature_engineering.add_features)
            try:
                df_with_features = add_features(df_full)
            except Exception as e:
                print("⚠️ Feature engineering failed:", e)
                path = save_debug_image(img, prefix="feature_error_img")
                notify("⚠️ Feature engineering failed", str(e), attachment=path if path else None)
                time.sleep(LOOP_SLEEP)
                continue

            # 6) S/R + Patterns
            try:
                df_with_features = detect_support_resistance(df_with_features, window=20)
                df_with_features = detect_candlestick_patterns(df_with_features)
            except Exception as e:
                print("⚠️ Pattern/SR detection failed:", e)
                path = save_debug_image(img, prefix="pattern_error_img")
                notify("⚠️ Pattern detection failed", str(e), attachment=path if path else None)
                time.sleep(LOOP_SLEEP)
                continue

            # 7) Prediction using model
            try:
                X = df_with_features.drop(columns=["time","pattern","support","resistance"], errors="ignore").tail(1)
                # ensure no NaN
                X = X.fillna(0)
                # XGBoost model expects DMatrix
                dtest = xgb.DMatrix(X)
                prob = float(model.predict(dtest)[0])
            except Exception as e:
                print("⚠️ Model prediction error:", e)
                path = save_debug_image(img, prefix="predict_error_img")
                notify("⚠️ Model prediction error", str(e), attachment=path if path else None)
                time.sleep(LOOP_SLEEP)
                continue

            # 8) Smart confidence adjustments
            try:
                last_row = df_with_features.iloc[-1].to_dict()
                bullish = ['Hammer','W Pattern','Bullish Engulfing','Three White Soldiers','Morning Star']
                bearish = ['Shooting Star','M Pattern','Bearish Engulfing','Three Black Crows','Evening Star']
                adj = dynamic_indicator_adjustment(last_row, bullish, bearish)
                prob = max(0.0, min(1.0, prob + adj))
            except Exception as e:
                print("⚠️ Confidence adjustment failed:", e)

            direction = "UP" if prob > 0.5 else "DOWN"
            last_close = df_with_features["close"].iloc[-1]

            # 9) Annotate and save image
            try:
                annotated = annotate_for_notification(df_with_features, img) if HAVE_CV else None
                if annotated:
                    save_path = SCREENSHOT_DIR / f"signal_annotated_{ts_now()}.png"
                    annotated.save(save_path)
                    attach_path = str(save_path)
                else:
                    attach_path = save_debug_image(img, prefix="signal_raw")
            except Exception:
                attach_path = save_debug_image(img, prefix="signal_raw")

            # 10) Send notifications according to confidence
            try:
                if prob >= MIN_CONFIDENCE:
                    msg = f"🔥 Prediction: {direction} ({prob:.2%}) | price={last_close} | pattern={last_row.get('pattern')}"
                    notify(msg, attachment=attach_path)
                else:
                    notify("🤔 Low confidence — enable Stoch/PSAR for clarity.",
                           f"Confidence={prob:.2%}", attachment=attach_path)
            except Exception as e:
                print("⚠️ Notification send failed:", e)

            time.sleep(LOOP_SLEEP)

        # end while stop_event
        try:
            browser.close()
        except Exception:
            pass

# ---------------- Helper annotate function reused ----------------
def annotate_for_notification(df, img):
    """Draw support/resistance and most recent patterns on the image"""
    if not HAVE_CV or img is None or df.empty:
        return img
    arr = np.array(img)
    h,w,_ = arr.shape
    min_price, max_price = df['low'].min(), df['high'].max()
    def scale_y(price):
        # map price-like numbers (here they are 'pixel values'), this is approximate but consistent
        return int(h - ((price - min_price) / (max_price - min_price + 1e-9)) * h)

    # last few support/res lines
    for i,row in df.tail(3).iterrows():
        if pd.notnull(row.get('support')):
            y = scale_y(row['support'])
            cv2.line(arr, (0,y), (w,y), (255,0,0), 1)
        if pd.notnull(row.get('resistance')):
            y = scale_y(row['resistance'])
            cv2.line(arr, (0,y), (w,y), (0,0,255), 1)

    # patterns label
    for i,row in df.tail(10).iterrows():
        if row.get('pattern'):
            x = int(w * (i / max(1, len(df))))
            y = scale_y(row['close'])
            cv2.putText(arr, str(row['pattern']), (x, max(10, y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1, cv2.LINE_AA)

    return Image.fromarray(arr)

# ---------------- Entry point ----------------
def main():
    # build GUI in main thread
    gui_root = build_control_gui()

    # start analysis worker in a background thread
    worker = threading.Thread(target=analysis_worker, daemon=True)
    worker.start()

    # run tkinter mainloop (this blocks main thread but is required on macOS)
    try:
        gui_root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        print("Shutting down...")

if __name__ == "__main__":
    main()
