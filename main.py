import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import queue
import os
import time
from datetime import datetime
import cv2
from PIL import Image, ImageTk
import csv
import re

import sys
from utils import resource_path, validate_rtsp_url, create_directory
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS   # PyInstaller temp folder
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# =========================
# OCR (Doctr)
# =========================
class DoctrOCR:
    def __init__(self):
        try:
            from doctr.io import DocumentFile
            from doctr.models import ocr_predictor
            self.DocumentFile = DocumentFile
            self.model = ocr_predictor(pretrained=True)
            self.available = True
            print("OCR ready")
        except Exception as e:
            print("OCR unavailable:", e)
            self.available = False

    def extract_numbers(self, img_path):
        if not self.available:
            return []
        doc = self.DocumentFile.from_images(img_path)
        result = self.model(doc)
        text = result.render()
        return re.findall(r"\b\d{5,9}\b", text)

# CONFIG
# =========================
SNAPSHOT_DIR = os.path.join(os.getcwd(), "snapshots")  # writable
CAMERA_CSV = resource_path("cameras.csv")
LOG_CSV = os.path.join(os.getcwd(), "detected_log.csv")

BOX_SIZE = 120
MAX_BOXES = 9
MOTION_THRESHOLD = 500
SNAP_COOLDOWN = 2
GALLERY_COLS = 6

# =========================
# CAMERA THREAD
# =========================
class CameraWorker(threading.Thread):
    def __init__(self, cam_name, rtsp_url, ui_queue):
        super().__init__(daemon=True)
        self.cam_name = cam_name
        self.rtsp_url = rtsp_url
        self.ui_queue = ui_queue
        self.running = True
        self.last_snap = 0
        self.ocr = DoctrOCR()

        self.out_dir = os.path.join(SNAPSHOT_DIR, cam_name)
        os.makedirs(self.out_dir, exist_ok=True)

    def stop(self):
        self.running = False

    def run(self):
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            print(f"[{self.cam_name}] RTSP open failed")
            return

        ret, prev = cap.read()
        if not ret:
            cap.release()
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(1)
                continue

            gray1 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray1, gray2)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            motion = any(cv2.contourArea(c) > MOTION_THRESHOLD for c in cnts)

            if motion and time.time() - self.last_snap > SNAP_COOLDOWN:
                ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                tmp_path = os.path.join(self.out_dir, f"tmp_{ts}.jpg")
                cv2.imwrite(tmp_path, frame)

                numbers = self.ocr.extract_numbers(tmp_path)

                if numbers:
                    num = numbers[0]
                    final_path = os.path.join(
                        self.out_dir, f"{ts}_{self.cam_name}_{num}.jpg"
                    )
                    os.rename(tmp_path, final_path)

                    detect_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.ui_queue.put(
                        ("snapshot", self.cam_name, final_path, num, detect_time)
                    )
                    self.last_snap = time.time()
                else:
                    os.remove(tmp_path)

            prev = frame
            time.sleep(0.05)

        cap.release()
        print(f"[{self.cam_name}] stopped")

# =========================
# GUI
# =========================
class RTSPGui:
    def __init__(self, root):
        self.root = root
        self.root.title("RTSP OCR Snapshot Monitor")

        self.ui_queue = queue.Queue()
        self.workers = {}
        self.snap_boxes = {}
        self.snap_labels = {}
        self.images = {}
        self.snap_paths = {}
        self.detect_log = {}

        self.build_ui()
        self.load_cameras_from_csv()
        self.root.after(200, self.process_queue)

    # -------------------------
    def build_ui(self):
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=10, pady=5)

        ttk.Button(top, text="Start", command=self.start_all).pack(side="left")
        ttk.Button(top, text="Stop", command=self.stop_all).pack(side="left", padx=5)
        ttk.Button(top, text="View Log", command=self.open_log_window).pack(side="right", padx=5)

        self.canvas = tk.Canvas(self.root)
        self.scroll = ttk.Scrollbar(self.root, command=self.canvas.yview)
        self.container = ttk.Frame(self.canvas)

        self.container.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.container, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scroll.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scroll.pack(side="right", fill="y")

    # -------------------------
    def load_cameras_from_csv(self):
        if not os.path.exists(CAMERA_CSV):
            messagebox.showerror("Error", "cameras.csv not found")
            return

        with open(CAMERA_CSV, newline="") as f:
            for cam, rtsp in csv.reader(f):
                cam = cam.strip()
                self.detect_log[cam] = []
                self.add_camera_row(cam)
                self.workers[cam] = None
                self.workers[f"{cam}_url"] = rtsp.strip()

    # -------------------------
    def add_camera_row(self, cam):
        row = ttk.LabelFrame(self.container, text=cam)
        row.pack(fill="x", padx=10, pady=5)

        box_frame = ttk.Frame(row)
        box_frame.pack(side="left", padx=5)

        boxes, labels = [], []

        for _ in range(MAX_BOXES):
            col = ttk.Frame(box_frame)
            col.pack(side="left", padx=3)

            img_lbl = ttk.Label(col, relief="solid", width=BOX_SIZE // 10)
            img_lbl.pack()

            txt_lbl = ttk.Label(col, text="-", foreground="blue")
            txt_lbl.pack()

            boxes.append(img_lbl)
            labels.append(txt_lbl)

        ttk.Button(row, text="+", command=lambda c=cam: self.open_gallery(c)).pack(side="right")

        self.snap_boxes[cam] = boxes
        self.snap_labels[cam] = labels

    # -------------------------
    def start_all(self):
        for cam in self.detect_log.keys():
            worker = self.workers.get(cam)
            if isinstance(worker, CameraWorker) and worker.is_alive():
                continue
            worker = CameraWorker(cam, self.workers[f"{cam}_url"], self.ui_queue)
            self.workers[cam] = worker
            worker.start()

    def stop_all(self):
        for cam, worker in self.workers.items():
            if isinstance(worker, CameraWorker):
                worker.stop()
                self.workers[cam] = None

    # -------------------------
    def process_queue(self):
        try:
            while True:
                msg = self.ui_queue.get_nowait()
                if msg[0] == "snapshot":
                    _, cam, path, num, t = msg
                    self.update_snapshot(cam, path, num)
                    self.detect_log[cam].append((num, t))
        except queue.Empty:
            pass
        self.root.after(200, self.process_queue)

    # -------------------------
    def update_snapshot(self, cam, path, num):
        img = Image.open(path).resize((BOX_SIZE, BOX_SIZE))
        tkimg = ImageTk.PhotoImage(img)

        self.images.setdefault(cam, []).append(tkimg)
        self.snap_paths.setdefault(cam, []).append(path)

        self.images[cam] = self.images[cam][-MAX_BOXES:]
        self.snap_paths[cam] = self.snap_paths[cam][-MAX_BOXES:]

        for i, im in enumerate(self.images[cam]):
            lbl = self.snap_boxes[cam][i]
            lbl.configure(image=im)
            lbl.image = im
            self.snap_labels[cam][i].configure(text=num)

    # -------------------------
    def open_gallery(self, cam):
        win = tk.Toplevel(self.root)
        win.title(f"Snapshots - {cam}")

        folder = os.path.join(SNAPSHOT_DIR, cam)
        if not os.path.exists(folder):
            return

        # -------------------------
        # Search bar
        search_var = tk.StringVar()
        search_frame = ttk.Frame(win)
        search_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(search_frame, text="Search Snapshot:").pack(side="left")
        search_entry = ttk.Entry(search_frame, textvariable=search_var)
        search_entry.pack(side="left", padx=5)
        ttk.Button(search_frame, text="Search", command=lambda: update_gallery(search_var.get())).pack(side="left", padx=5)

        # -------------------------
        result_frame = ttk.Frame(win)
        result_frame.pack(fill="both", expand=True, padx=10, pady=10)

        canvas = tk.Canvas(result_frame)
        scroll = ttk.Scrollbar(result_frame, command=canvas.yview)
        container = ttk.Frame(canvas)

        container.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=container, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)

        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        # -------------------------
        # Load images
        all_files = sorted(os.listdir(folder), reverse=True)
        images = []

        def update_gallery(term=""):
            nonlocal images
            for widget in container.winfo_children():
                widget.destroy()
            images = []
            row_idx, col_idx = 0, 0

            for fname in all_files:
                if term.lower() not in fname.lower():
                    continue
                path = os.path.join(folder, fname)
                img = Image.open(path).resize((200, 200))
                tkimg = ImageTk.PhotoImage(img)
                images.append(tkimg)

                frame = ttk.Frame(container, relief="ridge", padding=5)
                frame.grid(row=row_idx, column=col_idx, padx=5, pady=5)
                ttk.Label(frame, image=tkimg).pack()
                ttk.Label(frame, text=fname, wraplength=200).pack(pady=4)

                col_idx += 1
                if col_idx >= GALLERY_COLS:
                    col_idx = 0
                    row_idx += 1

            win.images = images  # keep reference

        # Initial load
        update_gallery()

    # -------------------------
    def open_log_window(self):
        # Auto-save CSV
        with open(LOG_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Camera", "Number", "Detected Time"])
            for cam, entries in self.detect_log.items():
                for num, t in entries:
                    writer.writerow([cam, num, t])

        win = tk.Toplevel(self.root)
        win.title("Detected Numbers Log")

        # Search
        search_var = tk.StringVar()
        ttk.Label(win, text="Search:").pack(anchor="w", padx=10, pady=(10, 0))
        ttk.Entry(win, textvariable=search_var).pack(fill="x", padx=10)

        # Download button
        ttk.Button(win, text="Download Log", command=self.download_log).pack(anchor="e", padx=10, pady=5)

        tree = ttk.Treeview(win, columns=("cam", "number", "time"), show="headings")
        tree.heading("cam", text="Camera")
        tree.heading("number", text="Number")
        tree.heading("time", text="Detected Time")
        tree.pack(fill="both", expand=True, padx=10, pady=10)

        all_rows = []
        for cam, entries in self.detect_log.items():
            for num, t in entries:
                all_rows.append((cam, num, t))
                tree.insert("", "end", values=(cam, num, t))

        def filter_log(*_):
            term = search_var.get().lower()
            tree.delete(*tree.get_children())
            for row in all_rows:
                if term in " ".join(row).lower():
                    tree.insert("", "end", values=row)

        search_var.trace_add("write", filter_log)

    # -------------------------
    def download_log(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
            title="Save Log As"
        )
        if not file_path:
            return

        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Camera", "Number", "Detected Time"])
            for cam, entries in self.detect_log.items():
                for num, t in entries:
                    writer.writerow([cam, num, t])

        messagebox.showinfo("Download Complete", f"Log saved to:\n{file_path}")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    root = tk.Tk()
    RTSPGui(root)
    root.mainloop()
