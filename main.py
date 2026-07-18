import os
import cv2
import numpy as np
import requests
from threading import Thread
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image as KivyImage
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.checkbox import CheckBox
from kivy.clock import Clock
from kivy.utils import platform

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

PROCESS_EVERY = 5
MATCH_THRESH = 0.5
BLUR_KERNEL = (99, 99)
BLUR_FRAME_SKIP = 2

class FaceBlurApp(App):
    def build(self):
        self.title = "星澜小月月的人脸识别处理系统"
        if platform == 'android':
            from android.storage import app_storage_path
            self.work_dir = app_storage_path()
        else:
            self.work_dir = os.path.join(os.path.expanduser("~"), "FaceBlurApp")

        self.models_dir = os.path.join(self.work_dir, "models")
        self.faces_dir = os.path.join(self.work_dir, "faces")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.faces_dir, exist_ok=True)
        self.builtin_models_dir = os.path.join(os.path.dirname(__file__), "models")

        self.video_path = None
        self.keep_ids = set()
        self.all_faces = []
        self.is_processing = False

        layout = BoxLayout(orientation='vertical', padding=20, spacing=10)
        self.btn_select = Button(text="选择视频", size_hint=(1, 0.1))
        self.btn_select.bind(on_release=self.select_video)
        layout.add_widget(self.btn_select)
        self.btn_analyze = Button(text="分析视频中的人脸", size_hint=(1, 0.1), disabled=True)
        self.btn_analyze.bind(on_release=self.analyze_video)
        layout.add_widget(self.btn_analyze)
        self.scroll = ScrollView(size_hint=(1, 0.5))
        self.grid = GridLayout(cols=4, spacing=5, size_hint_y=None)
        self.grid.bind(minimum_height=self.grid.setter('height'))
        self.scroll.add_widget(self.grid)
        layout.add_widget(self.scroll)
        self.lbl_status = Label(text="", size_hint=(1, 0.1))
        layout.add_widget(self.lbl_status)
        self.btn_process = Button(text="开始打码", size_hint=(1, 0.1), disabled=True)
        self.btn_process.bind(on_release=self.start_blur)
        layout.add_widget(self.btn_process)
        return layout

    def select_video(self, instance):
        if platform == 'android':
            from android.storage import primary_external_storage_path
            from kivy.uix.popup import Popup
            from kivy.uix.filechooser import FileChooserListView
            content = FileChooserListView(path=primary_external_storage_path(), filters=['*.mp4', '*.mov', '*.avi', '*.mkv'])
            popup = Popup(title='选择视频', content=content, size_hint=(0.9, 0.9))
            content.bind(on_submit=lambda instance, selection, _: self._file_selected(selection, popup))
            popup.open()
        else:
            from plyer import filechooser
            filechooser.open_file(on_selection=self._file_selected)

    def _file_selected(self, selection, popup=None):
        if popup: popup.dismiss()
        if selection:
            self.video_path = selection[0]
            self.lbl_status.text = f"已选择: {os.path.basename(self.video_path)}"
            self.btn_analyze.disabled = False

    def _prepare_models(self):
        for fname in ["yunet.onnx", "sface.onnx"]:
            dst = os.path.join(self.models_dir, fname)
            if not os.path.exists(dst):
                src = os.path.join(self.builtin_models_dir, fname)
                if os.path.exists(src):
                    with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst: fdst.write(fsrc.read())
                else:
                    url = YUNET_URL if "yunet" in fname else SFACE_URL
                    try:
                        r = requests.get(url, timeout=30)
                        with open(dst, 'wb') as f: f.write(r.content)
                    except: return False
        return True

    def analyze_video(self, instance):
        if not self.video_path: return
        self.is_processing = True
        self.btn_analyze.disabled = True
        self.btn_select.disabled = True
        self.lbl_status.text = "正在准备模型..."
        Thread(target=self._run_analysis, daemon=True).start()

    def _run_analysis(self):
        if not self._prepare_models():
            self._enable_buttons(); return
        detector = cv2.FaceDetectorYN.create(os.path.join(self.models_dir, "yunet.onnx"), "", (320, 320))
        recognizer = cv2.FaceRecognizerSF.create(os.path.join(self.models_dir, "sface.onnx"), "")
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.lbl_status.text = "无法打开视频"; self._enable_buttons(); return
        known_features, face_thumbnails, frame_idx = [], [], 0
        self.all_faces = []
        for f in os.listdir(self.faces_dir): os.remove(os.path.join(self.faces_dir, f))
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            if frame_idx % PROCESS_EVERY != 0: continue
            h, w = frame.shape[:2]
            detector.setInputSize((w, h))
            _, faces = detector.detect(frame)
            if faces is None: continue
            for face in faces:
                aligned = recognizer.alignCrop(frame, face)
                feat = recognizer.feature(aligned)
                matched_id = None
                if known_features:
                    dists = [np.linalg.norm(feat - kf) for kf in known_features]
                    min_d = min(dists)
                    if min_d < MATCH_THRESH: matched_id = dists.index(min_d)
                if matched_id is not None:
                    known_features[matched_id] = (known_features[matched_id] + feat) / 2
                else:
                    new_id = len(known_features)
                    known_features.append(feat)
                    x, y, fw, fh = face[:4].astype(int)
                    face_img = frame[max(0,y):y+fh, max(0,x):x+fw]
                    thumb = os.path.join(self.faces_dir, f"id_{new_id}.jpg")
                    cv2.imwrite(thumb, face_img)
                    face_thumbnails.append(thumb)
            if frame_idx % 100 == 0:
                Clock.schedule_once(lambda dt, f=frame_idx: setattr(self.lbl_status, 'text', f"已分析 {f} 帧，发现 {len(known_features)} 人"))
        cap.release()
        self.all_faces = [(i, face_thumbnails[i]) for i in range(len(face_thumbnails))]
        Clock.schedule_once(self._show_face_grid)
        Clock.schedule_once(lambda dt: setattr(self.btn_process, 'disabled', False))
        Clock.schedule_once(lambda dt: setattr(self.lbl_status, 'text', f"分析完成，共 {len(self.all_faces)} 人"))
        self._enable_buttons()

    def _show_face_grid(self, *args):
        self.grid.clear_widgets(); self.keep_ids.clear()
        for fid, path in self.all_faces:
            box = BoxLayout(orientation='vertical', size_hint_y=None, height=150)
            box.add_widget(KivyImage(source=path, size_hint=(1,0.8)))
            chk = CheckBox(size_hint=(1,0.2), active=True)
            chk.bind(active=lambda cb, val, i=fid: self.on_face_check(i, val))
            box.add_widget(chk)
            self.grid.add_widget(box)
            self.keep_ids.add(fid)

    def on_face_check(self, fid, checked):
        if checked: self.keep_ids.add(fid)
        else: self.keep_ids.discard(fid)

    def start_blur(self, instance):
        if self.is_processing: return
        self.is_processing = True; self.btn_process.disabled = True
        self.lbl_status.text = "正在处理视频..."
        Thread(target=self._run_blur, daemon=True).start()

    def _run_blur(self):
        detector = cv2.FaceDetectorYN.create(os.path.join(self.models_dir, "yunet.onnx"), "", (320, 320))
        recognizer = cv2.FaceRecognizerSF.create(os.path.join(self.models_dir, "sface.onnx"), "")
        keep_feats = []
        for fid in self.keep_ids:
            img_path = os.path.join(self.faces_dir, f"id_{fid}.jpg")
            if not os.path.exists(img_path): continue
            img = cv2.imread(img_path)
            if img is None: continue
            h, w = img.shape[:2]
            detector.setInputSize((w, h))
            _, faces = detector.detect(img)
            if faces is not None:
                feat = recognizer.feature(recognizer.alignCrop(img, faces[0]))
                keep_feats.append(feat)
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_path = os.path.join(self.work_dir, "output_blurred.mp4")
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            idx += 1
            if idx % BLUR_FRAME_SKIP != 0:
                out.write(frame); continue
            detector.setInputSize((w, h))
            _, faces = detector.detect(frame)
            if faces is not None:
                for face in faces:
                    feat = recognizer.feature(recognizer.alignCrop(frame, face))
                    blur = True
                    if keep_feats:
                        dists = [np.linalg.norm(feat - kf) for kf in keep_feats]
                        if min(dists) < MATCH_THRESH: blur = False
                    if blur:
                        x, y, fw, fh = face[:4].astype(int)
                        x, y = max(0,x), max(0,y)
                        roi = frame[y:y+fh, x:x+fw]
                        if roi.size > 0:
                            frame[y:y+fh, x:x+fw] = cv2.GaussianBlur(roi, BLUR_KERNEL, 30)
            out.write(frame)
            if idx % 50 == 0:
                Clock.schedule_once(lambda dt, a=idx, b=total: setattr(self.lbl_status, 'text', f"处理进度: {a}/{b}"))
        cap.release(); out.release()
        Clock.schedule_once(lambda dt: setattr(self.lbl_status, 'text', f"完成！输出: {out_path}"))
        self._enable_buttons()

    def _enable_buttons(self):
        Clock.schedule_once(lambda dt: setattr(self.btn_select, 'disabled', False))
        Clock.schedule_once(lambda dt: setattr(self.btn_analyze, 'disabled', False))
        self.is_processing = False

if __name__ == '__main__':
    FaceBlurApp().run()
