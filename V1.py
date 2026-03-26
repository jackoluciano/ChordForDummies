import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

# valid notes
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
INTERVAL_NAMES = {1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V', 6: 'VI', 7: 'VII'}

# map C ke 0, D ke 2, E ke 4 dst berdasarkan posisi di indeks
INTERVAL_STEP = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11}

# map koneksi antar titik (edge dari graf)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Jempol
    (0, 5), (5, 6), (6, 7), (7, 8),        # Telunjuk
    (5, 9), (9, 10), (10, 11), (11, 12),   # Jari Tengah
    (9, 13), (13, 14), (14, 15), (15, 16), # Jari Manis
    (13, 17), (17, 18), (18, 19), (19, 20),# Kelingking
    (0, 17)                                # Telapak bawah (Wrist ke kelingking)
]

# tampilan
def draw_custom_landmarks(image, landmarks):
    h, w, _ = image.shape
    points = []
    
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        points.append((cx, cy))
        cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        
    for connection in HAND_CONNECTIONS:
        pt1 = points[connection[0]]
        pt2 = points[connection[1]]
        cv2.line(image, pt1, pt2, (0, 255, 0), 2)

class FingerTracker:


    def __init__(self):
        self.root_idx = 0  # default C
        self.fist_start_time = None
        self.middle_was_low = True
        self.pinky_was_low = True
        self.y_threshold = 0.05 # jump (terjadi perubahan posisi y signifikan) sensitivity untuk transpose

    def get_chord(self, interval, is_sharp):
        if is_sharp and interval==3:
            return "D#" # karena III# ga ada, maka ditulis saja II# karena suka salah notasi
        if is_sharp and interval==7:
            return "A#" # karena VII# ga ada, maka ditulis saja VI# karena suka salah notasi
        
        step = INTERVAL_STEP.get(interval, 0)
        if is_sharp: step += 1 # dari mapping interval ke indeks ditambah 1 (misal C=0 jadi C#=1)
            
        chord_idx = (self.root_idx + step) % 12 # nada itu sirkular jangan lupa
        return NOTES[chord_idx]

    def process_logic(self, lm):
        # definisi pointing down adalah ketika lm[9] (tertinggi di antara semua knuckles) berada di bawah pangkal tangan
        is_pointing_down = lm[9].y > lm[0].y

        # definisi jari terbuka adalah ujung jari (tip) lebih jauh dari sendi kedua dari pangkal (pip) ke pangkal
        def is_extended(tip, pip):
            dist_tip = (lm[tip].x - lm[0].x)**2 + (lm[tip].y - lm[0].y)**2
            dist_pip = (lm[pip].x - lm[0].x)**2 + (lm[pip].y - lm[0].y)**2
            return dist_tip > dist_pip

        # definisi jempol terbuka
        def is_thumb_open():
            if is_pointing_down:
                # tangan ke bawah (sharp): pip jari tengah -> tip jempol : pip jari tengah -> pip jempol
                dist_tip = (lm[4].x - lm[10].x)**2 + (lm[4].y - lm[10].y)**2
                dist_ip = (lm[3].x - lm[10].x)**2 + (lm[3].y - lm[10].y)**2 
            else:
                # tangan ke atas (normal) bandingkan:
                # 1. pilih nilai terdekat antara pangkal tangan ke tip jempol dan pip tengah ke tip jempol
                # 2. pilih nilai terdekat antara pangkal tangan ke pip jempol dan pip tengah ke pip jempol
                dist_tip = min((lm[4].x - lm[0].x)**2 + (lm[4].y - lm[0].y)**2, (lm[4].x - lm[11].x)**2 + (lm[4].y - lm[11].y)**2)
                dist_ip = min((lm[3].x - lm[0].x)**2 + (lm[3].y - lm[0].y)**2, (lm[3].x - lm[11].x)**2 + (lm[3].y - lm[11].y)**2)
            
            return dist_tip > dist_ip
        
        up = {
            'T': is_thumb_open(),       # jari jempol
            'I': is_extended(8, 6),     # jari telunjuk
            'M': is_extended(12, 10),   # jari tengah
            'R': is_extended(16, 14),   # jari manis
            'P': is_extended(20, 18)    # jari kelingking
        }

        # reset nada dasar ke default setelah 10 detik menggenggam tangan
        all_closed = not any([up['I'], up['M'], up['R'], up['P']])
        if all_closed:
            if self.fist_start_time is None: self.fist_start_time = time.time()
            elif time.time() - self.fist_start_time > 10: self.root_idx = 0
        else:
            self.fist_start_time = None

        # peak detection (ketika jari tengah/kelingking jump) untuk transpose
        if not is_pointing_down:
            # jari tengah (+1)
            if up['M'] and not (up['I'] or up['R'] or up['P']):
                if lm[12].y < (lm[10].y - self.y_threshold):
                    if self.middle_was_low:
                        self.root_idx = (self.root_idx + 1) % 12
                        self.middle_was_low = False
                else:
                    self.middle_was_low = True
            else:
                self.middle_was_low = True
            
            # kelingking (-1)
            if up['P'] and not (up['I'] or up['M'] or up['R']):
                if lm[20].y < (lm[18].y - self.y_threshold):
                    if self.pinky_was_low:
                        self.root_idx = (self.root_idx - 1) % 12
                        self.pinky_was_low = False
                else:
                    self.pinky_was_low = True
            else:
                self.pinky_was_low = True
        else:
            self.middle_was_low = True
            self.pinky_was_low = True

        # map posisi jari dengan interval chord
        interval = 0
        sharp = is_pointing_down 

        if up['I'] and up['T'] and not up['M']: interval = 7                                # VII = telunjuk jempol
        elif up['I'] and not up['M'] and not up['T']: interval = 1                          # I= telunjuk
        elif up['I'] and up['M'] and not up['R']: interval = 2                              # II = telunjuk tengah
        elif up['I'] and up['M'] and up['R'] and not up['P']: interval = 3                  # III = telunjuk tengah manis
        elif up['I'] and up['M'] and up['R'] and up['P'] and not up['T']: interval = 4      # IV = telunjuk tengah manis kelingkung
        elif up['I'] and up['M'] and up['R'] and up['P'] and up['T']: interval = 5          # V = semua jari
        elif up['T'] and not up['I']: interval = 6                                          # VI = jempol doang

        if interval > 0:
            current_chord = self.get_chord(interval, sharp)
            return f"Key: {NOTES[self.root_idx]} | Interval: {INTERVAL_NAMES[interval]}{'#' if sharp else ''}", current_chord
        
        return f"Key: {NOTES[self.root_idx]} | Ready...", ""

# setup mediapipe api dengan file .task
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# main loop
cap = cv2.VideoCapture(0)
tracker = FingerTracker()
window_name = 'chord for dummies'

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    detection_result = detector.detect(mp_image)
    
    display_text = "No Hand Detected"
    chord = ""

    if detection_result.hand_landmarks:
        hand_landmarks = detection_result.hand_landmarks[0]
        display_text, chord = tracker.process_logic(hand_landmarks)
        draw_custom_landmarks(frame, hand_landmarks)

    cv2.putText(frame, display_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f"CHORD: {chord}", (10, 150), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)

    cv2.imshow(window_name, frame)
    
    if cv2.waitKey(1) & 0xFF == 27: break
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: break

cap.release()
detector.close()
cv2.destroyAllWindows()