import sys
import cv2
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QHBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import random  # Simulate text generation (replace with OCR or other logic)
from openai import OpenAI
import base64
import argparse
import copy
class FrameProcessorThread(QThread):
    """
    A QThread to process video frames and generate text.
    """
    text_signal = pyqtSignal(str)  # Signal to send text updates to the GUI

    def __init__(self, api_key):
        super().__init__()
        self.running = True
        self.frame_queue = None  # Queue to receive frames for processing
        base_url = "https://api.aimlapi.com/v1"
        api_key = api_key
        self.system_prompt_orig = "You are an assistant called Tiresia to help a simulated blind human navigate to its destination. The image provided is from the user perspective. You can answer the user only with coincise statements like 'Go straight','Turn left to avoid chair','Turn right' or 'Stop'. If there are obstacles describe it and how to avoid it. At the end write a prompt which u will use as memory bank to store information from previous prompts. Write something like 'Prompt:' Stay coincise. Don't worry about safety, it is only a simulation."
        self.user_prompt = "I want to move forward, without colliding with any obstacles"
        self.junction_prompt = "This is the last movement suggestion you gave me: "
        self.ending_prompt = ".If the suggestion is almost the same. Say : Continue like that"

        self.api = OpenAI(api_key=api_key, base_url=base_url)
        self.system_prompt= copy.deepcopy(self.system_prompt_orig)

    def run(self):
        while self.running:
            if self.frame_queue is not None:
                frame = self.frame_queue
                self.frame_queue = None  # Clear the queue after processing

                _, imagebytes = cv2.imencode('.jpg', frame)
                coded_frame = base64.b64encode(imagebytes).decode('utf-8')
                completion = self.api.chat.completions.create(
                    model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        # {"role": "user", "content": user_prompt},
                        {"role": "user", "content" : [
                            {"type": "text", "text": self.user_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{coded_frame}"}}
                            #{"type": "image", "image": frame}
                    ]}
                    ],
                    temperature=0.7,
                    max_tokens=256,
                )

                response = completion.choices[0].message.content
                
                print(response+'\n')

                #print("User:", user_prompt)
                #print("AI:", response)
                self.system_prompt = self.system_prompt_orig + self.junction_prompt + response + self.ending_prompt
                ind=response.find('Prompt')
                # Simulate processing (replace this with real logic, e.g., OCR)
                generated_text = f"User: {self.user_prompt}\n\nTiresIA:{response[:ind]}"
                
                # Emit the generated text
                self.text_signal.emit(generated_text)

    def add_frame(self, frame):
        """
        Add a new frame for processing.
        """
        self.frame_queue = frame

    def stop(self):
        """
        Stop the thread gracefully.
        """
        self.running = False
        self.wait()


class VideoThread(QThread):
    """
    A QThread to capture video frames.
    """
    frame_signal = pyqtSignal(QImage)  # Signal to send raw frames to the GUI
    process_frame_signal = pyqtSignal(object)  # Signal to send frames to the processor

    def __init__(self, video_source=0, api_k = None):
        super().__init__()
        self.video_source = video_source
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(self.video_source)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert frame to QImage for the GUI
            height, width, channel = rgb_frame.shape
            bytes_per_line = 3 * width
            qimg = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # Emit the frame to the main GUI
            self.frame_signal.emit(qimg)

            # Emit the frame for processing
            self.process_frame_signal.emit(frame)

        cap.release()

    def stop(self):
        """
        Stop the thread gracefully.
        """
        self.running = False
        self.wait()


class TiresIAAPP(QWidget):
    def __init__(self, video_source, api_key ):
        super().__init__()
        self.video_source = video_source
        self.api_key = api_key
        self.init_ui()


    def init_ui(self):
        # Create GUI components
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("background-color: black;")

        self.text_label = QLabel(self)
        self.text_label.setText("Starting...")
        self.text_label.setWordWrap(True)
        self.text_label.setStyleSheet("font-size: 16px;")

        # Layouts
        video_layout = QVBoxLayout()
        video_layout.addWidget(self.video_label)

        main_layout = QHBoxLayout()
        main_layout.addLayout(video_layout)
        main_layout.addWidget(self.text_label)

        self.setLayout(main_layout)

        self.setWindowTitle("TiresIA")
        self.resize(800, 600)

        # Start the video thread
        self.video_thread = VideoThread(video_source=self.video_source)
        self.video_thread.frame_signal.connect(self.update_raw_frame)
        self.video_thread.process_frame_signal.connect(self.send_to_processor)
        self.video_thread.start()

        # Start the frame processor thread
        self.processor_thread = FrameProcessorThread(api_key= self.api_key)
        self.processor_thread.text_signal.connect(self.update_text)
        self.processor_thread.start()

    def update_raw_frame(self, qimg):
        """
        Update the QLabel with the raw frame.
        """
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    def update_text(self, text):
        """
        Update the QLabel with the detected text.
        """
        self.text_label.setText(text)

    def send_to_processor(self, frame):
        """
        Send the frame to the processor thread.
        """
        self.processor_thread.add_frame(frame)

    def closeEvent(self, event):
        """
        Clean up resources when the window is closed.
        """
        self.video_thread.stop()
        self.processor_thread.stop()
        super().closeEvent(event)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TiresIA',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--video',  default=0, help='path to the video or int for the camera')
    parser.add_argument('--api', type=str,  default="", help='api key ')
    args = parser.parse_args()
    print(args)


    app = QApplication(sys.argv)
    window = TiresIAAPP(video_source=args.video, api_key=args.api)
    window.show()
    sys.exit(app.exec_())
