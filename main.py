import os
from kivy.app import App
import pafy
import cv2
import pandas as pd

from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.stacklayout import StackLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from matplotlib import pyplot as plt
from kivy.uix.videoplayer import VideoPlayer

from detect import detect

class MainLayout(BoxLayout):
	"""
	Define class MainLayout
	"""

	def __init__(self):

		super(MainLayout, self).__init__()

		if not os.path.isfile('data.csv'):
			self.data = pd.DataFrame({
				"date_time": [],
				"person": [],
				"car": []
			})
		else:
			self.data = pd.read_csv('data.csv', index_col=0)
			self.data["date_time"] = pd.to_datetime(self.data.date_time)

		# url of the video
		self.url = "https://www.youtube.com/watch?v=AdUw5RdyZxI"

		# creating pafy object of the video
		self.video = pafy.new(self.url)

		# getting best stream
		self.best = self.video.getbest()		

		self.vp = VideoPlayer(source=self.best.url,
			thumbnail="iaschool.jpg",
            state='play',
            volume=0,
		)
		
		aside = BoxLayout(orientation='vertical', size_hint=(.4, 1))

		self.result_label = Label(size_hint=(1, .4))
		self.result_label.text = "_"

		self.image = Image()
		ia_school = cv2.imread("iaschool.jpg")
		buffer = cv2.flip(ia_school, 0).tostring()
		texture = Texture.create(size=(ia_school.shape[1],ia_school.shape[0]), colorfmt='bgr')
		texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
		self.image.texture = texture

		self.plt_layout = StackLayout()
		self.plt_layout.add_widget(FigureCanvasKivyAgg(plt.gcf()))

		aside.add_widget(self.vp)
		aside.add_widget(self.result_label)
		aside.add_widget(self.plt_layout)

		self.add_widget(aside)
		self.add_widget(self.image)

		Clock.schedule_interval(self.do_something, 15)

	def do_something(self, *args):
		capture = cv2.VideoCapture(self.video.getbest().url)
		ret, frame = capture.read()
		self.image_frame = frame
		# self.save_sample_picture()
		frame, labels = detect(frame)
		self.result_label.text = self.format_counter_to_label(labels)

		date_time = pd.Timestamp.now().replace(microsecond=0)
		person = labels["person"] if "person" in labels else 0
		car = labels["car"] if "car" in labels else 0
		self.data.loc[date_time] = [date_time, person, car]

		# self.data.append(new_row, ignore_index=True)

		x = self.data["date_time"]
		y1 = self.data["person"]
		y2 = self.data["car"]
		plt.clf()
		plt.plot(x, y1, label="person", color="blue")
		plt.plot(x, y2, label="car", color="red")
		plt.xlabel("time")
		plt.xticks(rotation = 45)
		plt.legend()
		
		self.plt_layout.clear_widgets()
		self.plt_layout.add_widget(FigureCanvasKivyAgg(plt.gcf()))

		buffer = cv2.flip(frame, 0).tostring()
		texture = Texture.create(size=(frame.shape[1],frame.shape[0]), colorfmt='bgr')
		texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
		self.image.texture = texture

	def save_sample_picture(self, *args):
		image_name = "sample_image.png"
		cv2.imwrite(image_name, self.image_frame)
	
	def format_counter_to_label(self, counter):
		formatted = ""
		for key in counter:
			formatted = f"{formatted} {key}: {counter[key]};"
		return formatted

	def save_csv(self):
		self.data.to_csv('data.csv', header='column_names')
	

# création de la classe ProjetIAApp
class ProjetIAApp(App):

	def build(self):
		self.main = MainLayout()
		Window.bind(on_request_close=self.main.save_csv)
		return self.main


# Exécution de l'application
if __name__ == '__main__':
	projetIAApp = ProjetIAApp()
	projetIAApp.run()
