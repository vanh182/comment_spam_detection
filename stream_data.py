import tkinter as tk
import tkinter.filedialog
from tkinter import font, ttk, messagebox
from confluent_kafka import Producer, KafkaException, Consumer, KafkaError
import threading
from colorama import Fore, Style
import time
import sms_model
from datetime import datetime
import csv

# Thông tin cấu hình Kafka
kafka_bootstrap_servers = 'localhost:9092'
kafka_topic = 'test'

# Màu sắc tin nhắn spam
sms_color = Fore.WHITE

# Danh sách tin nhắn
messages = []

# Hàm lấy tin nhắn từ Kafka
def consume_messages():
    global sms_color
    consumer_conf = {
        'bootstrap.servers': kafka_bootstrap_servers,
        'group.id': 'ten_consumer_group',
        'auto.offset.reset': 'earliest'
    }
    consumer = Consumer(consumer_conf)
    consumer.subscribe([kafka_topic])

    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    print(msg.error())
                    break

            result = sms_model.get_input(msg.value().decode('utf-8'))
            if result == 'spam':
                sms_color = Fore.RED
            else:
                sms_color = Fore.WHITE
             #real time
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            message = f"{result}-{current_time}: {msg.value().decode('utf-8')}"
            messages.append(message)
            # time.sleep(1)
            root.after(1, update_ui)  # Gọi update_ui trong main thread

    except KeyboardInterrupt:
        pass

    finally:
        consumer.close()

# Hàm cập nhật giao diện
def update_ui():
    if messages:
        message_text.config(state=tk.NORMAL)
        message_text.insert(tk.END, messages[-1] + "\n")
        message_text.see(tk.END)  # Cuộn xuống cuối cùng
        message_text.config(state=tk.DISABLED)

# Hàm gửi bình luận lên Kafka
def send_comment(*args):
    comment = comment_entry.get()
    if comment:
        try:
            producer.produce(kafka_topic, value=comment.encode('utf-8'))
            producer.flush()
            comment_entry.delete(0, tk.END)
        except KafkaException as e:
            messagebox.showerror("Lỗi Kafka", f"Lỗi khi gửi bình luận: {str(e)}")
    else:
        messagebox.showwarning("Lỗi", "Vui lòng nhập bình luận.")
# Hàm mở file và đẩy dữ liệu lên server
def open_file():
    file_path = tkinter.filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        # Đẩy dữ liệu từ file CSV lên server
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                try:
                    producer.produce(kafka_topic, key=None, value=','.join(row).encode('utf-8'))
                    producer.flush()
                except KafkaException as e:
                    messagebox.showerror("Lỗi Kafka", f"Lỗi khi gửi dữ liệu từ file: {str(e)}")
        messagebox.showinfo("Thành công", "Đã gửi dữ liệu từ file thành công!")

# Khởi tạo Kafka Producer
producer_conf = {'bootstrap.servers': kafka_bootstrap_servers}
producer = Producer(producer_conf)

# Tạo giao diện
root = tk.Tk()
root.title("Kafka Server")

# Thay đổi font chữ
font_style = font.Font(family="Arial", size=12)

message_text = tk.Text(root, wrap=tk.WORD, state=tk.DISABLED, font=font_style)
message_text.pack(expand=True, fill='both')

# Thêm ô nhập bình luận
comment_entry = tk.Entry(root, font=font_style, width=50)
comment_entry.pack(pady=5)

# Bắt sự kiện Enter cho ô nhập bình luận
comment_entry.bind('<Return>', send_comment)

# Nút thêm bình luận
send_button = ttk.Button(root, text="Send message", command=send_comment)
send_button.pack(pady=5)

# Nút đẩy data lên kafka server
open_button = ttk.Button(root, text="Push data", command=open_file)
open_button.pack(pady=5)

# Bắt đầu thread lấy tin nhắn từ Kafka
kafka_thread = threading.Thread(target=consume_messages)
kafka_thread.start()

# Main loop của giao diện
root.mainloop()

# Join thread để đảm bảo chương trình kết thúc đúng cách
kafka_thread.join()
