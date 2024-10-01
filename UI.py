# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 13:22:37 2024

@author: janac
"""

import tkinter as tk
from confluent_kafka import Producer

# Thêm cấu hình Kafka Producer
kafka_bootstrap_servers = 'localhost:9092'
kafka_topic = 'sms'

# Khởi tạo Kafka Producer
producer_conf = {'bootstrap.servers': kafka_bootstrap_servers}
producer = Producer(producer_conf)

# Hàm gửi tin nhắn lên Kafka
def send_message():
    message = entry_message.get()
    producer.produce(kafka_topic, value=message)
    producer.flush()
    entry_message.delete(0, tk.END)  # Xóa nội dung của ô nhập sau khi gửi

# Khởi tạo UI
root = tk.Tk()
root.title("Kafka SMS Sender")

# Widget nhập tin nhắn
entry_message = tk.Entry(root, width=50)
entry_message.pack(pady=10)

# Widget nút gửi tin nhắn
button_send = tk.Button(root, text="Gửi Tin Nhắn", command=send_message)
button_send.pack()

# Chạy vòng lặp chính của Tkinter
root.mainloop()
