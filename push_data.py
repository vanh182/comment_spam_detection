
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 16:45:44 2024

@author: janac
"""

from confluent_kafka import Producer
import csv

# Thông tin cấu hình Kafka
kafka_bootstrap_servers = 'localhost:9092'
kafka_topic = 'sms'

# Hàm callback khi gửi message thành công
def delivery_report(err, msg):
    if err is not None:
        print('Gửi thất bại: {}'.format(err))
    else:
        print('Gửi thành công: Partition {} - Offset {}'.format(msg.partition(), msg.offset()))

# Khởi tạo Producer
producer = Producer({'bootstrap.servers': kafka_bootstrap_servers})

# Đọc dữ liệu từ file CSV và gửi lên Kafka
with open('sms_fb.csv', newline='') as csvfile:
# with open('data_realtime.csv', newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        # Gửi message với giá trị từ cột CSV
        producer.produce(kafka_topic, key=None, value=','.join(row), callback=delivery_report)

# Chờ cho tất cả các message được gửi đi
producer.flush()