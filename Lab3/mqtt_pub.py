import paho.mqtt.publish as publish
MQTT_SERVER = "0.0.0.0" # same mosquitto server ip.
MQTT_PATH = "ee180d/test" # same topic
publish.single(MQTT_PATH, "Hello World!", hostname=MQTT_SERVER)