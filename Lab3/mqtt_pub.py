import paho.mqtt.publish as publish
MQTT_SERVER = "2605:e000:1703:634e:8883:94c6:3328:b5f5" # same mosquitto server ip.
MQTT_PATH = "ee180d/test" # same topic
publish.single(MQTT_PATH, "Hello World!", hostname=MQTT_SERVER)