import paho.mqtt.client as mqtt
MQTT_SERVER = "192.168.1.249" #mosquitto server ip address.
MQTT_PATH = "ee180d/test" # name of a topic.
# The callback for when the client receives a connect response from the server.
def on_connect(client, userdata, flags, rc):
  print("Connected with result code "+str(rc))
  # on_connect() means that if we lose the connection and reconnect then subscriptions will be renewed.
  client.subscribe(MQTT_PATH)

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
  print(msg.topic+" "+str(msg.payload))
  # more callbacks, etc
  # as an example, here you can save sensor data into a global variable.
  # alternatively, you can do simple operations in here.

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_SERVER, 1883, 60) # look into connect_async() for non-blocking.

# Blocking call that processes network traffic, dispatches callbacks and
# handles reconnecting.
# Other loop*() functions are available that give a threaded interface and a
# manual interface.

client.loop_forever() # look into loop_start() / loop_stop() for non-blocking.