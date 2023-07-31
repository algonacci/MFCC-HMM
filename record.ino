#include <WiFi.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>
#include <I2S.h>
#include <vector>

const char* ssid = "YourWiFiSSID";
const char* password = "YourWiFiPassword";
const char* serverAddress = "YourServerAddress"; // Replace with your server's IP or domain
const int serverPort = 81;

WebSocketsClient webSocket;
WiFiClient wifiClient;

void onWebSocketEvent(WStype_t type, uint8_t* payload, size_t length) {
  // WebSocket event handling code (if needed)
}

void setup() {
  Serial.begin(115200);
  delay(2000);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(100);
  }

  webSocket.begin(serverAddress, serverPort, "/");
  webSocket.onEvent(onWebSocketEvent);

  // I2S Configuration for audio recording
  I2S.begin(I2S_PHILIPS_MODE, 16000, 16); // Adjust settings as per your hardware

  // Rest of your setup code...
}

void loop() {
  webSocket.loop();

  // Record audio and store in a buffer
  const size_t bufferSize = 1024;
  std::vector<int> audioBuffer(bufferSize);
  size_t bytesRead = I2S.read(audioBuffer.data(), bufferSize);

  if (bytesRead > 0) {
    // Code to extract MFCC features from the audio buffer
    std::vector<float> mfccFeatures = extract_mfcc_delta_delta(audioBuffer);

    // Convert MFCC features to JSON for sending via WebSocket
    DynamicJsonDocument jsonDocument(1024);
    JsonArray jsonArray = jsonDocument.to<JsonArray>();
    for (size_t i = 0; i < mfccFeatures.size(); i++) {
      jsonArray.add(mfccFeatures[i]);
    }
    String jsonData;
    serializeJson(jsonArray, jsonData);

    // Send MFCC features to the server via WebSocket
    webSocket.sendTXT(jsonData);
  }

  // Rest of your loop code...
}
