#include <WiFi.h>
#include <WiFiUdp.h>

 // WIFI details
  const char* ssid = "";
  const char* password = "";

// Laptop IP address
const char* laptopIpAddress = ""; 

// Using port 4210 for communication
const int udpPort; 

WiFiUDP udp;
int counter = 0;

// Defining the buzzer pin
int buzzerPin = 27;

// First FSR on GPIO 34
const int fsrPin1 = 34;

// Second FSR on GPIO 35  
const int fsrPin2 = 35;  

// Helper function to decode Wi-Fi connection errors
void printWiFiError(int status) {
  Serial.print("Wi-Fi Error Code [");
  Serial.print(status);
  Serial.print("]: ");
  
  switch (status) {
    case WL_NO_SSID_AVAIL: 
      Serial.println("SSID not found. Check network name."); 
      break;
    case WL_CONNECT_FAILED: 
      Serial.println("Connection failed. Check password."); 
      break;
    case WL_CONNECTION_LOST: 
      Serial.println("Connection lost. Signal might be too weak."); 
      break;
    case WL_DISCONNECTED: 
      Serial.println("Disconnected from network."); 
      break;
    case WL_IDLE_STATUS:
      Serial.println("Wi-Fi is changing states...");
      break;
    default: 
      Serial.println("Unknown Wi-Fi error."); 
      break;
  }
}

void setup() {
  Serial.begin(115200);
  
  // Configure the buzzer pin as an output
  pinMode(buzzerPin, OUTPUT);
  // Ensure the buzzer is off by default
  digitalWrite(buzzerPin, LOW);



  // Scan for networks
  Serial.println("Scanning WiFi networks...");
  int n = WiFi.scanNetworks();
  for (int i = 0; i < n; ++i) {
    Serial.print("Network found: ");
    Serial.println(WiFi.SSID(i));
  }
  
  // Attempt to connect to Wi-Fi
  WiFi.begin(ssid, password);
  Serial.println("Connecting to Wi-Fi");

  int attempts = 0;
  // Timeout after 10 seconds to prevent infinite hanging
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  // Check final connection status
  if (WiFi.status() != WL_CONNECTED) 
  {
    Serial.println("");
    printWiFiError(WiFi.status());
  } 
  else 
  {
    Serial.println("\nConnected to Wi-Fi!");
    Serial.print("ESP32 IP Address: ");
    Serial.println(WiFi.localIP());
    
    // Beeping to confirm wifi connection
    digitalWrite(buzzerPin, HIGH);
    delay(500);
    digitalWrite(buzzerPin, LOW);
  }
}

void sendPacket(int fsrReading1, int fsrReading2) {
   // Format a text packet
  String dataToSend = String(fsrReading1) + " " + String(fsrReading2);

  Serial.println(dataToSend);
  
  // Create the udp packet
  int beginStatus = udp.beginPacket(laptopIpAddress, udpPort);
  if (beginStatus == 0) 
  {
      Serial.println("UDP Error: Could not resolve IP address or port.");
  } 
  else 
  {
    // Write the data to the packet
    udp.print(dataToSend);
    
    // Dispatch the packet
    int endStatus = udp.endPacket();
    if (endStatus == 0) {
      Serial.println("UDP Error: Failed to send packet over the network.");
    } else {
      Serial.println("Sent: " + dataToSend);
    }
  }
  
  counter++;
  delay(1000);
}

void loop() {
  // Verify connection is still alive
  if (WiFi.status() != WL_CONNECTED) 
  {
    Serial.println("Warning: Wi-Fi dropped. Attempting to reconnect...");
    WiFi.disconnect();
    WiFi.reconnect();
    delay(5000); 
    return;
  }

  // Send the pressure sensor readings to the python script
  int fsr1Reading = analogRead(fsrPin1);
  int fsr2Reading = analogRead(fsrPin2);

  Serial.println(fsr1Reading);
  Serial.println(fsr2Reading);


  sendPacket(fsr1Reading, fsr2Reading);

 
}