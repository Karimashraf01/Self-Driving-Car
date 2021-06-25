#include <WiFi.h>
#include <WebSocketsServer.h>
#include <ESP32Servo.h>

int in;
int speed1 = 220;
int speed2 = 160;
// Motor A
const int motor1Pin1 = 27;
const int motor1Pin2 = 14;
const int enablem1Pin3 = 32;

//Motor B
const int motor2Pin1 = 25;
const int motor2Pin2 = 26;
const int enablem2Pin3 = 33;

// Setting PWM properties
const int freq = 30000;
const int pwmChannel = 2;
const int resolution = 8;
int dutyCycle = 200;

//buzzer
const int Buzzer = 2;


Servo myservo;  // create servo object to control a servo
// 16 servo objects can be created on the ESP32
int pos = 0;    // variable to store the servo position
// Recommended PWM GPIO pins on the ESP32 include 2,4,12-19,21-23,25-27,32-33 
int servoPin = 18;
//on board LED


String x = "hello";
int angle = 0;


// Constants
const char* ssid = "NETGEAR";
const char* password = "33334444";

// Globals
WebSocketsServer webSocket = WebSocketsServer(80);



void Wifi_notify(){
  for(int i = 0; i<5;i++)
  {
      digitalWrite (Buzzer, HIGH); //turn buzzer on
        delay(200);
        digitalWrite (Buzzer, LOW);  //turn buzzer off
        delay(200);
  }
}



void forward(){
        digitalWrite(motor1Pin1, LOW);
        digitalWrite(motor1Pin2, HIGH); 
        digitalWrite(motor2Pin1, LOW);
        digitalWrite(motor2Pin2, HIGH);
       
}



void backward(){
        digitalWrite(motor1Pin1, HIGH); 
        digitalWrite(motor1Pin2, LOW);
        digitalWrite(motor2Pin1, HIGH);
        digitalWrite(motor2Pin2, LOW);
       
}


void Stop(){
        digitalWrite(motor1Pin1, LOW); 
        digitalWrite(motor1Pin2, LOW); 
        digitalWrite(motor2Pin1, LOW);
        digitalWrite(motor2Pin2, LOW);        
}





// Called when receiving any WebSocket message
void onWebSocketEvent(uint8_t num,
                      WStype_t type,
                      uint8_t * payload,
                      size_t length) {

  // Figure out the type of WebSocket event
  switch(type) {

    // Client has disconnected
    case WStype_DISCONNECTED:
      Serial.printf("[%u] Disconnected!\n", num);
      break;

    // New client has connected
    case WStype_CONNECTED:
      {
        IPAddress ip = webSocket.remoteIP(num);
        Serial.printf("[%u] Connection from ", num);
        Serial.println(ip.toString());
        Wifi_notify();
       
      }
      break;

    // Echo text message back to client
    case WStype_TEXT:
      Serial.printf("[%u] Text: %s\n", num, payload);
      webSocket.sendTXT(num, payload);
      x  = (char*)payload;
       angle = x.toInt();
       in = angle;
       if (in > 160 && in < 250)
       speed1 = in; 
       if (in >= 45 && in <= 135)
        {
          	myservo.write(angle);    // tell servo to go to position in variable 'pos'
	         delay(15);             // waits 15ms for the servo to reach the position
            
        }
           switch (angle) {

             
       
             case 1:
             Stop();
                
             break; 

             case 2:
             forward();
            ledcWrite(pwmChannel, speed1);   
                         
             break;

             case 3:
             forward();
            ledcWrite(pwmChannel, 130);
             break;

     case 4:
             backward();
            ledcWrite(pwmChannel, 200);   
                         
             break;
             default:
             //backward();
             break;
           }





      break;



    // For everything else: do nothing
    case WStype_BIN:
    case WStype_ERROR:
    case WStype_FRAGMENT_TEXT_START:
    case WStype_FRAGMENT_BIN_START:
    case WStype_FRAGMENT:
    case WStype_FRAGMENT_FIN:
    default:
      break;
  }
}

void setup() {

  pinMode(motor1Pin1, OUTPUT);
  pinMode(motor1Pin2, OUTPUT);
  pinMode(enablem1Pin3, OUTPUT);
  pinMode(motor2Pin1, OUTPUT);
  pinMode(motor2Pin2, OUTPUT);
  pinMode(enablem2Pin3, OUTPUT);
  pinMode(Buzzer, OUTPUT);
    

  // configure LED PWM functionalitites
ledcSetup(pwmChannel, freq, resolution);
  
// attach the channel to the GPIO to be controlled
ledcAttachPin(enablem1Pin3, pwmChannel);
ledcAttachPin(enablem2Pin3, pwmChannel);



	// Allow allocation of all timers
	ESP32PWM::allocateTimer(0);
	ESP32PWM::allocateTimer(1);
	ESP32PWM::allocateTimer(2);
	ESP32PWM::allocateTimer(3);
	myservo.setPeriodHertz(50);    // standard 50 hz servo
	myservo.attach(servoPin, 500 , 2400); // attaches the servo on pin 18 to the servo object
	// using default min/max of 1000us and 2000us
	// different servos may require different min/max settings
	// for an accurate 0 to 180 sweep





  // Start Serial port
  Serial.begin(115200);

  // Connect to access point
  Serial.println("Connecting");
  WiFi.begin(ssid, password);
  while ( WiFi.status() != WL_CONNECTED ) {
    delay(500);
    Serial.print(".");
  }

  // Print our IP address
  Serial.println("Connected!");
  Serial.print("My IP address: ");
  Serial.println(WiFi.localIP());

  // Start WebSocket server and assign callback
  webSocket.begin();
  webSocket.onEvent(onWebSocketEvent);
}

void loop() {

  // Look for and handle WebSocket data
  webSocket.loop();
}
